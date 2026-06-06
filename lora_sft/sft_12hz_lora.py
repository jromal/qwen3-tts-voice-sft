# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
except ImportError as exc:
    raise SystemExit(
        "peft is required for LoRA fine-tuning. Install it with: pip install peft"
    ) from exc

# =====================================================================
# WATCH THE BOOK: GradScaler FP16 gradient override patch
# =====================================================================
try:
    import torch.amp.grad_scaler
    _orig_unscale = torch.amp.grad_scaler.GradScaler._unscale_grads_
    torch.amp.grad_scaler.GradScaler._unscale_grads_ = lambda self, opt, inv, inf, allow=False: _orig_unscale(self, opt, inv, inf, True)
except Exception:
    pass

try:
    import torch.cuda.amp.grad_scaler
    _orig_unscale_cuda = torch.cuda.amp.grad_scaler.GradScaler._unscale_grads_
    torch.cuda.amp.grad_scaler.GradScaler._unscale_grads_ = lambda self, opt, inv, inf, allow=False: _orig_unscale_cuda(self, opt, inv, inf, True)
except Exception:
    pass
# =====================================================================


# WATCH THE BOOK: Dynamic attention fallback detection
def get_attention_implementation():
    """Return best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        # Fallback to PyTorch's native Scaled Dot Product Attention (SDPA)
        return "sdpa"


target_speaker_embedding = None


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def compute_loss(model, batch):
    global target_speaker_embedding

    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    with torch.no_grad():
        speaker_embedding = model.speaker_encoder(
            ref_mels.to(dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
        ).detach()

    if model.training and target_speaker_embedding is None:
        target_speaker_embedding = speaker_embedding.detach().to("cpu")

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    # Bug Fix 1: Apply the missing text projection layer on text embeddings
    input_text_embedding = model.talker.model.text_embedding(input_text_ids)
    if hasattr(model.talker, 'text_projection'):
        input_text_embedding = model.talker.text_projection(input_text_embedding)
    input_text_embedding = input_text_embedding * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_codec_embedding[:, 6, :] = speaker_embedding

    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    # Bug Fix 2: Avoid manual shift to prevent double label-shifting in ForCausalLMLoss
    outputs = model.talker(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        labels=codec_0_labels,
        output_hidden_states=True,
    )

    # Adjust hidden states slicing to align correctly with unshifted inputs
    hidden_states = outputs.hidden_states[0][-1][:, :-1, :]
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]

    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden_states
    )

    # Balanced SFT loss: scale the sub-talker by 0.3 so it does not overpower the primary talker (which predicts EOS)
    return outputs.loss + 0.3 * sub_talker_loss


def evaluate(model, dataloader, accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            loss = compute_loss(model, batch)
            gathered = accelerator.gather_for_metrics(loss.detach())
            if gathered.ndim == 0:
                gathered = gathered.unsqueeze(0)
            losses.append(gathered)
    if not losses:
        return None
    val_loss = torch.cat(losses).mean().item()
    return val_loss


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output_lora")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--resume_adapter", type=str, default=None)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn_implementation", type=str, default=get_attention_implementation())  # Fixed: Dynamically fallback to SDPA if flash-attn is missing
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--val_jsonl", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=1)
    args = parser.parse_args()

    # Select dynamic precision based on real hardware capability (Ampere+ Score >= 8.0 required for BF16)
    device_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    target_dtype = torch.bfloat16 if device_cap[0] >= 8 else torch.float16
    print(f"Selected hardware dtype: {target_dtype} (Turing/Volta compatible)")

    # Setup logging directory for TensorBoard to prevent accelerate value errors
    logging_dir = os.path.join(args.output_model_path, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else ("bf16" if target_dtype == torch.bfloat16 else "fp16"),
        log_with="tensorboard",
        project_dir=logging_dir
    )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=target_dtype,
        attn_implementation=args.attn_implementation,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    if args.resume_adapter:
        model = PeftModel.from_pretrained(
            qwen3tts.model,
            args.resume_adapter,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=_parse_list(args.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(qwen3tts.model, lora_config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    train_data = [json.loads(line) for line in open(args.train_jsonl).readlines()]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    val_dataloader = None
    if args.val_jsonl:
        val_data = [json.loads(line) for line in open(args.val_jsonl)]
        val_dataset = TTSDataset(val_data, qwen3tts.processor, config)
        eval_bs = args.eval_batch_size or args.batch_size
        val_dataloader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False, collate_fn=val_dataset.collate_fn)

    # Setup optimizer (fallback to standard AdamW if bitsandbytes is missing)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("🚀 Loaded bitsandbytes PagedAdamW8bit optimizer successfully.")
    except ImportError:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("⚠️ bitsandbytes not found, falling back to standard AdamW.")

    if val_dataloader is not None:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

    num_epochs = args.num_epochs
    model.train()

    for local_epoch in range(num_epochs):
        epoch = args.start_epoch + local_epoch
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = compute_loss(model, batch)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if val_dataloader is not None and (local_epoch + 1) % args.eval_every == 0:
            val_loss = evaluate(model, val_dataloader, accelerator)
            if accelerator.is_main_process and val_loss is not None:
                accelerator.print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
            model.train()

        if accelerator.is_main_process and (local_epoch + 1) % args.save_every == 0:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, safe_serialization=True)

            # Resolve the actual local model cache directory using snapshot_download
            from huggingface_hub import snapshot_download
            if os.path.isdir(MODEL_PATH):
                model_cache_path = MODEL_PATH
            else:
                model_cache_path = snapshot_download(MODEL_PATH)

            input_config_file = os.path.join(model_cache_path, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config.setdefault("spk_id", {})
            talker_config.setdefault("spk_is_dialect", {})
            talker_config["spk_id"][args.speaker_name] = 3000
            talker_config["spk_is_dialect"][args.speaker_name] = False
            config_dict["talker_config"] = talker_config

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            if target_speaker_embedding is not None:
                save_file(
                    {"target_speaker_embedding": target_speaker_embedding[0]},
                    os.path.join(output_dir, "speaker_embedding.safetensors"),
                )

            # Checkpoint Pruning using stable numerical sort by epoch index
            output_path_obj = Path(args.output_model_path)
            saved_checkpoints = sorted(
                output_path_obj.glob("checkpoint-epoch-*"),
                key=lambda x: int(x.name.split("-epoch-")[-1]) if "-epoch-" in x.name else -1
            )
            max_keep = 1
            if len(saved_checkpoints) > max_keep:
                for old_ckpt in saved_checkpoints[:-max_keep]:
                    try:
                        shutil.rmtree(old_ckpt)
                        print(f"🧹 Pruned older checkpoint to conserve disk: {old_ckpt.name}")
                    except Exception as e:
                        print(f"⚠️ Failed to delete older checkpoint {old_ckpt.name}: {e}")


if __name__ == "__main__":
    train()
