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

# =====================================================================
# GRADIO/ACCELERATE FP16 gradient override patch
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


def get_attention_implementation():
    """Return best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        # Fallback to PyTorch's native Scaled Dot Product Attention (SDPA)
        return "sdpa"


target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)  # Recommended range [1e-6, 2e-6]
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    # Select dynamic precision based on real hardware capability (Ampere+ Score >= 8.0 required for BF16)
    device_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    target_dtype = torch.bfloat16 if device_cap[0] >= 8 else torch.float16
    print(f"Selected hardware dtype: {target_dtype} (Turing/Volta compatible)")

    # Setup logging directory for TensorBoard to prevent accelerate value errors
    logging_dir = os.path.join(args.output_model_path, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=4, 
        mixed_precision="bf16" if target_dtype == torch.bfloat16 else "fp16", 
        log_with="tensorboard",
        project_dir=logging_dir
    )

    MODEL_PATH = args.init_model_path

    # Detect attention implementation
    attn_implementation = get_attention_implementation()
    print(f"Using attention implementation: {attn_implementation}")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=target_dtype,
        attn_implementation=attn_implementation,
    )

    # Force enable gradient checkpointing on the model to reduce activation VRAM bounds
    if hasattr(qwen3tts.model, "model") and hasattr(qwen3tts.model.model, "gradient_checkpointing_enable"):
        qwen3tts.model.model.gradient_checkpointing_enable()
        print("🚀 Enabled gradient checkpointing on underlying model.")
    elif hasattr(qwen3tts.model, "gradient_checkpointing_enable"):
        qwen3tts.model.gradient_checkpointing_enable()
        print("🚀 Enabled gradient checkpointing on model.")

    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # Setup optimizer (fallback to standard AdamW if bitsandbytes is missing)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
        print("🚀 Loaded bitsandbytes PagedAdamW8bit optimizer successfully.")
    except ImportError:
        optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
        print("⚠️ bitsandbytes not found, falling back to standard AdamW.")

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # Bug Fix 1: Apply the missing text projection layer on text embeddings
                input_text_embedding = (
                    model.talker.text_projection(
                        model.talker.model.text_embedding(input_text_ids)
                    ) * text_embedding_mask
                )
                
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
                    output_hidden_states=True
                )

                # Fix: Resolve slice-alignment mismatch for the sub-talker outputs
                hidden_states = outputs.hidden_states[0][-1]
                target_codec_mask = codec_mask[:, 1:]
                talker_hidden_states = hidden_states[:, :-1, :][target_codec_mask]
                talker_codec_ids = codec_ids[:, 1:][target_codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                # Balanced SFT loss: scale the sub-talker by 0.3 so it does not overpower the primary talker
                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                # Fix: Wrap optimizer update step inside the sync_gradients block to keep accumulation statistics intact
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            
            # Resolve the actual local model cache directory using snapshot_download
            from huggingface_hub import snapshot_download
            if os.path.isdir(MODEL_PATH):
                model_cache_path = MODEL_PATH
            else:
                model_cache_path = snapshot_download(MODEL_PATH)
                
            shutil.copytree(model_cache_path, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(model_cache_path, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {
                k: v.detach().to("cpu").to(torch.bfloat16) 
                for k, v in unwrapped_model.state_dict().items()
            }

            # Fix: Comment out the dropping of speaker_encoder weights to retain in-context zero-shot capabilities during inference
            # drop_prefix = "speaker_encoder"
            # keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            # for k in keys_to_drop:
            #     del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

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
