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
    parser.add_argument('--init_model_path', type=str, default='Qwen/Qwen3-TTS-12Hz-1.7B-Base')
    parser.add_argument('--output_model_path', type=str, default='output')
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-6)  # Recommended range [1e-6, 2e-6]
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--speaker_name', type=str, default='speaker_test')
    args = parser.parse_args()

    # Select dynamic precision based on real hardware capability (Ampere+ Score >= 8.0 required for BF16)
    device_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    target_dtype = torch.bfloat16 if device_cap[0] >= 8 else torch.float16

    # Setup logging directory for TensorBoard to prevent accelerate value errors
    logging_dir = os.path.join(args.output_model_path, 'logs')
    os.makedirs(logging_dir, exist_ok=True)

    # Hardcoded gradient accumulation value matching SFT setup
    grad_accum_steps = 4

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps, 
        mixed_precision='bf16' if target_dtype == torch.bfloat16 else 'fp16', 
        log_with='tensorboard',
        project_dir=logging_dir
    )

    MODEL_PATH = args.init_model_path

    # Detect attention implementation
    attn_implementation = get_attention_implementation()

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=target_dtype,
        attn_implementation=attn_implementation,
    )

    # ── DYNAMIC SPEAKER ENCODER RESTORATION (Initializes with base config to prevent dimension mismatch) ──
    if not hasattr(qwen3tts.model, 'speaker_encoder') or qwen3tts.model.speaker_encoder is None:
        if accelerator.is_main_process:
            print('📥 CustomVoice base detected. Dynamically restoring Speaker Encoder from Base model...', flush=True)
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        try:
            # Target correct base config dynamically to match the model scale scale parameters
            base_model_id = 'Qwen/Qwen3-TTS-12Hz-1.7B-Base' if '1.7B' in MODEL_PATH else 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
            base_config = AutoConfig.from_pretrained(base_model_id)
            
            # Instantiates with 2048-dim configurations to avoid CustomVoice 1024-dim mismatch failures
            qwen3tts.model.speaker_encoder = Qwen3TTSSpeakerEncoder(base_config.speaker_encoder_config)
            
            base_model_file = hf_hub_download(repo_id=base_model_id, filename='model.safetensors')
            base_state = load_file(base_model_file)
            encoder_state = {k.replace('speaker_encoder.', ''): v for k, v in base_state.items() if k.startswith('speaker_encoder.')}
            qwen3tts.model.speaker_encoder.load_state_dict(encoder_state)
            qwen3tts.model.speaker_encoder.to(device=qwen3tts.model.device, dtype=qwen3tts.model.dtype)
            if accelerator.is_main_process:
                print('✅ Speaker Encoder successfully restored for training.')
        except Exception as e:
            if accelerator.is_main_process:
                print(f'⚠️ Failed to restore Speaker Encoder: {e}', flush=True)

    # Force enable gradient checkpointing on the model to reduce activation VRAM bounds
    if hasattr(qwen3tts.model, 'model') and hasattr(qwen3tts.model.model, 'gradient_checkpointing_enable'):
        qwen3tts.model.model.gradient_checkpointing_enable()
    elif hasattr(qwen3tts.model, 'gradient_checkpointing_enable'):
        qwen3tts.model.gradient_checkpointing_enable()

    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # ── component-level freeze logic (Fixes scrambled voice) ──
    # Freeze the speaker_encoder and acoustic vocoder, limiting optimization to model.talker
    if hasattr(qwen3tts.model, 'speaker_encoder') and qwen3tts.model.speaker_encoder is not None:
        for p in qwen3tts.model.speaker_encoder.parameters():
            p.requires_grad = False

    # Mark all top-level parameters as frozen first
    for p in qwen3tts.model.parameters():
        if not any(name.startswith('talker') for name, _ in qwen3tts.model.named_parameters()):
            p.requires_grad = False

    # Explicitly unfreeze only the talker component
    for name, p in qwen3tts.model.named_parameters():
        if name.startswith('talker'):
            p.requires_grad = True

    # Setup optimizer strictly on talker parameters (fallback to standard AdamW if bitsandbytes is missing)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(qwen3tts.model.talker.parameters(), lr=args.lr, weight_decay=0.01)
        optimizer_name = "PagedAdamW8bit (bitsandbytes)"
    except ImportError:
        optimizer = AdamW(qwen3tts.model.talker.parameters(), lr=args.lr, weight_decay=0.01)
        optimizer_name = "Standard AdamW (PyTorch)"

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs

    # Calculate dynamic execution variables for the metadata block
    total_samples = len(dataset)
    steps_per_epoch = len(train_dataloader)
    total_raw_steps = steps_per_epoch * num_epochs
    effective_opt_steps = total_raw_steps // grad_accum_steps

    # ── REGISTERED TRAINING DASHBOARD (Just Before the First Epoch) ──
    if accelerator.is_main_process:
        print('\n' + '='*70)
        print('🎙️  QWEN3-TTS FULL-PARAMETER SFT TRAINING ENGINE RUNTIME')
        print('='*70)
        print(f'👤 Target Speaker ID         : {args.speaker_name}')
        print(f'🤖 Initial Model Checkpoint  : {args.init_model_path}')
        print(f'📂 Output Checkpoint Path    : {args.output_model_path}')
        print(f'📊 Dataset Size              : {total_samples} samples')
        print(f'⏱️  Training Epochs Limit     : {num_epochs}')
        print(f'🔄 Steps per Training Epoch  : {steps_per_epoch}')
        print(f'📈 Total Batch Accumulations : {total_raw_steps}')
        print(f'📉 Effective Optimization Steps: {effective_opt_steps} updates')
        print(f'⚡ Base SFT Learning Rate    : {args.lr}')
        print(f'🔋 Physical Batch Size       : {args.batch_size}')
        print(f'🔄 Gradient Accumulation     : {grad_accum_steps}')
        print(f'🧮 Auto Hardware Precision   : {target_dtype} (Mixed Precision Mode)')
        print(f'🎯 Local Attention Type     : {attn_implementation}')
        print(f'🛠️  Loaded Optimizer         : {optimizer_name}')
        print(f'⚙️  Optimized Module          : model.talker (Frozen encoder/vocoder)')
        print('='*70 + '\n')

    model.train()

    for epoch in range(num_epochs):
        # Keep speaker encoder strictly in evaluation mode
        if hasattr(model, 'speaker_encoder') and model.speaker_encoder is not None:
            model.speaker_encoder.eval()

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

                # Dynamic Check: Avoid applying text_projection on 1.7B (fixes scrambled voice mismatch)
                raw_text_embedding = model.talker.model.text_embedding(input_text_ids)
                if raw_text_embedding.shape[-1] != model.talker.model.codec_embedding.weight.shape[-1]:
                    input_text_embedding = (
                        model.talker.text_projection(raw_text_embedding) * text_embedding_mask
                    )
                else:
                    input_text_embedding = raw_text_embedding * text_embedding_mask
                
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

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
            # ── PRE-PRUNING OLDER CHECKPOINTS ──
            # Deletes older checkpoints BEFORE saving the new one to strictly keep the Kaggle disk usage under 9.1 GB (Resolves quota error)
            output_path_obj = Path(args.output_model_path)
            saved_checkpoints = sorted(
                output_path_obj.glob('checkpoint-epoch-*'),
                key=lambda x: int(x.name.split('-epoch-')[-1]) if '-epoch-' in x.name else -1
            )
            for old_ckpt in saved_checkpoints:
                try:
                    shutil.rmtree(old_ckpt)
                    print(f'🧹 Pruned older checkpoint to free space: {old_ckpt.name}')
                except Exception:
                    pass

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
            config_dict["tts_model_type"] = "base"  # Retains full-parameter base config to support ICL
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

            # Dropping speaker_encoder weights is bypassed to preserve in-context zero-shot capabilities during inference

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            print(f'✅ Saved updated model weights to: {output_dir}')

if __name__ == "__main__":
    train()
