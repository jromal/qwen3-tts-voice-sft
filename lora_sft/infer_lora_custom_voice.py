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

import soundfile as sf
import torch
from safetensors.torch import load_file

from qwen_tts import Qwen3TTSModel

try:
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
except ImportError as exc:
    raise SystemExit(
        "peft is required for LoRA inference. Install it with: pip install peft"
    ) from exc


def _resolve_core_model(model):
    if hasattr(model, "talker"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "talker"):
        return model.model
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "talker"):
            return base
        if hasattr(base, "model") and hasattr(base.model, "talker"):
            return base.model
    return model


def _apply_speaker_config(core_model, adapter_dir, speaker_name, speaker_id=None):
    config_path = os.path.join(adapter_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        talker_config = config_dict.get("talker_config", {})
        spk_id_map = talker_config.get("spk_id", {})
        spk_is_dialect = talker_config.get("spk_is_dialect", {})
        if isinstance(spk_id_map, dict):
            core_model.config.talker_config.spk_id.update(spk_id_map)
        if isinstance(spk_is_dialect, dict):
            core_model.config.talker_config.spk_is_dialect.update(spk_is_dialect)

    if speaker_name not in core_model.config.talker_config.spk_id and speaker_id is not None:
        core_model.config.talker_config.spk_id[speaker_name] = speaker_id
        core_model.config.talker_config.spk_is_dialect[speaker_name] = False

    core_model.config.tts_model_type = "custom_voice"
    core_model.tts_model_type = "custom_voice"
    core_model.supported_speakers = core_model.config.talker_config.spk_id.keys()


def _apply_speaker_embedding(core_model, adapter_dir, speaker_name, speaker_embedding_path=None):
    embedding_path = speaker_embedding_path or os.path.join(
        adapter_dir, "speaker_embedding.safetensors"
    )
    if not os.path.isfile(embedding_path):
        raise FileNotFoundError(f"Missing speaker embedding: {embedding_path}")

    payload = load_file(embedding_path)
    if "target_speaker_embedding" not in payload:
        raise KeyError(f"'target_speaker_embedding' not found in {embedding_path}")

    spk_id = core_model.config.talker_config.spk_id[speaker_name]
    weight = core_model.talker.model.codec_embedding.weight
    weight.data[spk_id] = payload["target_speaker_embedding"].to(
        device=weight.device, dtype=weight.dtype
    )


def _set_lora_scale(peft_model, scale):
    if scale == 1.0:
        return
    adapter = getattr(peft_model, "active_adapter", None)
    for module in peft_model.modules():
        if isinstance(module, LoraLayer):
            if adapter is None:
                module.scale_layer(scale)
            elif isinstance(adapter, (list, tuple)):
                for name in adapter:
                    module.set_scale(name, scale)
            else:
                module.set_scale(adapter, scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--speaker_name", default="female01")
    parser.add_argument("--speaker_id", type=int, default=3000)
    parser.add_argument("--speaker_embedding_path", default=None)
    parser.add_argument("--text", default="She said she would be here by noon.")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--output_wav", default="output_lora.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--no_merge_lora", action="store_true")
    parser.add_argument("--lora_scale", type=float, default=0.3)
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Cap generation length to prevent EOS failures (~0.5%% of inferences)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix random seed for reproducible output (useful for chunk consistency)")
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    tts = Qwen3TTSModel.from_pretrained(
        args.base_model_path,
        device_map=args.device,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attn_implementation,
    )

    peft_model = PeftModel.from_pretrained(tts.model, args.adapter_path)
    _set_lora_scale(peft_model, args.lora_scale)
    if args.no_merge_lora:
        tts.model = peft_model
    else:
        tts.model = peft_model.merge_and_unload()

    core_model = _resolve_core_model(tts.model)
    _apply_speaker_config(core_model, args.adapter_path, args.speaker_name, args.speaker_id)
    _apply_speaker_embedding(
        core_model, args.adapter_path, args.speaker_name, args.speaker_embedding_path
    )
    core_model.eval()

    # Fix seed for reproducible output (prevents timbre shift across chunks)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker_name,
        language=args.language,
    )
    sf.write(args.output_wav, wavs[0], sr)


if __name__ == "__main__":
    main()
