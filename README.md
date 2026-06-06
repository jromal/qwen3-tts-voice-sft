# 🎙️ qwen3-tts-voice-sft

### Unified Training Studio for Qwen3-TTS-12Hz (Full SFT & LoRA SFT)

**Maintained and owned by [@jromal](https://github.com/jromal), creator of [Watch the Book](https://www.youtube.com/@WatchTheBook).**

This repository acts as the consolidated, highly optimized backend file engine for the official **Watch the Book** 1-click execution notebooks. It consolidates both full-parameter SFT and parameter-efficient LoRA SFT training workflows into a single workspace.

---

## 📁 Repository Structure

```text
qwen3-tts-voice-sft/
├── README.md
├── full_sft/
│   └── sft_12hz.py                # Stable Full-Parameter SFT script
└── lora_sft/
    ├── sft_12hz_lora.py           # Stable LoRA SFT script
    └── infer_lora_custom_voice.py # Standard LoRA Inference script
```

---

## 🛠️ Integrated Stability & VRAM Fixes

The scripts in this repository have been corrected to resolve known upstream bugs and hardware constraints on Turing-architecture GPUs (such as the Tesla T4):

1. **Hardware-Adaptive Precision (`target_dtype`):** Automatically detects the GPU's Compute Capability. It defaults to native `bfloat16` on Ampere or Hopper architectures (Compute Capability $\ge 8.0$), but cleanly falls back to standard `float16` on Turing (T4) or Volta architectures to prevent performance lag or OOM crashes.
2. **`GradScaler` FP16 Override:** Patches PyTorch's native `_unscale_grads_` module during execution to stably allow FP16 gradient scaling and clipping, preventing terminal runtime errors during backward passes.
3. **Loss Gradient Rebalancing:** Restores the `0.3` multiplier scaling factor on the `sub_talker_loss` to prevent sub-talker gradients from overpowering the primary talker (which is responsible for predicting the end-of-speech `<|EOS|>` token).
4. **Hugging Face Cache Resolver:** Intercepts model path declarations right before checkpoints are serialized, resolving the model's actual local caching directory on disk using `snapshot_download` to prevent `FileNotFoundError` during file copying.
5. **Paged Optimizer Fallback:** Swaps standard PyTorch `AdamW` for `bitsandbytes` `PagedAdamW8bit` [2] to drastically compress the static optimizer state overhead, fitting the 1.7B parameter Full SFT footprint comfortably inside 15 GB of VRAM.
6. **Numeric Epoch Pruner:** Automatically sorts output checkpoint folders numerically by epoch index (rather than unstable OS directory modified timestamps) to safely prune older checkpoints, preserving virtual disk space on cloud runtimes.

---

## 📦 Dataset Preparation & Layout

To train either a Full SFT model or a LoRA adapter, format your custom dataset into a single folder or a compressed `.zip` archive structured as follows:

```text
Your-Voice-Dataset/
├── train_raw.jsonl (Your transcription file)
├── ref.wav         (Your reference speaker audio)
└── wavs/           (A folder containing all training audio clips)
    ├── utt0001.wav
    ├── utt0002.wav
    └── ...
```

### Manifest Format (`train_raw.jsonl`)
Your manifest must be a standard JSON Lines file where each entry references paths matching your directory structure. 

**Example entry:**
```json
{"audio": "./wavs/utt0001.wav", "text": "其实我真的有发现...", "ref_audio": "./ref.wav"}
```

---

## 🚀 Notebook Integration

Your 1-click execution notebooks are designed to clone this unified repository and copy the required training files directly into the official `Qwen3-TTS/finetuning/` folder during the setup phase, completely bypassing `git apply` or other patch-engine execution bugs [2].

### 1. For the Full SFT Notebook (`WtB_Qwen3_TTS_Finetuning.ipynb`)
Step 3 in the notebook clones your repository and executes this copy command:
```bash
!cp qwen3-tts-voice-sft/full_sft/sft_12hz.py Qwen3-TTS/finetuning/sft_12hz.py
```

### 2. For the LoRA SFT Notebook (`WtB_Qwen3_TTS_LoRA_Finetuning.ipynb`)
Step 3 in the notebook clones your repository and executes these copy commands:
```bash
!cp qwen3-tts-voice-sft/lora_sft/sft_12hz_lora.py Qwen3-TTS/finetuning/sft_12hz_lora.py
!cp qwen3-tts-voice-sft/lora_sft/infer_lora_custom_voice.py Qwen3-TTS/finetuning/infer_lora_custom_voice.py
```
