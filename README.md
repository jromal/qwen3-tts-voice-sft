# 🎙️ Qwen3-TTS Voice Fine-Tuning Studio

A unified repository for stable single-speaker voice adaptation on Alibaba's Qwen3-TTS 12Hz series, supporting both full-parameter supervised fine-tuning (Full SFT) and parameter-efficient adapter training (PEFT LoRA) on consumer-grade hardware.

*   **Repository Path:** `https://github.com/jromal/qwen3-tts-voice-sft`

---

## 📂 Repository Directory Structure

All patched scripts and inference tools are consolidated under a single monorepo:

```text
qwen3-tts-voice-sft/
├── README.md
├── full_sft/
│   └── sft_12hz.py                # Patched Full-Parameter SFT script
└── lora_sft/
    ├── sft_12hz_lora.py           # Patched PEFT LoRA SFT script
    └── infer_lora_custom_voice.py # Patched PEFT LoRA Inference script
```

---

## 🛠️ Integrated Stability, Precision & VRAM Patches

The scripts in this repository correct key upstream mathematical issues that otherwise lead to model corruption, progressive temporal acceleration, or load-time crashes during deployment:

1. **The `text_projection` Fix:** Explicitly applies the missing text projection layer transformation (`model.talker.text_projection`) on text embeddings during forward passes [2]. This ensures that the text and audio features align properly, preventing the model from outputting pure noise or generic static [2].
2. **Double-Shift Slice Correction:** Resolves the upstream temporal alignment mismatch [2]. Rather than pre-shifting model inputs, target labels are passed directly to `ForCausalLMLoss`, while slicing is restricted to aligning hidden states `[:-1]` with target codec indices `[1:]` via a contiguous mask [2]. This stops the model from learning compressed speech patterns that cause progressive acceleration across successive training epochs [2].
3. **Gradient Accumulation Optimizer Sync:** Wraps the `optimizer.step()` and `optimizer.zero_grad()` operations strictly inside the `if accelerator.sync_gradients:` block [2]. This prevents premature gradient resets during micro-batching when training with accumulation steps $> 1$, ensuring gradient statistics remain mathematically stable [2].
4. **Speaker Encoder Weight Retention:** Preserves the underlying speaker encoder layers in the final saved state dictionary [4]. Keeping these weights allows the model to leverage zero-shot prompt-based guidance via `model.generate_voice_clone(x_vector_only_mode=False)` during inference, which significantly stabilizes timbre and pronunciation consistency [4].
5. **Hardware-Adaptive Precision (`target_dtype`):** Detects hardware capability at runtime. It maps to native `bfloat16` on Ampere and newer architectures ($\ge 8.0$), but automatically falls back to standard `float16` on Turing (e.g. Tesla T4) to prevent memory allocation regressions or runtime exceptions.
6. **Paged Optimizer Fallback:** Falls back from standard `AdamW` to `bitsandbytes` `PagedAdamW8bit` [2]. This dramatically reduces the memory footprint of the optimizer states, enabling the 1.7B parameter SFT loop to run within standard 15 GB and 16 GB VRAM limits.
7. **Loss Gradient Rebalancing:** Stabilizes multi-token objective functions by applying a balanced `0.3` multiplier scaling factor on the frozen sub-talker outputs to prevent auxiliary layers from overpowering primary token prediction pathways [2].

---

## 📦 Dataset Preparation & Layout

To train either a Full SFT model or a LoRA adapter, compress your custom single-speaker dataset as a `.zip` archive or upload it directly with this directory layout:

```text
Your-Voice-Dataset/
├── train_raw.jsonl (Your transcription manifest)
├── ref.wav (Your raw target voice reference)
└── wavs/ (Folder containing training clips)
    ├── utt0001.wav
    ├── utt0002.wav
    └── ...
```

### Critical Processing & Metadata Rules:
* **Audio Resampling:** Every `.wav` file inside `wavs/` **must be pre-resampled to exactly 24,000 Hz (24 kHz) Mono** prior to running the training pipeline [1]. Uploading standard 44.1 kHz or 48 kHz clips will cause the tokenizer to extract incorrect codec sequences, degrading speaker identity and vocal clarity [1].
* **Temporal Sizing:** Aim for clean audio clips cut to lengths between 2 and 10 seconds, with background noise removed (SNR > 20dB).
* **Identity Lock:** Each line of your JSONL manifest should contain a `"language": "en"` key (or match your target speaker locale) to prevent identity drift during multi-turn script rendering [2, 3].

### Manifest Example (`train_raw.jsonl`)
```json
{"audio": "./wavs/utt0001.wav", "text": "This is a clean, resampled training sentence.", "ref_audio": "./ref.wav", "language": "en"}
```

---

## 🚀 Notebook Integration

Your execution notebooks clone this repository and inject the patched scripts directly into the upstream training environment, bypassing standard git merge and path resolution errors [2].

### 1. Standalone Full SFT Notebook (`WtB_Qwen3_TTS_Finetuning.ipynb`)
Clones this repository and executes the following setup command to overwrite the training file:
```bash
!cp qwen3-tts-voice-sft/full_sft/sft_12hz.py Qwen3-TTS/finetuning/sft_12hz.py
```
*   **Standalone Checkpoint Preservation:** The export script preserves the foundational text processor configurations (`vocab.json`, `merges.txt`, `tokenizer_config.json`, and `preprocessor_config.json`) inside the final saved directory [1.4.6]. This ensures the output checkpoint remains fully loadable via standard `Qwen3TTSModel.from_pretrained(...)` [1.4.6].
*   **Rule of 720:** Epochs scale dynamically via:
    $$\text{Epochs} = \text{round}\left(\frac{720}{N}\right) \quad \text{[Bounded between 4 and 20]}$$

### 2. PEFT LoRA Training Notebook (`WtB_Qwen3_TTS_LoRA_Training.ipynb`)
Clones this repository and copies both the PEFT training script and the corresponding inference adapter logic:
```bash
!cp qwen3-tts-voice-sft/lora_sft/sft_12hz_lora.py Qwen3-TTS/finetuning/sft_12hz_lora.py
!cp qwen3-tts-voice-sft/lora_sft/infer_lora_custom_voice.py Qwen3-TTS/finetuning/infer_lora_custom_voice.py
```
*   **Rule of 1200:** Epochs scale dynamically via:
    $$\text{Epochs} = \text{round}\left(\frac{1200}{N}\right) \quad \text{[Bounded between 6 and 30]}$$
*   **Metadata Bypass:** Automatically deletes `README.md` right before uploading checkpoints to Hugging Face, bypassing standard server-side metadata parser rejections [2].

---

## 👥 Credits & Acknowledgments

This training repository builds upon contributions from the following open-source frameworks and community developers:
*   [Alibaba Qwen Team](https://github.com/QwenLM/Qwen3-TTS) for the base Qwen3-TTS architecture and streaming engines.
*   [vspeech/Qwen3-TTS-Train](https://github.com/vspeech/Qwen3-TTS-Train) for community research regarding model synchronization, training behavior analyses, and dataset size guidelines [4].
*   [Finrandojin/alexandria-audiobook](https://github.com/Finrandojin/alexandria-audiobook) for the audiobook rendering pipelines and custom UI application components.
