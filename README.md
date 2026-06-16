# 🎙️ Qwen3-TTS Voice Fine-Tuning Studio

A unified, self-contained repository for stable single-speaker voice adaptation on Alibaba's Qwen3-TTS 12Hz series. This studio supports both full-parameter supervised fine-tuning (Full SFT) and parameter-efficient adapter training (PEFT LoRA) on consumer-grade hardware, with direct deployment to a unified inference and audio auditing server.

*   **Repository Path:** `https://github.com/jromal/qwen3-tts-voice-sft`

---

## 📂 Repository Directory Structure

All notebooks, patched scripts, and deployment configurations are consolidated inside this monorepo, allowing you to clone the codebase and begin execution immediately:

```text
qwen3-tts-voice-sft/
├── README.md
├── WtB_Qwen3_TTS_Finetuning.ipynb    # Native Full-Parameter SFT Notebook
├── WtB_Qwen3_TTS_LoRA_Training.ipynb # Native Parameter-Efficient LoRA SFT Notebook
├── WtB_Qwen3_TTS_and_Whisper.ipynb   # Native Unified Inference & Audit Notebook
├── full_sft/
│   └── sft_12hz.py                   # Patched Full-Parameter SFT script
└── lora_sft/
    ├── sft_12hz_lora.py              # Patched PEFT LoRA SFT script
    └── infer_lora_custom_voice.py    # Patched PEFT LoRA Inference script
```

---

## 🛠️ Upgraded SFT Core Patches (`full_sft/sft_12hz.py`)

The full-parameter SFT training engine has been updated with several critical algorithmic fixes to prevent weight corruption and audio degradation:

### 1. Dynamic `text_projection` Bypass Check
*   **The Issue:** On the 1.7B parameter base model, the text embedding dimension (2048) matches the transformer hidden dimension (2048) [3.1.3]. Consequently, **the 1.7B model bypasses the text projection layer entirely during standard inference**.
*   **The Problem:** Forcing text embeddings through `model.talker.text_projection()` during 1.7B SFT training introduces an extra linear transformation that is completely absent during inference [1.4.7]. This mismatch causes the language model to read unaligned features during generation, resulting in a totally scrambled, distorted, robotic sound (resembling "aliens underwater") [3.1.9, 4.3.1].
*   **The Fix:** We have introduced a dynamic, dimension-based check inside the forward pass:
    ```python
    raw_text_embedding = model.talker.model.text_embedding(input_text_ids)
    if raw_text_embedding.shape[-1] != model.talker.model.codec_embedding.weight.shape[-1]:
        input_text_embedding = model.talker.text_projection(raw_text_embedding) * text_embedding_mask
    else:
        input_text_embedding = raw_text_embedding * text_embedding_mask
    ```
    This automatically applies the projection layer on the 0.6B variant (to map 2048 dimensions to 1024), but bypasses it on the 1.7B model (dimensions 2048 vs 2048), maintaining alignment with the inference pipeline [4.3.1].

### 2. Acoustic Scrambling Mitigation (Component Freezing)
*   **The Problem:** Standard SFT pipelines often call `model.train()` on the top-level class, putting all modules—including the pre-trained `speaker_encoder` and `speech_tokenizer` (VQ vocoder)—into training mode. Allowing the vocoder's batch normalization statistics and weights to float during SFT destroys its acoustic reconstruction capabilities, producing severe voice scrambling [2.4.4].
*   **The Fix:** The training script explicitly freezes the speaker encoder and vocoder parameters (`requires_grad = False`), and enforces `.eval()` mode on the `speaker_encoder` throughout the SFT loop [2.4.4]. Fine-tuning is strictly isolated to the `model.talker` (the semantic language model) [2.4.4].

### 3. Double-Shift Slice Correction (Resolving Progressive Acceleration)
*   **The Problem:** Manoevering target labels before they reach the Hugging Face causal loss module can cause double-shifting [2]. This causes the model to learn a temporal compression error, meaning the synthesized voice accelerates progressively over successive training epochs until it is completely fast-forwarded and unintelligible [2].
*   **The Fix:** Rather than pre-shifting, standard unshifted labels are passed directly to `ForCausalLMLoss` [2.2.6]. Slicing is performed strictly on aligning the hidden states `[:-1]` with target codec indices `[1:]` through a contiguous mask [2]:
    ```python
    hidden_states = outputs.hidden_states[0][-1]
    target_codec_mask = codec_mask[:, 1:]
    talker_hidden_states = hidden_states[:, :-1, :][target_codec_mask]
    talker_codec_ids = codec_ids[:, 1:][target_codec_mask]
    ```

### 4. Gradient Accumulation Sync wrapping
*   **The Problem:** In standard Accelerate loops with `gradient_accumulation_steps > 1`, executing `optimizer.step()` and `optimizer.zero_grad()` outside the `sync_gradients` check forces parameter resets on every micro-batch, corrupting accumulated gradient math.
*   **The Fix:** Both operations are wrapped strictly inside the `sync_gradients` block:
    ```python
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    ```

---

## 🧬 PEFT LoRA & Alexandria Integration (Expressive Adapters)

The Parameter-Efficient Fine-Tuning (PEFT) pipeline allows you to compile highly expressiveness adapters (averaging **~58 MB** instead of 3.83 GB) [4], making them ideal for integration with consumer audiobook rendering platforms such as **Alexandria Audiobook Studio** [1]:

### 1. The Alexandria Workflow Concept
*   **Description-Based Voice Design:** Alexandria's "Voice Design" interface takes plain-text descriptions (e.g., "An elderly, low-register female narrator with a warm, steady cadence") to synthesize a 30 to 50 sentence training dataset [1].
*   **Rapid Adapter Training:** These synthetic wav clips are compiled into a dataset and trained using `sft_12hz_lora.py` [1]. This targets strictly the Qwen3-TTS attention projection layers (`q_proj,k_proj,v_proj,o_proj`), preserving overall linguistic pronunciation while reshaping vocal timbre and expressive patterns [4].

### 2. Core LoRA SFT Script Patches (`lora_sft/sft_12hz_lora.py`)
Training PEFT models on top of a multi-component neural speech model requires targeted adaptations to prevent gradient corruption:
*   **The PEFT Unpeeler:** Wrapping the inner `talker` module in a PEFT framework turns it into a `PeftModel`. When the training loop accesses embedding parameters, resolving attributes on the wrapped model throws a fatal `AttributeError`. SFT scripts use a custom unpeeler to access the raw `Qwen2Model` layers cleanly:
    ```python
    talker = model.talker
    if hasattr(talker, "base_model") and hasattr(talker.base_model, "model"):
        raw_talker = talker.base_model.model
    else:
        raw_talker = talker
    ```
*   **Frozen Sub-Talker Loss Isolation:** Under a PEFT configuration, the auxiliary code predictors (layers 1-15) are frozen. Backpropagating auxiliary sub-talker losses through frozen parameters into active attention adapters pollutes the gradient landscape, warping speech outputs. The SFT script sets the sub-talker loss scale strictly to `0.0` during LoRA runs, focusing the adapters 100% on the primary codebook predictions.

### 3. Dynamic Inference Loading & Embedding Injection
When loading trained adapters onto the unified Gradio server, the model dynamically incorporates the learned features:
*   **Active PEFT Injection:** The server uses `PeftModel.from_pretrained` to load the low-rank weights directly onto the base model's inner `talker` module, applying a scaled context window.
*   **Static/Dynamic Embedding Mapping:** SFT weights are mapped to index `3000` of the model's vocal embeddings. The server loads the speaker tensor from `speaker_embedding.safetensors` [1.1.2] or extracts it dynamically on-the-fly from a 9-second `ref_sample.wav` on disk [1.1.2, 1.3.1]. It injects this tensor into the model's active parameters, enabling you to hot-swap and generate dozens of custom voices.

---

## 🛠️ Dynamic Speaker Encoder Restoration

To create a voice that supports both your speaker's identity and natural language style/emotion control (e.g., "whisper", "sad tone"), you must train on top of the **`CustomVoice`** model (which possesses the instruction-following attention weights) rather than the `Base` model [1.1.3, 2.3.4].

*   **The Obstacle:** `CustomVoice` is configured with `tts_model_type = "custom_voice"` on initialization, which forces `self.speaker_encoder = None` to conserve memory [1.1.2]. Attempting to run standard SFT on it causes a fatal `NoneType` AttributeError during the forward pass [1.1.2].
*   **The Fix:** During training startup, if `sft_12hz.py` detects a `CustomVoice` variant, it automatically:
    1. Instantiates a standard `Qwen3TTSSpeakerEncoder` [1.1.2].
    2. Downloads the matching configuration from the `Base` model cache (dynamically resolving to `1.7B-Base` or `0.6B-Base` depending on scale) to define the correct 2048-dimensional projection layers, preventing dimension mismatch errors [1.1.5, 2.3.4].
    3. Downloads and extracts the speaker encoder weights, loads them into the active module, and casts them to the correct hardware device and precision dtype (`float16` or `bfloat16`) [1.1.2].

---

## 🛠️ Kaggle & Hugging Face Storage Optimizations

Trained checkpoints are uploaded directly to the Hugging Face Hub. To prevent storage-quota and memory-exhaustion failures on public notebook environments, two designed optimizations are implemented:

### 1. Pre-Pruning Disk Optimization (Kaggle 20 GB Quota Fix)
*   **The Problem:** Standard training containers (such as Kaggle) enforce a strict **20 GB local disk space limit**. Waiting until after the new checkpoint is saved to delete the previous epoch's files forces the disk to temporarily hold multiple 4.52 GB checkpoints at the same time, exceeding the 20 GB ceiling and leading to silent file-writing failures [2.1.3].
*   **The Fix:** The script executes **Pre-Pruning**. It scans and deletes the previous epoch's checkpoint folder *before* starting the new save block, keeping peak disk usage strictly under **~9.12 GB** [2.1.3].

### 2. Zero-History Branch-Based Architecture (HF 100 GB Private LFS Fix)
*   **The Problem:** Overwriting or deleting a file on a standard Git branch creates a new commit while keeping the old LFS blobs in the Git history commit tree. This causes a single SFT repository's private storage footprint to increase by ~4.5 GB on every run [1.1.5], quickly exceeding Hugging Face's 100 GB private storage limit.
*   **The Fix:** Instead of saving folders under the `main` branch, each voice is mapped to its own **independent Git branch** named after the speaker [1.2.2]. During Step 5:
    1. The notebook calls `api.delete_branch` to clear out historical LFS caches [1.1.3].
    2. It re-initializes a fresh branch via `api.create_branch` to prevent 404 Revision errors [1.2.2].
    3. It uploads the directory as a single clean commit using `api.upload_folder` [1.2.2].
    
This completely deletes your old branch's history on every push, ensuring your account only ever uses exactly **4.5 GB of space per voice**, with zero history overhead, while keeping your other branches (other voices) untouched [1.1.5].

---

## 📂 Dataset Preparation & Layout

To train either a Full SFT model or a LoRA adapter, compress your custom single-speaker dataset as a `.zip` archive or upload it directly with this directory layout:

```text
Your-Voice-Dataset/
├── train_raw.jsonl (Your transcription manifest)
├── ref.wav (Your raw target voice reference)
├── ref.txt (The transcription text of your ref.wav)
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
*   **Dual-Naming Reference Copies:** During Step 5, the notebook automatically copies both `ref.wav` and `ref.txt` to the final epoch folder using dual names (`ref.wav`/`ref_sample.wav` and `ref.txt`/`ref_sample.txt`) [1.3.1].
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

## 🚀 Unified Inference & Deployment

Trained models are deployed utilizing the unified inference server (`WtB_Qwen3_TTS_and_Whisper.ipynb`) [1]. The production deployment architecture relies on several designed optimizations [1]:

### 1. Zero-History Branch-Based Architecture (HF Git LFS Fix)
To prevent massive Git LFS file history bloat inside centralized Hugging Face repositories, each voice model is saved on its own independent, zero-history branch named after the `target_speaker_name` [2.1]. 
*   **Dynamic Scanning:** The server queries branches dynamically (`api.list_repo_refs`) to populate character choices instead of scanning repo files [2.1].
*   **On-Demand Downloads:** Checked models are loaded by setting the target branch as the snapshot revision (`revision=voice_name` / `subfolder=None`) [2.1]. This completely isolates SFT weight downloads and prevents downloading unrelated models [2.1].

### 2. Dual-Inference Execution Modes
*   **SFT Custom Voice Mode (Timbre SFT + Active Style Controls):** SFT models trained on top of `1.7B-CustomVoice` inherit the pre-trained, instruction-following attention maps [2, 2.2.7]. At load-time, the server forces `"tts_model_type": "custom_voice"` [2]. This disables the speaker encoder and routes generation natively through `model.generate_custom_voice()` using SFT index `3000` [2]. This maps style instructions cleanly as a conditioning latent (`instruct`), preventing style prompts from being read aloud [2].
*   **SFT + ICL Hybrid Mode (For Low-Resource Datasets):** For short datasets (under 30 minutes of voice data), index-based embedding can experience mathematical collapse [2]. The server dynamically keeps the model in `"base"` mode, activating the `speaker_encoder` to extract a stable timbre from your `ref.wav` on disk while using the SFT-trained attention layers to guide expressive cadence [1, 2].

### 3. Dynamic Text-Proportional Token Limiter
To prevent infinite generation loops (EOS token failures) [1], the inference engine computes a dynamic, text-proportional safe ceiling based on character count [1, 2]:
$$\text{max\_new\_tokens} = \min\left(4096, \max\left(250, \text{round}\left(C \times 3.75\right)\right)\right)$$

*   **20-Second Floor (250 tokens):** Guarantees a minimum headroom of 20 seconds [1, 2], preventing the truncation of short sentences with long expressive pauses [1, 2].
*   **Context Safety Window (4096 tokens):** Caps absolute sequence boundaries to prevent attention matrix memory overflow (OOM) crashes on large text passages [1, 2].
*   **Explicit EOS Boundaries:** Explicitly passes the validated Qwen3-TTS tokenizer EOS array (`[2150, 2157, 151670, 151673, 151645, 151643]`) to all generation loops, providing explicit targets for clean stops [1].

### 4. Precision & Hardware Tuning
*   **Turing GPU Precision Stability:** Turing-architecture GPUs (such as the Tesla T4) suffer from PyTorch SDPA attention stability issues under native `float16` precision. Forcing `DTYPE_TTS = torch.bfloat16` with `"attn_implementation": "sdpa"` completely prevents underflow NaNs and process-halting crashes on T4 cloud VMs [1].
*   **Physical Hardware Casting:** High-level wrapper pipelines can cause weights to remain partially on CPU memory, spike system CPU to 100%, and hang inference. Explicitly casting the model via `current_tts_model.model.to(DEVICE)` right after load-time moves all weight matrices onto GPU memory, accelerating execution speeds [1].

---

## 👥 Credits & Acknowledgments

This training repository builds upon contributions from the following open-source frameworks and community developers:
*   [Alibaba Qwen Team](https://github.com/QwenLM/Qwen3-TTS) for the base Qwen3-TTS architecture and streaming engines.
*   [vspeech/Qwen3-TTS-Train](https://github.com/vspeech/Qwen3-TTS-Train) for community research regarding model synchronization, training behavior analyses, and dataset size guidelines [4].
*   [Finrandojin/alexandria-audiobook](https://github.com/Finrandojin/alexandria-audiobook) for the audiobook rendering pipelines and custom UI application components.
