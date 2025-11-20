`train_pipeline.sh` stitches together the project phases: generate/prepare data, finetune models, then quantize the finetuned checkpoints. This guide covers a quick start (using the provided labels) and the full end-to-end pipeline.

## Quick start: train with provided labels
Labels are already prepared in this repo. You mainly need to build the merged datasets, create prompts, train, then quantize. From repo root:
```bash
cd train

# 1) Build merged datasets (copies videos to data/videos)
python -m generate_data.get_data_1 --base-dir traffic_buddy_train+public_test
python -m generate_data.get_data_2 --base-dir traffic_buddy_train+public_test

# 2) Create SFT prompts/labels
python -m generate_data.create_train_with_systemprompt \
  --path-data-train-1 data/json/train_public_argument.json \
  --path-data-train-2 data/json/train_public.json \
  --path-video data/videos \
  --output-dir data/labels

# 3) Finetune (two runs)
bash scripts/finetune_video_train_public_argument.sh
bash scripts/finetune_video_train_public.sh

# 4) Quantize
python -m quantization.quantization_model \
  --model-path checkpoint/train_public_argument \
  --save-dir checkpoint/quantization_model_store/train_public_argument
python -m quantization.quantization_model \
  --model-path checkpoint/train_public \
  --save-dir checkpoint/quantization_model_store/train_public
```
Environment overrides (optional):
```bash
export DATA_ROOT="traffic_buddy_train+public_test"
export CHECKPOINT_DIR="checkpoint"
export QUANTIZATION_STORE="checkpoint/quantization_model_store"
```

## End-to-end pipeline (`train_pipeline.sh`)
Use this when you also need to regenerate synthetic data with Gemini/OpenAI. From repo root:
```bash
cd train
bash train_pipeline.sh [all|data|train|quantize]
```
- `all` (default): data generation → training → quantization.
- `data`: only the data synthesis/processing steps.
- `train`: only finetuning (`scripts/finetune_video_train_public_argument.sh`, then `scripts/finetune_video_train_public.sh`).
- `quantize`: only quantization for the two checkpoints.

### Common environment overrides
```bash
# Dataset and I/O
export DATA_ROOT="traffic_buddy_train+public_test"
export TRAIN_JSON="$DATA_ROOT/train/train.json"
export RAW_JSON_DIR="data/raw_json"
export JSON_DIR="data/json"
export LABEL_DIR="data/labels"
export VIDEO_DIR="data/videos"

# Training/quantization outputs
export CHECKPOINT_DIR="checkpoint"
export QUANTIZATION_STORE="checkpoint/quantization_model_store"

# API keys (needed for data stage)
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
```
You can change the Python binary with `PYTHON=python3.11`.

## Pipeline phase details
### 1) Data generation (`bash train_pipeline.sh data`)
1. `generate_data.gemini_generate_step_1`  
   - Inputs: `TRAIN_JSON`, `DATA_ROOT`.  
   - Output: `$RAW_JSON_DIR/gemini_result_step_1.json`.
2. `generate_data.gemini_generate_step_2`  
   - Inputs: `--json-path $RAW_JSON_DIR/gemini_result_step_1.json`, `--api-key $GEMINI_API_KEY`.  
   - Output: `$RAW_JSON_DIR/gemini_result_step_2.json`.
3. `generate_data.generate_argumentation_question_answer`  
   - Uses `OPENAI_API_KEY`. The script’s internal `INPUT_FILE`/`OUTPUT_FILE` are inside the Python file; adjust there if your dataset differs. The pipeline moves `data_argument.json` to `$RAW_JSON_DIR/data_argument.json`.
4. `generate_data.get_data_1` and `get_data_2`  
   - Merge/normalize JSONs and copy videos into `$VIDEO_DIR` (default `data/videos`).
5. `generate_data.create_train_with_systemprompt`  
   - Builds SFT labels into `$LABEL_DIR` using `train_public_argument.json` and `train_public.json`.

Artifacts: processed JSONs under `$RAW_JSON_DIR`/`$JSON_DIR`, copied videos in `$VIDEO_DIR`, and label files in `$LABEL_DIR`.

### 2) Training (`bash train_pipeline.sh train`)
- Runs:
  - `scripts/finetune_video_train_public_argument.sh` → checkpoint at `$CHECKPOINT_DIR/train_public_argument`
  - `scripts/finetune_video_train_public.sh` → checkpoint at `$CHECKPOINT_DIR/train_public`
- Expects to be run from `train/` with `PYTHONPATH=src`.

### 3) Quantization (`bash train_pipeline.sh quantize`)
- Calls `quantization.quantization_model` twice:
  - `--model-path $CHECKPOINT_DIR/train_public_argument --save-dir $QUANTIZATION_STORE/train_public_argument`
  - `--model-path $CHECKPOINT_DIR/train_public --save-dir $QUANTIZATION_STORE/train_public`
- Outputs quantized model + processor copies into `$QUANTIZATION_STORE`.

## Quick sanity checks
- Confirm raw/processed JSONs exist: `ls $RAW_JSON_DIR` and `ls $JSON_DIR`.
- Confirm videos copied: `ls $VIDEO_DIR | head`.
- After training, checkpoints should live under `$CHECKPOINT_DIR`. After quantization, expect subfolders under `$QUANTIZATION_STORE`.
