# Russian Hallucination Detection Training

Training Gemma 270M for Russian hallucination detection using ACE framework.

## Quick Start

### 1. Install Dependencies

```bash
# Install MLX-LM for M3 optimization
pip install mlx-lm

# Install ACE dependencies
cd ../ace
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# For Gemini (Reflector/Curator)
export GEMINI_API_KEY="your-gemini-api-key"
# OR
export GOOGLE_API_KEY="your-google-api-key"
```

### 3. Run Training

```bash
# Test run on 160 examples (fast)
python train_russian_hallucination.py

# Full dataset training
python train_russian_hallucination.py --dataset ../data/full_dataset.json --epochs 3

# Resume from checkpoint
python train_russian_hallucination.py --resume playbook_russian.json --epochs 1
```

## Architecture

### Hybrid ACE Setup

- **Generator**: Gemma 270M via MLX (Apple Silicon optimized)
  - Produces answers using learned playbook
  - Runs locally on M3 MacBook

- **Reflector**: Gemini 2.5 Flash via API
  - Analyzes what went right/wrong
  - Only used during training

- **Curator**: Gemini 2.5 Flash via API
  - Updates playbook with new strategies
  - Only used during training

### Training Flow

```
For each sample:
  1. Generator (Gemma) produces answer using playbook
  2. Environment evaluates (correct/hallucinated/refused)
  3. Reflector (Gemini) analyzes the outcome
  4. Curator (Gemini) updates playbook strategies
  5. Repeat
```

After training, only **Gemma + playbook** needed for inference.

## Output

- `playbook_russian.json`: Learned strategies for hallucination detection
- Contains bullets like:
  - "Check for anachronisms before answering"
  - "Say 'Я не знаю' for impossible combinations"
  - "Verify entity existence before answering"

## Command Line Arguments

```
--dataset PATH          Dataset JSON file (default: ../data/final_dataset.json)
--epochs N              Training epochs (default: 2)
--resume PATH           Resume from existing playbook
--output PATH           Output playbook file (default: playbook_russian.json)
--gemma-model PATH      Gemma model (default: mlx-community/gemma-3-270m-f16)
--temperature FLOAT     Gemma temperature (default: 0.3)
--max-samples N         Limit training samples (for testing)
--validation-split      Validation ratio (default: 0.0)
```
