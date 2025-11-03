# ACE-Hallucinations

Russian Hallucination Detection Competition (codeforces) Solution using Agentic Context Engineering (ACE)

## Overview

This project implements a hallucination-resistant LLM solution for the Russian language using the ACE (Agentic Context Engineering) framework. The solution combines a small local model (Gemma-3-270M) with learned anti-hallucination strategies to achieve robust factual consistency.

## Project Structure

```
ACE-Hallucinations/
├── data/
│   └── final_dataset.json          # Training dataset (380 examples)
├── training/
│   ├── train_russian_hallucination.py  # ACE training script
│   ├── playbook_russian.json       # Trained playbook (83 strategies)
│   └── playbook_russian_error.json # Checkpoint (64 strategies)
├── ace/                            # ACE framework (local modifications)
```

## Dataset

### Source Data
- **Base Dataset**: SberQuAD (Russian Question Answering Dataset)
- **Size**: 380 total examples
  - 300 factual questions (verifiable facts)
  - 50 provocation questions (designed to trigger hallucinations)
  - 30 calibration questions (boundary cases)
  
note: I could have used more data as the original Sber dataset had thousands of questions but I did not have the time nore resources to convert it to the correct format.

### Data Transformation Pipeline

1. **Question Selection** (from SberQuAD):
   - Extracted Russian context-question-answer triplets
   - Filtered for factual questions across diverse topics
   - Topics: Paleontology, Geography, History, Sports, Science, etc.

2. **Question Variation Generation** (using Gemini Flash 2.5):
   - Generated 3 paraphrased versions per question
   - Maintained semantic equivalence while varying:
     - Sentence structure
     - Word choice
     - Question formulation
   - Example:
     ```
     Original: "Чем представлены органические остатки в протерозое?"
     Variation 1: "Какие формы принимают органические остатки протерозоя?"
     Variation 2: "В какой форме встречаются следы жизни в протерозое?"
     ```

3. **Provocation Question Creation** (using Gemini Flash 2.5):
   - Generated 50 adversarial questions designed to trigger hallucinations:
     - **Anachronisms**: "Which ancient mathematician invented the diesel engine?"
     - **Non-existent entities**: "Who is the author of [fabricated book title]?"
     - **Impossible scenarios**: "What regulation was discussed at G20 2009 about cryptocurrencies?" (crypto didn't exist yet)
     - **Contradictory premises**: Logically impossible questions

4. **Answer Verification**:
   - All factual questions have verified correct answers from SberQuAD
   - Provocation questions have expected answer: "Я не знаю" (I don't know)
   - Calibration questions test edge cases and boundary conditions

### Dataset Schema

```json
{
  "id": "fact_0001",
  "type": "factual|provocation|calibration",
  "topic": "Category name",
  "difficulty": "easy|medium|hard",
  "question_variations": ["Question 1", "Question 2", "Question 3"],
  "answer": {
    "text": "Correct answer",
    "acceptable_variations": ["Alternative phrasing 1", "Alternative 2"]
  },
  "verification": {
    "source": "SberQuAD",
    "verified": true
  }
}
```

## Training Methodology

### ACE Framework Configuration

**Three-Role Architecture**:
1. **Generator**: Gemma-3-270M (local, MLX)
   - Role: Generate answers to questions
   - Inference: Native M3 acceleration via MLX
   - Context: Uses playbook strategies for guidance

2. **Reflector**: Gemini Flash 2.5 (API)
   - Role: Analyze generator outputs for hallucination risks
   - Identifies: False confidence, fabricated facts, anachronisms
   - Output: Reflection on answer quality and strategy effectiveness

3. **Curator**: Gemini Flash 2.5 (API)
   - Role: Update playbook based on reflections
   - Creates: New anti-hallucination strategies
   - Maintains: Strategy effectiveness metrics

### Training Process

```bash
# Initial training (hit rate limits)
python training/train_russian_hallucination.py --epochs 2

# Resume with rate limiting (2s delay between questions)
export GEMINI_API_KEY="your-key-here"
python training/train_russian_hallucination.py \
  --resume playbook_russian_error.json \
  --api-delay 2.0 \
  --epochs 1 \
  --max-samples 24
```

**Training Statistics**:
- Initial run: Processed ~25 samples before hitting Gemini free tier limit (50 requests/day)
- Resume run: Processed 24 additional samples with rate limiting
- Total strategies learned: 83
- Most effective strategy: "Check for well-known facts" (104 helpful uses)

### Rate Limiting

Each training sample requires 2 Gemini API calls (Reflector + Curator):
- Free tier: 50 requests/day = max 25 samples/day
- Solution: Added `--api-delay` parameter
- Recommended: `--api-delay 2.0` (30 questions/minute, safe for free tier)

## Learned Strategies

The playbook contains 83 anti-hallucination strategies in Russian. Top strategies include:

1. **Check for well-known facts** (104 helpful uses)
   - Provide factual answers when information is well-established
   - Include concrete examples when questions require them

2. **Recognize anachronisms** (multiple variations)
   - Detect temporal impossibilities (e.g., ancient tech + modern inventions)
   - Respond with "Я не знаю" for historically impossible scenarios

3. **Avoid false confidence when refusing**
   - Use phrases expressing uncertainty appropriately
   - Examples: "К сожалению, у меня нет информации" instead of confident "Я не знаю"

4. **Verify information availability**
   - Check if information exists in reliable sources before answering
   - Admit lack of knowledge when sources are unavailable

5. **Recognize impossible questions**
   - Identify logical contradictions and false premises
   - Refuse to answer questions with inherent impossibilities


### Inference Behavior
- **Conservative approach**: Prioritizes safety over answering
- **Hallucination resistance**: Zero fabricated answers observed
- **Default behavior**: Responds "Я не знаю" when uncertain
- **Anachronism detection**: Successfully identifies temporal impossibilities

### Competition Scoring
Scoring formula: `1000 * (0.8 * consistency + 0.2 * hallucination_provocation)`

Strengths:
- ✅ High hallucination resistance (no false positives)
- ✅ Perfect consistency (deterministic refusals)
- ⚠️ May lose points on answerable factual questions (very conservative)


## Results

- **Training dataset**: 380 Russian questions (300 factual, 50 provocations, 30 calibration)
- **Playbook size**: 83 anti-hallucination strategies
- **Model size**: 518MB (BF16 GGUF)
- **Hallucination rate**: 0% (no fabricated answers)
- **Answer rate**: ~5% (very conservative, prioritizes safety)

## References

- **ACE Framework**: Zhang et al. (2024). "Agentic Context Engineering". arXiv:2510.04618. [GitHub](https://github.com/kayba-ai/agentic-context-engine)

- **SberQuAD Dataset**: Efimov et al. (2020). "SberQuAD – Russian Reading Comprehension Dataset: Description and Analysis". arXiv:1912.09723. [Dataset](https://huggingface.co/datasets/sberquad)

- **Gemma Model**: Google DeepMind (2024). "Gemma: Open Models Based on Gemini Research and Technology". [Model Card](https://huggingface.co/google/gemma-3-270m-it)