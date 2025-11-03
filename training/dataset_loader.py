"""Dataset loader for Russian hallucination detection competition."""

import json
from typing import List

# ACE is installed via pip, no need to modify sys.path
from ace.adaptation import Sample


def load_russian_dataset(json_path: str, shuffle: bool = True) -> List[Sample]:
    """
    Load Russian hallucination dataset and convert to ACE Sample format.

    Args:
        json_path: Path to final_dataset.json or full_dataset.json
        shuffle: Whether to shuffle samples (default: True for better learning)

    Returns:
        List of Sample objects ready for ACE training

    Dataset Format:
        {
            "dataset_version": "2.0",
            "examples": [
                {
                    "id": "fact_0001",
                    "type": "factual",
                    "question_variations": ["Q1", "Q2", "Q3"],
                    "answer": {"text": "Answer"}
                },
                {
                    "id": "prov_0001",
                    "type": "provocation",
                    "question": "Impossible question",
                    "correct_behavior": {"response_type": "refuse"}
                }
            ]
        }

    Sample Creation Logic:
        - Factual: 3 samples (one per rephrasing) with same answer
        - Provocation: 1 sample with ground_truth="Я не знаю"
        - Calibration: 3 samples (treated as factual)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    examples = data.get("examples", [])

    print(f"Loading dataset from {json_path}")
    print(f"Dataset version: {data.get('dataset_version', 'unknown')}")
    print(f"Total examples: {len(examples)}")

    factual_count = 0
    provocation_count = 0
    calibration_count = 0

    for example in examples:
        example_id = example.get("id", "unknown")
        example_type = example.get("type")

        if example_type == "factual":
            # Create 3 samples for 3 rephrasings
            question_variations = example.get("question_variations", [])
            answer_text = example.get("answer", {}).get("text", "")

            if not answer_text:
                print(f"Warning: Empty answer for {example_id}, skipping")
                continue

            for i, question_variant in enumerate(question_variations):
                samples.append(Sample(
                    question=question_variant,
                    ground_truth=answer_text,
                    metadata={
                        "type": "factual",
                        "example_id": example_id,
                        "rephrasing_index": i,
                        "total_rephrasings": len(question_variations),
                        "topic": example.get("topic", "unknown"),
                        "difficulty": example.get("difficulty", "unknown")
                    }
                ))
                factual_count += 1

        elif example_type == "provocation":
            # Create 1 sample - correct behavior is to refuse
            question = example.get("question", "")

            if not question:
                print(f"Warning: Empty question for {example_id}, skipping")
                continue

            # Ground truth for provocations is refusal
            samples.append(Sample(
                question=question,
                ground_truth="Я не знаю",  # Expected behavior
                metadata={
                    "type": "provocation",
                    "example_id": example_id,
                    "trap_type": example.get("trap_type", "unknown"),
                    "difficulty": example.get("difficulty", "unknown")
                }
            ))
            provocation_count += 1

        elif example_type == "calibration":
            # Create 3 samples - treat as factual (obscure but real facts)
            question_variations = example.get("question_variations", [])
            answer_text = example.get("answer", {}).get("text", "")

            if not answer_text:
                print(f"Warning: Empty answer for {example_id}, skipping")
                continue

            for i, question_variant in enumerate(question_variations):
                samples.append(Sample(
                    question=question_variant,
                    ground_truth=answer_text,
                    metadata={
                        "type": "factual",  # Treat as factual
                        "example_id": example_id,
                        "rephrasing_index": i,
                        "total_rephrasings": len(question_variations),
                        "is_calibration": True,
                        "difficulty": "hard"
                    }
                ))
                calibration_count += 1

        else:
            print(f"Warning: Unknown example type '{example_type}' for {example_id}")

    print(f"\nDataset loaded successfully:")
    print(f"  Factual samples: {factual_count}")
    print(f"  Provocation samples: {provocation_count}")
    print(f"  Calibration samples: {calibration_count}")
    print(f"  Total samples: {len(samples)}")

    # Shuffle for better learning (mix factual and provocations)
    if shuffle:
        import random
        random.shuffle(samples)
        print("  Samples shuffled for training")

    return samples


def split_dataset(samples: List[Sample], train_ratio: float = 0.8):
    """
    Split dataset into train and validation sets.

    Args:
        samples: List of Sample objects
        train_ratio: Fraction for training (default: 0.8)

    Returns:
        Tuple of (train_samples, val_samples)
    """
    split_idx = int(len(samples) * train_ratio)
    return samples[:split_idx], samples[split_idx:]


if __name__ == "__main__":
    # Test the loader
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "../data/final_dataset.json"

    samples = load_russian_dataset(dataset_path)

    print("\n" + "="*60)
    print("Sample examples:")
    print("="*60)

    # Show 2 factual and 2 provocation examples
    factual_samples = [s for s in samples if s.metadata.get("type") == "factual"][:2]
    provocation_samples = [s for s in samples if s.metadata.get("type") == "provocation"][:2]

    print("\nFactual examples:")
    for i, sample in enumerate(factual_samples, 1):
        print(f"\n{i}. Question: {sample.question}")
        print(f"   Answer: {sample.ground_truth}")
        print(f"   Metadata: {sample.metadata}")

    print("\nProvocation examples:")
    for i, sample in enumerate(provocation_samples, 1):
        print(f"\n{i}. Question: {sample.question}")
        print(f"   Expected: {sample.ground_truth}")
        print(f"   Metadata: {sample.metadata}")
