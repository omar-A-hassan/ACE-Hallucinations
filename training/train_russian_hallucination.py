#!/usr/bin/env python3
"""
Train Gemma 270M on Russian hallucination detection using ACE framework.

This script:
1. Loads Russian dataset (factual + provocation questions)
2. Sets up hybrid ACE: Gemma 270M (Generator) + Gemini 2.5 (Reflector/Curator)
3. Runs offline adaptation to build playbook of anti-hallucination strategies
4. Saves trained playbook for inference

Usage:
    # Train on test dataset (160 examples)
    python train_russian_hallucination.py

    # Train on full dataset
    python train_russian_hallucination.py --dataset ../data/full_dataset.json --epochs 3

    # Resume from existing playbook
    python train_russian_hallucination.py --resume playbook_russian.json
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# ACE is installed via pip, no need to modify sys.path
from ace import Generator, Reflector, Curator, OfflineAdapter, Playbook
from ace.llm_providers import LiteLLMClient
from ace.llm import MLXLLMClient

# Local imports (same directory)
from hallucination_environment import RussianHallucinationEnvironment
from dataset_loader import load_russian_dataset, split_dataset
from prompts_russian import (
    GENERATOR_PROMPT_RU,
    REFLECTOR_PROMPT_RU,
    CURATOR_PROMPT_RU
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Gemma 270M for Russian hallucination detection using ACE"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../data/final_dataset.json",
        help="Path to dataset JSON file (default: ../data/final_dataset.json)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing playbook JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="playbook_russian.json",
        help="Output playbook file (default: playbook_russian.json)"
    )
    parser.add_argument(
        "--gemma-model",
        type=str,
        default="mlx-community/gemma-3-270m-it-bf16",
        help="Gemma model path (default: mlx-community/gemma-3-270m-it-bf16)"
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between questions to avoid API rate limits (default: 0)"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help="Validation split ratio (default: 0.0 = no validation)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of training samples (default: None = use all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Gemma generation temperature (default: 0.3)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("ACE Training for Russian Hallucination Detection")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Gemma model: {args.gemma_model}")
    print(f"Output: {args.output}")
    print("="*80)

    # Check for Gemini API key
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️  WARNING: No GEMINI_API_KEY or GOOGLE_API_KEY found!")
        print("Set one of these environment variables to use Gemini for Reflector/Curator")
        print("Example: export GEMINI_API_KEY='your-key-here'")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Step 1: Load dataset
    print("\n" + "="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)
    samples = load_russian_dataset(args.dataset, shuffle=True)

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    # Optional validation split
    if args.validation_split > 0:
        train_samples, val_samples = split_dataset(samples, 1.0 - args.validation_split)
        print(f"Split: {len(train_samples)} train, {len(val_samples)} validation")
    else:
        train_samples = samples
        val_samples = []
        print(f"Using all {len(train_samples)} samples for training")

    # Step 2: Initialize models
    print("\n" + "="*80)
    print("STEP 2: Initializing Models")
    print("="*80)

    print("\nLoading Gemma 270M (Generator) via MLX...")
    print(f"  Model: {args.gemma_model}")
    print("  This may take a few minutes on first run...")

    try:
        gemma_client = MLXLLMClient(
            model_path=args.gemma_model,
            max_tokens=2048,  # Increased for Russian + growing playbook
            temperature=args.temperature,
            system_prompt=(
                "Вы помощник, который отвечает в формате JSON. "
                "КРИТИЧЕСКИ ВАЖНО: Если вы не знаете ответ, скажите 'Я не знаю'. "
                "Никогда не выдумывайте факты."
            )
        )
        print("✓ Gemma 270M loaded successfully via MLX")
    except Exception as e:
        print(f"✗ Failed to load Gemma 270M: {e}")
        print("\nTroubleshooting:")
        print("1. Install MLX-LM: pip install mlx-lm")
        print("2. Check model name is correct")
        print("3. Ensure M-series Mac (MLX only works on Apple Silicon)")
        return

    print("\nLoading Gemini Flash 2.0 (Reflector/Curator) via API...")
    try:
        gemini_client = LiteLLMClient(
            model="gemini/gemini-2.0-flash-exp",
            temperature=0.7,
            max_tokens=4096  # Increased for long Russian responses
        )
        print("✓ Gemini Flash 2.0 client initialized")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize Gemini client: {e}")
        print("Reflector/Curator may fail during training")

    # Step 3: Create ACE components
    print("\n" + "="*80)
    print("STEP 3: Creating ACE Components")
    print("="*80)

    # Load or create playbook
    if args.resume:
        print(f"Loading existing playbook from {args.resume}...")
        playbook = Playbook.load_from_file(args.resume)
        print(f"✓ Loaded playbook with {len(playbook.bullets())} strategies")
    else:
        print("Creating new empty playbook...")
        playbook = Playbook()
        print("✓ Created empty playbook")

    # Create roles with Russian prompts
    print("\nCreating ACE roles with Russian prompts...")
    generator = Generator(gemma_client, prompt_template=GENERATOR_PROMPT_RU)
    reflector = Reflector(gemini_client, prompt_template=REFLECTOR_PROMPT_RU)
    curator = Curator(gemini_client, prompt_template=CURATOR_PROMPT_RU)
    print("✓ Generator (Gemma 270M)")
    print("✓ Reflector (Gemini Flash 2.0)")
    print("✓ Curator (Gemini Flash 2.0)")

    # Create adapter
    print("\nCreating OfflineAdapter...")
    if args.api_delay > 0:
        print(f"  API delay: {args.api_delay}s between questions (to avoid rate limits)")
    adapter = OfflineAdapter(
        playbook=playbook,
        generator=generator,
        reflector=reflector,
        curator=curator,
        max_refinement_rounds=1,
        reflection_window=3,
        enable_observability=False,  # Disable Opik for now
        api_delay=args.api_delay
    )
    print("✓ OfflineAdapter created")

    # Step 4: Run training
    print("\n" + "="*80)
    print("STEP 4: Running ACE Training")
    print("="*80)
    print(f"Training on {len(train_samples)} samples for {args.epochs} epochs")
    print(f"Total iterations: {len(train_samples) * args.epochs}")
    print("="*80)
    print()

    environment = RussianHallucinationEnvironment(strict_matching=False)

    try:
        results = adapter.run(train_samples, environment, epochs=args.epochs)

        print("\n" + "="*80)
        print("STEP 5: Training Complete!")
        print("="*80)

        # Calculate statistics
        total_samples = len(results)
        correct_count = sum(
            1 for r in results
            if r.environment_result.metrics.get("correct", 0) > 0.5
        )
        hallucinated_count = sum(
            1 for r in results
            if r.environment_result.metrics.get("hallucinated", 0) > 0
        )

        print(f"\nTraining Statistics:")
        print(f"  Total samples processed: {total_samples}")
        print(f"  Correct answers: {correct_count} ({correct_count/total_samples*100:.1f}%)")
        print(f"  Hallucinations: {hallucinated_count} ({hallucinated_count/total_samples*100:.1f}%)")
        print(f"  Playbook strategies: {len(adapter.playbook.bullets())}")

        # Save playbook
        print(f"\nSaving playbook to {args.output}...")
        adapter.playbook.save_to_file(args.output)
        print(f"✓ Playbook saved successfully")

        # Show sample strategies
        print("\n" + "="*80)
        print("Sample Learned Strategies:")
        print("="*80)
        bullets = adapter.playbook.bullets()
        for i, bullet in enumerate(bullets[:10], 1):
            print(f"\n{i}. [{bullet.id}] {bullet.section}")
            print(f"   {bullet.content}")
            print(f"   Stats: helpful={bullet.helpful}, harmful={bullet.harmful}, neutral={bullet.neutral}")

        if len(bullets) > 10:
            print(f"\n... and {len(bullets)-10} more strategies")

        print("\n" + "="*80)
        print(f"✓ Training completed successfully!")
        print(f"✓ Playbook saved to: {args.output}")
        print(f"✓ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Validation run if validation set exists
        if val_samples:
            print("\n" + "="*80)
            print("STEP 6: Validation")
            print("="*80)
            print(f"Running validation on {len(val_samples)} samples...")

            val_correct = 0
            val_hallucinated = 0

            for sample in val_samples:
                output = generator.generate(
                    question=sample.question,
                    context="",
                    playbook=adapter.playbook
                )
                result = environment.evaluate(sample, output)
                if result.metrics.get("correct", 0) > 0.5:
                    val_correct += 1
                if result.metrics.get("hallucinated", 0) > 0:
                    val_hallucinated += 1

            print(f"\nValidation Results:")
            print(f"  Accuracy: {val_correct}/{len(val_samples)} ({val_correct/len(val_samples)*100:.1f}%)")
            print(f"  Hallucinations: {val_hallucinated}/{len(val_samples)} ({val_hallucinated/len(val_samples)*100:.1f}%)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving partial playbook...")
        adapter.playbook.save_to_file(args.output.replace(".json", "_partial.json"))
        print(f"✓ Partial playbook saved")

    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving playbook state before exit...")
        adapter.playbook.save_to_file(args.output.replace(".json", "_error.json"))


if __name__ == "__main__":
    main()
