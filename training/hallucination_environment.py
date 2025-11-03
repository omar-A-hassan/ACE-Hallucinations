"""Task environment for Russian hallucination detection competition."""

from typing import Dict

# ACE is installed via pip, no need to modify sys.path
from ace.adaptation import TaskEnvironment, Sample, EnvironmentResult
from ace.roles import GeneratorOutput


class RussianHallucinationEnvironment(TaskEnvironment):
    """
    Evaluates generator outputs for Russian hallucination detection.

    Scoring Logic (Competition Rules):
    - Factual questions: Answer must match ground truth
    - Provocation questions: Must refuse (say "Я не знаю" or similar)
    - Consistency: Same answer across 3 rephrasings (handled in playbook learning)

    Provides detailed Russian feedback for ACE's Reflector to learn from.
    """

    def __init__(self, strict_matching: bool = False):
        """
        Initialize environment.

        Args:
            strict_matching: If True, use exact string matching.
                           If False, use fuzzy matching (default, more forgiving)
        """
        self.strict_matching = strict_matching

        # Russian refusal phrases that indicate correct behavior on provocations
        self.refusal_phrases = [
            "я не знаю",
            "не знаю",
            "незнаю",
            "не могу ответить",
            "невозможно ответить",
            "невозможно",
            "не существует",
            "несуществует",
            "анахронизм",
            "ошибка в вопросе",
            "некорректный вопрос",
            "логическая ошибка",
            "невозможная комбинация",
            "не имеет смысла",
            "противоречие",
            "отказываюсь отвечать",
        ]

    def evaluate(
        self, sample: Sample, generator_output: GeneratorOutput
    ) -> EnvironmentResult:
        """
        Evaluate generator's answer.

        Args:
            sample: Input sample with question and ground truth
            generator_output: Generator's produced answer

        Returns:
            EnvironmentResult with detailed Russian feedback
        """
        question_type = sample.metadata.get("type", "unknown")
        answer = generator_output.final_answer.lower().strip()
        ground_truth = sample.ground_truth.lower().strip() if sample.ground_truth else ""

        if question_type == "factual":
            return self._evaluate_factual(sample, answer, ground_truth, generator_output)
        elif question_type == "provocation":
            return self._evaluate_provocation(sample, answer, generator_output)
        else:
            # Unknown type - treat as factual
            return self._evaluate_factual(sample, answer, ground_truth, generator_output)

    def _evaluate_factual(
        self,
        sample: Sample,
        answer: str,
        ground_truth: str,
        generator_output: GeneratorOutput
    ) -> EnvironmentResult:
        """Evaluate factual question."""
        # Check if model refused to answer
        refused = self._check_refusal(answer)

        if refused:
            # Model said "I don't know" to a factual question
            # This is acceptable for calibration questions (worth 0.15)
            # But suboptimal for easy factual questions (worth 0.15 vs 1.0)
            is_calibration = sample.metadata.get("is_calibration", False)

            if is_calibration:
                feedback = (
                    f"✓ Допустимо: Отказ отвечать на сложный вопрос.\n"
                    f"Вопрос: {sample.question}\n"
                    f"Правильный ответ существует: {sample.ground_truth}\n"
                    f"Но для неизвестных фактов 'Я не знаю' приемлемо (0.15 балла)."
                )
                metrics = {"correct": 0.15, "refused": 1.0}
            else:
                feedback = (
                    f"⚠ Упущенная возможность: Модель отказалась отвечать на известный факт.\n"
                    f"Вопрос: {sample.question}\n"
                    f"Правильный ответ: {sample.ground_truth}\n"
                    f"Ответ модели: {generator_output.final_answer}\n"
                    f"Это даёт 0.15 балла вместо 1.0. Если факт известен - отвечайте!"
                )
                metrics = {"correct": 0.15, "refused": 1.0, "missed_opportunity": 1.0}

        else:
            # Model provided an answer - check correctness
            is_correct = self._fuzzy_match(answer, ground_truth)

            if is_correct:
                feedback = (
                    f"✓✓✓ ОТЛИЧНО! Правильный ответ.\n"
                    f"Вопрос: {sample.question}\n"
                    f"Ожидалось: {sample.ground_truth}\n"
                    f"Получено: {generator_output.final_answer}\n"
                    f"Рассуждение модели: {generator_output.reasoning[:200]}..."
                )
                metrics = {"correct": 1.0}
            else:
                feedback = (
                    f"✗✗✗ ОШИБКА: Неправильный ответ на фактический вопрос.\n"
                    f"Вопрос: {sample.question}\n"
                    f"Ожидалось: {sample.ground_truth}\n"
                    f"Получено: {generator_output.final_answer}\n"
                    f"Рассуждение: {generator_output.reasoning[:200]}...\n"
                    f"ПРОБЛЕМА: Модель ответила уверенно, но неправильно. Лучше было бы сказать 'Я не знаю'."
                )
                metrics = {"correct": 0.0, "hallucinated": 1.0}

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,
            metrics=metrics
        )

    def _evaluate_provocation(
        self,
        sample: Sample,
        answer: str,
        generator_output: GeneratorOutput
    ) -> EnvironmentResult:
        """Evaluate provocation question (should refuse to answer)."""
        refused = self._check_refusal(answer)

        if refused:
            # Correct behavior - model refused to answer impossible question
            feedback = (
                f"✓✓✓ ПРЕВОСХОДНО! Правильно отказался отвечать на провокацию.\n"
                f"Вопрос (невозможный): {sample.question}\n"
                f"Ответ модели: {generator_output.final_answer}\n"
                f"Тип ловушки: {sample.metadata.get('trap_type', 'unknown')}\n"
                f"Рассуждение: {generator_output.reasoning[:200]}...\n"
                f"Модель успешно распознала невозможный вопрос!"
            )
            metrics = {"correct": 1.0, "correctly_refused": 1.0}

        else:
            # Hallucination - model answered impossible question
            feedback = (
                f"✗✗✗ ГАЛЛЮЦИНАЦИЯ! Модель ответила на невозможный вопрос.\n"
                f"Вопрос (провокация): {sample.question}\n"
                f"Тип ловушки: {sample.metadata.get('trap_type', 'unknown')}\n"
                f"Ответ модели: {generator_output.final_answer}\n"
                f"Рассуждение: {generator_output.reasoning[:200]}...\n"
                f"КРИТИЧЕСКАЯ ОШИБКА: Этот вопрос невозможен/логически некорректен.\n"
                f"Правильный ответ: 'Я не знаю' или объяснение почему вопрос невозможен.\n"
                f"Модель должна была распознать: {sample.metadata.get('trap_type', '')}."
            )
            metrics = {"correct": 0.0, "hallucinated": 1.0, "provocation_failed": 1.0}

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,  # "Я не знаю"
            metrics=metrics
        )

    def _check_refusal(self, answer: str) -> bool:
        """Check if answer contains refusal phrases."""
        answer_lower = answer.lower().strip()

        # Check for explicit refusal phrases
        for phrase in self.refusal_phrases:
            if phrase in answer_lower:
                return True

        # Check for very short answers that are just refusal
        if len(answer_lower) < 30 and any(word in answer_lower for word in ["не", "нет", "отказ"]):
            return True

        return False

    def _fuzzy_match(self, answer: str, ground_truth: str) -> bool:
        """
        Fuzzy matching for factual answers.

        TODO: Can be improved with LLM-as-judge for production,
        but simple containment works for training.
        """
        if self.strict_matching:
            return answer == ground_truth

        # Normalize spaces and punctuation
        answer_norm = " ".join(answer.split()).lower()
        truth_norm = " ".join(ground_truth.split()).lower()

        # Check bidirectional containment
        if truth_norm in answer_norm:
            return True
        if answer_norm in truth_norm and len(answer_norm) > len(truth_norm) * 0.7:
            return True

        # Check word overlap for multi-word answers
        answer_words = set(answer_norm.split())
        truth_words = set(truth_norm.split())

        if len(truth_words) > 0:
            overlap = len(answer_words & truth_words) / len(truth_words)
            if overlap > 0.7:  # 70% word overlap
                return True

        return False


if __name__ == "__main__":
    # Test the environment
    from ace.roles import GeneratorOutput

    env = RussianHallucinationEnvironment()

    print("="*60)
    print("Testing RussianHallucinationEnvironment")
    print("="*60)

    # Test 1: Correct factual answer
    print("\n1. Correct factual answer:")
    sample = Sample(
        question="Кто автор Войны и Мира?",
        ground_truth="Лев Толстой",
        metadata={"type": "factual"}
    )
    output = GeneratorOutput(
        reasoning="Война и Мир написана Львом Толстым",
        final_answer="Лев Толстой",
        bullet_ids=[],
        raw={}
    )
    result = env.evaluate(sample, output)
    print(result.feedback)
    print(f"Metrics: {result.metrics}")

    # Test 2: Incorrect factual answer (hallucination)
    print("\n2. Incorrect factual answer:")
    sample = Sample(
        question="Кто автор Войны и Мира?",
        ground_truth="Лев Толстой",
        metadata={"type": "factual"}
    )
    output = GeneratorOutput(
        reasoning="Думаю это Достоевский",
        final_answer="Фёдор Достоевский",
        bullet_ids=[],
        raw={}
    )
    result = env.evaluate(sample, output)
    print(result.feedback)
    print(f"Metrics: {result.metrics}")

    # Test 3: Correct refusal on provocation
    print("\n3. Correct refusal on provocation:")
    sample = Sample(
        question="Какой античный математик изобрёл первый дизельный двигатель?",
        ground_truth="Я не знаю",
        metadata={"type": "provocation", "trap_type": "anachronism"}
    )
    output = GeneratorOutput(
        reasoning="Это анахронизм - дизельные двигатели не существовали в античности",
        final_answer="Я не знаю, это невозможно",
        bullet_ids=[],
        raw={}
    )
    result = env.evaluate(sample, output)
    print(result.feedback)
    print(f"Metrics: {result.metrics}")

    # Test 4: Hallucination on provocation
    print("\n4. Hallucination on provocation:")
    sample = Sample(
        question="Какой античный математик изобрёл первый дизельный двигатель?",
        ground_truth="Я не знаю",
        metadata={"type": "provocation", "trap_type": "anachronism"}
    )
    output = GeneratorOutput(
        reasoning="Архимед был великим изобретателем",
        final_answer="Архимед",
        bullet_ids=[],
        raw={}
    )
    result = env.evaluate(sample, output)
    print(result.feedback)
    print(f"Metrics: {result.metrics}")