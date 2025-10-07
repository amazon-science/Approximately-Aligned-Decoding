from typing import Optional, Set, List

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from approx_aligned_decoding.backtracking_strategy import BacktrackingStrategy
from approx_aligned_decoding.backtracking_strategy.limit_backtracking import LimitBacktracking
from approx_aligned_decoding.generator import generate_with_asst, generate
from approx_aligned_decoding.hallucination_detector.hallucination_detector import HallucinationDetector, \
    HallucinationDetectionResult
from approx_aligned_decoding.utils import set_seed
from generator_probability_smell_test.generators.huggingface_transformers_generator import MockModelOneStepGeneration, \
    MockModelForSequenceGeneration
from generator_probability_smell_test.paired_distribution import PairedDistribution
from generator_probability_smell_test.speculative_generator import SpeculativeGenerator


def banned_tok_to_seq(banned_seq: int, num_toks_per_step: int, num_steps: int) -> List[int]:
    tok_seq = []
    for i in range(num_steps):
        this_step = (banned_seq % num_toks_per_step) + 1
        tok_seq.append(this_step)
        banned_seq = banned_seq // num_toks_per_step

    return tok_seq[::-1]


class MockHalluDetector(HallucinationDetector):
    def __init__(self, banned_seqs: Set[int], num_toks_per_step: int, num_steps: int, tokenizer: Tokenizer):
        self.banned_seqs = banned_seqs
        self.num_toks_per_step = num_toks_per_step
        self.num_steps = num_steps
        self.tokenizer = tokenizer
        self.banned_strs = [
            self.tokenizer.decode(banned_tok_to_seq(banned_seq, num_toks_per_step, num_steps)) for banned_seq
            in banned_seqs]

    def detect(self, left_context: str, right_context: str, generated_text: str, is_end: bool,
               index_previously_checked: int) -> HallucinationDetectionResult:
        if any(banned_str in generated_text for banned_str in self.banned_strs):
            return HallucinationDetectionResult(hallucination=True, index_of_hallucination=self.num_steps - 1)
        else:
            return HallucinationDetectionResult(hallucination=False, index_of_hallucination=None)


class CustomGenerator(SpeculativeGenerator):
    """
    Our speculative decoding method
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Doesn't matter for smell test

    def generate(self, distribution: PairedDistribution, seed):
        main_model = MockModelOneStepGeneration(distribution.main)
        spec_model = MockModelOneStepGeneration(distribution.speculative)

        output, _ = generate_with_asst(model=main_model,
                                       assistant_model=spec_model,
                                       hallucination_detector=None,
                                       input_ids=torch.tensor([[0]]),
                                       right_context="",
                                       left_context="",
                                       max_new_tokens=1,
                                       stopping_criteria=None,
                                       spec_sequence_length=2,
                                       num_speculative_sequences=2,
                                       seed_func=lambda x: set_seed((x, seed)),
                                       tokenizer=self.tokenizer)

        return output[0] - 1


class CustomGeneratorSequence(SpeculativeGenerator):
    def __init__(self, num_toks_per_step: int, num_steps: int,
                 backtrack_strategy: Optional[BacktrackingStrategy] = None):
        self.num_toks_per_step = num_toks_per_step
        self.num_steps = num_steps
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Doesn't matter for smell test
        self.backtrack_strategy = backtrack_strategy

    def generate(self, distribution: PairedDistribution, seed):
        main_model = MockModelForSequenceGeneration(distribution.main, self.num_toks_per_step, self.num_steps)
        spec_model = MockModelForSequenceGeneration(distribution.speculative, self.num_toks_per_step, self.num_steps)

        output, _ = generate_with_asst(model=main_model,
                                       assistant_model=spec_model,
                                       hallucination_detector=MockHalluDetector(distribution.hallus,
                                                                                tokenizer=self.tokenizer,
                                                                                num_steps=self.num_steps,
                                                                                num_toks_per_step=self.num_toks_per_step),
                                       input_ids=torch.tensor([[0]], dtype=torch.long),
                                       right_context="",
                                       left_context="",
                                       max_new_tokens=self.num_steps,
                                       stopping_criteria=None,
                                       spec_sequence_length=self.num_steps + 1,
                                       num_speculative_sequences=4,
                                       backtracking_strategy=LimitBacktracking(self.backtrack_strategy, 50,
                                                                               silent=True) if self.backtrack_strategy is not None else self.backtrack_strategy,
                                       tokenizer=self.tokenizer,
                                       seed_func=lambda x: set_seed((x, seed)),

                                       )

        out_num = 0
        for tok_num in range(self.num_steps):
            out_num = out_num * self.num_toks_per_step + (output[tok_num] - 1)

        return out_num


class CustomGeneratorNonSpeculative(SpeculativeGenerator):
    """
    Our generate method (non-speculative)
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Doesn't matter for smell test

    def generate(self, distribution: PairedDistribution, seed):
        main_model = MockModelOneStepGeneration(distribution.main)

        output, _ = generate(model=main_model,
                             hallucination_detector=None,
                             input_ids=torch.tensor([[0]]),
                             right_context="",
                             left_context="",
                             max_new_tokens=1,
                             stopping_criteria=None,
                             seed_func=lambda x: set_seed((x, seed)),
                             tokenizer=self.tokenizer)

        return output[0] - 1


class CustomGeneratorSequenceNonSpeculative(SpeculativeGenerator):
    def __init__(self, num_toks_per_step: int, num_steps: int,
                 backtrack_strategy: Optional[BacktrackingStrategy] = None):
        self.num_toks_per_step = num_toks_per_step
        self.num_steps = num_steps
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Doesn't matter for smell test
        self.backtrack_strategy = backtrack_strategy
        self.total_toks_model = 0
        self.total_output_toks = 0

    def generate(self, distribution: PairedDistribution, seed):
        main_model = MockModelForSequenceGeneration(distribution.main, self.num_toks_per_step, self.num_steps)

        output, stats = generate(model=main_model,
                             hallucination_detector=MockHalluDetector(distribution.hallus, tokenizer=self.tokenizer,
                                                                      num_steps=self.num_steps,
                                                                      num_toks_per_step=self.num_toks_per_step),
                             input_ids=torch.tensor([[0]], dtype=torch.long),
                             right_context="",
                             left_context="",
                             max_new_tokens=self.num_steps,
                             stopping_criteria=None,
                             backtracking_strategy=LimitBacktracking(self.backtrack_strategy,
                                                                     50,
                                                                     silent=True) if self.backtrack_strategy is not None else self.backtrack_strategy,

                             seed_func=lambda x: set_seed((x, seed)),
                             tokenizer=self.tokenizer)

        out_num = 0
        for tok_num in range(self.num_steps):
            out_num = out_num * self.num_toks_per_step + (output[tok_num] - 1)

        self.total_output_toks += self.num_steps
        self.total_toks_model += stats.total_generated_toks_model

        return out_num
