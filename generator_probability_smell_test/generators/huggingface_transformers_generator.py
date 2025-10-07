from typing import Tuple, List, Optional

import math
import torch
from torch.nn import Parameter
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from generator_probability_smell_test.paired_distribution import PairedDistribution
from generator_probability_smell_test.speculative_generator import SpeculativeGenerator


class MockModelConfig(PretrainedConfig):
    def __init__(self):
        super().__init__(bos_token_id=0, eos_token_id=0, pad_token_id=0)


class MockModelOneStepGeneration(PreTrainedModel):
    """
    Huggingface-transformers compatible distribution- simulates a LLM that always generates a single token
    """
    def __init__(self, distribution: List[float], *inputs, **kwargs):
        super().__init__(MockModelConfig())
        self.distribution = Parameter(torch.tensor([0] + distribution).log())
        self.eos = torch.full_like(self.distribution, -math.inf)
        self.eos[0] = 1

    def forward(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            *args, **kwargs
    ):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        output = []
        for i in range(seq_len):
            if i == 0:
                output.append(self.distribution.detach().view(1, -1).expand(batch_size, -1))
            else:
                output.append(self.eos.detach().view(1, -1).expand(batch_size, -1))

        key_vals = torch.zeros((batch_size, 1, seq_len, 1), dtype=torch.float32, device=input_ids.device)

        return CausalLMOutputWithPast(logits=torch.stack(output, 1), past_key_values=((key_vals, key_vals),))

    def can_generate(cls) -> bool:
        return True

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


class MockModelForSequenceGeneration(PreTrainedModel):
    """
    Simulates a LLM that always generates a short sequence of tokens
    """
    def __init__(self, distribution: List[float], num_toks_per_step: int, num_steps: int, *inputs, **kwargs):
        super().__init__(MockModelConfig())

        assert len(distribution) == num_toks_per_step ** num_steps
        self.distribution = Parameter(torch.tensor(distribution).view((num_toks_per_step,) * num_steps))
        self.num_toks_per_step = num_toks_per_step
        self.num_steps = num_steps

    def forward(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            *args, **kwargs
    ):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        num_skip = 0

        if past_key_values is not None:
            relevant_past = past_key_values[0][0]
            assert (relevant_past.shape[0] == batch_size)
            num_skip = relevant_past.shape[2]
            input_ids = torch.cat([relevant_past.view((batch_size, -1)), input_ids], dim=1)
            seq_len = input_ids.shape[1]

        output = []
        for i in range(num_skip, seq_len):
            this_step = []
            for j in range(batch_size):
                if i < self.num_steps:
                    input_ids_up_to_seq = input_ids[j, 1:i + 1] - 1
                    distr_after_seq = self.distribution[tuple(input_ids_up_to_seq.tolist())]
                    if i < self.num_steps - 1:
                        sum_over_remaining_dims = distr_after_seq.sum(dim=tuple(range(1, self.num_steps - i)))
                    else:
                        sum_over_remaining_dims = distr_after_seq
                    sum_over_remaining_dims = sum_over_remaining_dims / sum_over_remaining_dims.sum()

                    this_step.append(torch.cat([torch.tensor([float("-inf")]), sum_over_remaining_dims.detach().log()]))
                else:
                    this_step.append(
                        torch.cat([torch.ones((1,)), torch.full((self.num_toks_per_step,), float("-inf"))]))

            output.append(torch.stack(this_step, 0))

        key_vals = input_ids.view((batch_size, 1, -1, 1))

        return CausalLMOutputWithPast(logits=torch.stack(output, 1), past_key_values=((key_vals, key_vals),))

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


class TransformersGenerator(SpeculativeGenerator):
    """
    Use HF Transformers built-in speculative decoding (one-step)
    """

    def generate(self, distribution: PairedDistribution, seed):
        main_model = MockModelOneStepGeneration(distribution.main)
        spec_model = MockModelOneStepGeneration(distribution.speculative)

        output = main_model.generate(assistant_model=spec_model, do_sample=True, max_new_tokens=3)
        return output[0][1].item() - 1


class TransformersGeneratorNonSpeculative(SpeculativeGenerator):
    """
    Use HF Transformers regular generation (one-step), for validation
    """

    def generate(self, distribution: PairedDistribution, seed):
        main_model = MockModelOneStepGeneration(distribution.main)

        output = main_model.generate(do_sample=True, max_new_tokens=3)
        return output[0][1].item() - 1


class TransformersGeneratorSequenceNonSpeculative(SpeculativeGenerator):
    """
    Use HF Transformers regular generation (multi-step), for validation
    """
    def __init__(self, num_toks_per_step: int, num_steps: int):
        self.num_toks_per_step = num_toks_per_step
        self.num_steps = num_steps

    def generate(self, distribution: PairedDistribution, seed):
        main_model = MockModelForSequenceGeneration(distribution.main, self.num_toks_per_step, self.num_steps)

        output = main_model.generate(do_sample=True, max_new_tokens=self.num_steps + 2)
        out_num = 0
        for tok_num in range(self.num_steps):
            out_num = out_num * self.num_toks_per_step + (output[0][tok_num + 1].item() - 1)

        return out_num


class TransformersGeneratorSequenceSpeculative(SpeculativeGenerator):
    """
    Use HF Transformers built-in speculative decoding (multi-step)
    """
    def __init__(self, num_toks_per_step: int, num_steps: int):
        self.num_toks_per_step = num_toks_per_step
        self.num_steps = num_steps

    def generate(self, distribution: PairedDistribution, seed):
        main_model = MockModelForSequenceGeneration(distribution.main, self.num_toks_per_step, self.num_steps)
        spec_model = MockModelForSequenceGeneration(distribution.speculative, self.num_toks_per_step, self.num_steps)

        output = main_model.generate(assistant_model=spec_model, do_sample=True, max_new_tokens=self.num_steps + 2)
        out_num = 0
        for tok_num in range(self.num_steps):
            out_num = out_num * self.num_toks_per_step + (output[0][tok_num + 1].item() - 1)

        return out_num
