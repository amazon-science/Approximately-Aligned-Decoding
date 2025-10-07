from typing import Tuple, Optional, Sequence, List

import torch
from torch import Tensor
from transformers import PreTrainedModel, LogitsProcessorList
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithPast


class ModelWithHistory:
    """
    Encapsulates issues with key-value cache
    Could potentially be replaced with a DynamicCache, but having a functional interface is very nice when using speculative decoding
    """
    def __init__(self, model: PreTrainedModel, device: str, kv_cache: Optional[Tuple[Tuple[Tensor, Tensor]]] = None,
                 logits_hist: Optional[Tensor] = None, input_id_hist: Optional[Tensor] = None,
                 warper: LogitsProcessorList = None):
        self.model = model
        self.device = device
        self.kv_cache: Optional[Tuple[Tuple[Tensor, Tensor]]] = kv_cache
        self.logits_hist: Optional[torch.Tensor] = logits_hist
        self.input_id_hist: Optional[torch.Tensor] = input_id_hist
        self.warper = warper

    def __len__(self):
        if self.input_id_hist is None:
            return 0
        else:
            return self.input_id_hist.shape[0]

    def truncate(self, tok_idx: int) -> "ModelWithHistory":
        assert tok_idx <= len(self)
        if tok_idx == len(self):
            return self
        if tok_idx == 0:
            return ModelWithHistory(self.model, self.device, warper=self.warper)

        new_cache = tuple(
            tuple(
                tens[:, :, :tok_idx] for tens in layer
            )
            for layer in self.kv_cache
        )

        new_logits_hist = self.logits_hist[:tok_idx]
        new_input_id_hist = self.input_id_hist[:tok_idx]

        return ModelWithHistory(self.model, self.device, new_cache, new_logits_hist, new_input_id_hist, self.warper)


    def _apply_history(self, history: CausalLMOutputWithPast, batch_idx: int, input_ids: torch.Tensor) -> "ModelWithHistory":
        new_cache = tuple(
            tuple(
                new[batch_idx:batch_idx+1] for new in new_layer
            )
            for new_layer in history.past_key_values
        )
        new_logits_hist = torch.cat((self.logits_hist, history.logits[batch_idx])) if self.logits_hist is not None else history.logits[batch_idx]
        new_input_id_hist = torch.cat((self.input_id_hist, input_ids[batch_idx])) if self.input_id_hist is not None else input_ids[batch_idx]

        return ModelWithHistory(self.model, self.device, new_cache, new_logits_hist, new_input_id_hist, self.warper)

    def __call__(self, input_ids: torch.Tensor, include_prev_step_logits: bool = False) -> Tuple[CausalLMOutputWithPast, Tuple["ModelWithHistory", ...]]:
        batch_size = input_ids.shape[0]
        past_key_values = self.kv_cache

        if past_key_values is not None and batch_size > 1:
            past_key_values = tuple(
                tuple(
                    tens.tile((batch_size, 1, 1, 1))
                    for tens in layer
                )
                for layer in past_key_values
            )

        attention_mask = torch.ones((batch_size, len(self) + input_ids.shape[1]), device=self.device)
        position_ids = torch.arange(len(self), len(self) + input_ids.shape[1], device=self.device).view((1, -1)).tile(
            (batch_size, 1))

        result = self.model(input_ids=input_ids, past_key_values=past_key_values,
                               attention_mask=attention_mask, use_cache=True,
                               position_ids=position_ids)
        result.logits = torch.stack(
            [self.warper(input_ids, result.logits[:, i, :].to(torch.float64)) for i in range(result.logits.shape[1])],
            dim=1)
        ret_result = result
        if include_prev_step_logits:
            ret_result = CausalLMOutputWithPast(
                logits=torch.cat([self.logits_hist[-1:].view((1, 1, -1)).tile((batch_size, 1, 1)), result.logits], dim=1)
            )
        return ret_result, tuple(self._apply_history(result, i, input_ids) for i in range(batch_size))

    @staticmethod
    def batch_call(histories: Sequence["ModelWithHistory"], input_ids: torch.Tensor, include_prev_step_logits: bool = False) -> Tuple[CausalLMOutputWithPast, Tuple["ModelWithHistory", ...]]:
        assert len(histories) != 0
        assert len(input_ids.shape) == 2
        assert len(histories) == input_ids.shape[0]

        history_len = len(histories[0])
        model = histories[0].model
        warper = histories[0].warper
        for i in histories:
            assert len(i) == history_len
            assert model == i.model

        if history_len > 0:
            num_layers = len(histories[0].kv_cache)

            fused_past_key_values = tuple(
                (torch.cat([h.kv_cache[i][0] for h in histories], dim=0),
                 torch.cat([h.kv_cache[i][1] for h in histories], dim=0))
                for i in range(num_layers)
            )
        else:
            fused_past_key_values = None

        attention_mask = torch.ones((len(histories), history_len + input_ids.shape[1]), device=histories[0].device)
        position_ids = torch.arange(history_len, history_len + input_ids.shape[1]).view(1, -1).tile(
            (input_ids.shape[0], 1)).to(histories[0].device)

        result = model(input_ids=input_ids,
                          past_key_values=fused_past_key_values,
                          attention_mask=attention_mask,
                          position_ids=position_ids,
                          use_cache=True)
        ret_result = result
        result.logits = torch.stack(
            [warper(input_ids, result.logits[:, i, :].to(torch.float64)) for i in range(result.logits.shape[1])], dim=1)

        if include_prev_step_logits:
            prev_logits = torch.stack([h.logits_hist[-1:] for h in histories])

            ret_result = CausalLMOutputWithPast(
                logits=torch.cat([prev_logits, result.logits], dim=1),
            )

        return ret_result, tuple(h._apply_history(result, i, input_ids) for i, h in enumerate(histories))

    @staticmethod
    def _single_step_sample(logits: torch.Tensor):
        probs = logits[:, -1].softmax(dim=1)
        return torch.multinomial(probs, 1).squeeze(-1)

    @staticmethod
    def _combine_outputs(outputs: List[CausalLMOutputWithPast], selected_ids: List[torch.LongTensor]) -> GenerateDecoderOnlyOutput:
        return GenerateDecoderOnlyOutput(
            past_key_values=None, # We already handle these
            logits=torch.cat([o.logits for o in outputs], dim=1) if outputs[0].logits is not None else None,
            sequences=torch.cat(selected_ids, dim=1),
            attentions=None, # Currently aren't using these two, add if needed
            hidden_states=None
        )

    @staticmethod
    def _only_most_recent_result(output: CausalLMOutputWithPast) -> CausalLMOutputWithPast:
        if output.logits.shape[1] == 1:
            return output

        return CausalLMOutputWithPast(
            past_key_values=None,  # Already handle these
            logits=output.logits[:, -1:],
            hidden_states=None,  # Add if needed later on
            attentions=None
        )

    def generate(self, input_ids: torch.Tensor, num_return_sequences: int, max_new_tokens: int) -> Tuple[GenerateDecoderOnlyOutput, Tuple["ModelWithHistory", ...], int]:
        histories = [self] * num_return_sequences
        assert len(input_ids.shape) == 1 or (len(input_ids.shape) == 2 and input_ids.shape[0] == 1)
        input_ids = input_ids.view((1, -1)).tile((len(histories), 1))
        results = []
        selected_ids = []
        active_generations = [True] * num_return_sequences

        num_tokens_generated = 0
        for num_tokens_generated in range(max_new_tokens):
            this_result, histories = ModelWithHistory.batch_call(histories, input_ids)
            results.append(this_result)
            input_ids = ModelWithHistory._single_step_sample(this_result.logits).view((-1, 1))
            selected_ids.append(input_ids)
            input_ids_cpu = input_ids.tolist()

            for gen_num, input_id in enumerate(input_ids_cpu):
                if input_id == self.model.config.eos_token_id:
                    active_generations[gen_num] = False

            if not any(active_generations):
                break

        return ModelWithHistory._combine_outputs(results, selected_ids), histories, num_tokens_generated







