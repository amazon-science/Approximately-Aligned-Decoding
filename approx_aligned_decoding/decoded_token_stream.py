import bisect
from builtins import IndexError
from typing import Tuple, Sequence

from transformers import PreTrainedTokenizer

WARN_ON_NONSTANDARD_SPACING = True
THROW_ON_NONSTANDARD_SPACING = False

class DecodedTokenStream:
    """
    Class to handle issues with multi-token characters, weird spacing between chars, etc.
    Should be a nicer interface than BatchEncoding to translate characters to token indices and vice versa
    """
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 tokens: Tuple[int, ...] = (),
                 decoded: Tuple[str, ...] = (),
                 decoded_raw: Tuple[str, ...] = (),
                 offsets: Tuple[int, ...] = (),
                 deferred_toks: Tuple[int, ...] = ()):
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.decoded = decoded
        self.decoded_raw = decoded_raw
        self.offsets = offsets
        self.deferred_toks = deferred_toks

    def __len__(self) -> int:
        return len(self.tokens)

    def truncate(self, idx: int) -> "DecodedTokenStream":
        num_cut_off = len(self) - idx
        assert num_cut_off >= 0

        return DecodedTokenStream(
            tokenizer=self.tokenizer,
            tokens=self.tokens[:idx],
            decoded=self.decoded[:idx],
            decoded_raw=self.decoded_raw[:idx],
            offsets=self.offsets[:idx],
            deferred_toks=self.deferred_toks[:idx]
        )

    def add_toks(self, toks: Sequence[int]) -> "DecodedTokenStream":
        tokens = list(self.tokens)
        decoded = list(self.decoded)
        decoded_raw = list(self.decoded_raw)
        offsets = list(self.offsets)
        deferred_toks = list(self.deferred_toks)

        for tok in toks:
            if len(deferred_toks) > 0 and deferred_toks[-1] > 0:
                tok_to_decode = tokens[-deferred_toks[-1]:] + [tok]
            else:
                tok_to_decode = tok

            this_decoded = self.tokenizer.decode(tok_to_decode, clean_up_tokenization_spaces=False)

            tokens.append(tok)

            if len(this_decoded) > 0 and this_decoded[-1] == chr(0xFFFD):  # Partial token
                decoded.append("")
                decoded_raw.append(this_decoded)
                offsets.append(offsets[-1] if len(offsets) > 0 else 0)
                deferred_toks.append(deferred_toks[-1] + 1 if len(deferred_toks) > 0 else 1)
            else:  # Full token sequence: attribute full text to the final token
                decoded_raw.append(this_decoded)

                # Decide whether to add a space before the current token
                if len(tokens) > 1:  # Not the first token that we added
                    prev_tok_idx = len(tokens) - deferred_toks[-1] - 2
                    if prev_tok_idx > 0 and deferred_toks[prev_tok_idx - 1] > 0:
                        prev_tok_idx_s = prev_tok_idx - deferred_toks[prev_tok_idx - 1]
                    else:
                        prev_tok_idx_s = prev_tok_idx

                    prev_tok_and_this_decoded = self.tokenizer.decode(tokens[prev_tok_idx_s:],
                                                                      clean_up_tokenization_spaces=False)
                    if prev_tok_and_this_decoded == decoded_raw[prev_tok_idx] + this_decoded:
                        pass  # No space added
                    elif prev_tok_and_this_decoded == decoded_raw[prev_tok_idx] + " " + this_decoded:
                        this_decoded = " " + this_decoded  # Tokenizer added a space in between, attribute it to this_decoded
                    else:
                        if WARN_ON_NONSTANDARD_SPACING:
                            print("WARNING: Nonstandard Spacing:")
                            print(f"{tokens[prev_tok_idx_s:prev_tok_idx + 1]} -> {decoded_raw[prev_tok_idx]}")
                            print(f"{tok_to_decode} -> {this_decoded}")
                            print(f"{tokens[prev_tok_idx_s:]} -> {prev_tok_and_this_decoded}")
                        if THROW_ON_NONSTANDARD_SPACING:
                            raise ValueError("Nonstandard spacing")

                decoded.append(this_decoded)
                offsets.append(offsets[-1] + len(this_decoded) if len(offsets) > 0 else len(this_decoded))
                deferred_toks.append(0)

        return DecodedTokenStream(
            tokenizer=self.tokenizer,
            tokens=tuple(tokens),
            decoded=tuple(decoded),
            decoded_raw=tuple(decoded_raw),
            offsets=tuple(offsets),
            deferred_toks=tuple(deferred_toks)
        )

    def char_to_tok(self, char_idx: int) -> int:
        result = bisect.bisect(self.offsets, char_idx)
        if result == len(self) or char_idx < 0:
            raise IndexError(char_idx)
        else:
            return result

