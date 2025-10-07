import unittest

from transformers import AutoTokenizer

from approx_aligned_decoding.generator import get_hallucination_indices
from approx_aligned_decoding.hallucination_detector.banned_text_hallucination_detector import \
    BannedTextHallucinationDetector


class TestHallucinationIndices(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "Salesforce/codegen-350M-multi"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def test_hallucination_indices(self):
        prefix = "# Print Hello World \n"
        suffix = "\n"
        prev_generated_text = "def "
        encoding = self.tokenizer(["foo(bar):\n  print('Hello')<|endoftext|>a,", "foo(bar, baz):\n  print('Hello')", "foo(bar):\n  print('Hello, world')<|endoftext|>"], return_tensors="pt")

        hallucination_indices = get_hallucination_indices(
            input_ids=encoding.input_ids, hallucination_detector=BannedTextHallucinationDetector(","),
            left_context=prefix, right_context=suffix, prev_generated_text=prev_generated_text,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(hallucination_indices, [None, 3, 9])