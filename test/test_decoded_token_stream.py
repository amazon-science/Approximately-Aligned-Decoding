import unittest

from transformers import AutoTokenizer

from approx_aligned_decoding.decoded_token_stream import DecodedTokenStream


class TestDecodedTokenStream(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

    def test_normal_tokens(self):
        ts = DecodedTokenStream(self.tokenizer)
        ts = ts.add_toks([3435, 532])
        self.assertEqual(ts.tokens, (3435, 532))
        self.assertEqual(ts.decoded, (" characters", " -"))
        self.assertEqual(ts.char_to_tok(0), 0)
        self.assertEqual(ts.char_to_tok(10), 0)
        self.assertEqual(ts.char_to_tok(11), 1)
        self.assertEqual(ts.char_to_tok(12), 1)
        self.assertRaises(IndexError, lambda: ts.char_to_tok(13))

    def test_partial_toks_1(self):
        ts = DecodedTokenStream(self.tokenizer)
        ts = ts.add_toks([326, 447])

        self.assertEqual(ts.decoded, (" that", ""))
        self.assertEqual(ts.char_to_tok(4), 0)
        self.assertRaises(IndexError, lambda: ts.char_to_tok(5))

        ts = ts.add_toks([247])
        self.assertEqual(ts.decoded, (" that", "", "’"))
        self.assertEqual(ts.char_to_tok(4), 0)
        self.assertEqual(ts.char_to_tok(5), 2)

    def test_partial_toks_2(self):
        ts = DecodedTokenStream(self.tokenizer)
        ts = ts.add_toks([326, 447, 247])

        self.assertEqual(ts.decoded, (" that", "", "’"))
        self.assertEqual(ts.char_to_tok(4), 0)
        self.assertEqual(ts.char_to_tok(5), 2)

        ts = ts.truncate(2)
        self.assertEqual(ts.decoded, (" that", ""))
        self.assertEqual(ts.char_to_tok(4), 0)
        self.assertRaises(IndexError, lambda: ts.char_to_tok(5))

        ts = ts.add_toks([247])
        self.assertEqual(ts.decoded, (" that", "", "’"))
        self.assertEqual(ts.char_to_tok(4), 0)
        self.assertEqual(ts.char_to_tok(5), 2)
