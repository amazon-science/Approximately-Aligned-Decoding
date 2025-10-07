from approx_aligned_decoding.hallucination_detector.hallucination_detector import HallucinationDetector, \
    HallucinationDetectionResult


class BannedTextHallucinationDetector(HallucinationDetector):
    """
    Doesn't allow banned_text as a substring
    """
    def __init__(self, banned_text):
        super().__init__()
        self.banned_text = banned_text

    def detect(self, left_context: str, right_context: str, generated_text: str, is_end: bool,
               index_previously_checked: int) -> HallucinationDetectionResult:

        amount_of_left_context = min(max(len(self.banned_text) - index_previously_checked - 1, 0), len(left_context))
        text_to_check = generated_text
        if amount_of_left_context > 0:
            text_to_check = left_context[-amount_of_left_context:] + text_to_check

        first_idx_to_check = index_previously_checked - len(self.banned_text) + 1 + amount_of_left_context

        if amount_of_left_context > 0:
            assert first_idx_to_check == 0

        # Last character in banned text
        idx = text_to_check.find(self.banned_text, first_idx_to_check) + len(self.banned_text) - 1

        if idx == -1:
            return HallucinationDetectionResult(hallucination=False, index_of_hallucination=None)
        else:
            true_idx = idx - amount_of_left_context
            if true_idx < 0:
                return HallucinationDetectionResult(hallucination=True, index_of_hallucination=None)
            else:
                return HallucinationDetectionResult(hallucination=True, index_of_hallucination=true_idx)

