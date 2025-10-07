import itertools

from approx_aligned_decoding.hallucination_detector.hallucination_detector import HallucinationDetector, \
    HallucinationDetectionResult


class OnlyASCIIHallucinationDetector(HallucinationDetector):
    def detect(self, left_context: str, right_context: str, generated_text: str, is_end: bool,
               index_previously_checked: int) -> HallucinationDetectionResult:

        for idx, char in itertools.dropwhile(lambda x: x[0] < index_previously_checked, enumerate(generated_text)):
            if not char.isascii():
                return HallucinationDetectionResult(True, idx)

        return HallucinationDetectionResult(False, None)
