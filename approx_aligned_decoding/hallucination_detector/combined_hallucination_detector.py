from approx_aligned_decoding.hallucination_detector.hallucination_detector import HallucinationDetector, \
    HallucinationDetectionResult


class CombinedHallucinationDetector(HallucinationDetector):
    def __init__(self, *detectors: HallucinationDetector):
        super().__init__()
        self.detectors = detectors

    def detect(self, left_context: str, right_context: str, generated_text: str, is_end: bool,
               index_previously_checked: int) -> HallucinationDetectionResult:
        earliest_detection = None

        for detector in self.detectors:
            detection = detector.detect(left_context, right_context, generated_text, is_end, index_previously_checked)
            if detection.hallucination:
                if earliest_detection is None or earliest_detection.index_of_hallucination is None or (
                        detection.index_of_hallucination is not None and
                        detection.index_of_hallucination < earliest_detection.index_of_hallucination):
                    earliest_detection = detection

        return earliest_detection if earliest_detection is not None else HallucinationDetectionResult(False, None)
