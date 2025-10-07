import requests

from approx_aligned_decoding.hallucination_detector.hallucination_detector import HallucinationDetector, \
    HallucinationDetectionResult


class NetworkHallucinationDetector(HallucinationDetector):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def detect(self, left_context: str, right_context: str, generated_text: str, is_end: bool,
               index_previously_checked: int) -> HallucinationDetectionResult:
        result = requests.request(method="POST",
                                  url="http://" + self.path,
                                  headers={"Content-Type": "application/json"},
                                  json={
                                      "left_context": left_context + generated_text[:index_previously_checked],
                                      "right_context": right_context,
                                      "generation": generated_text[index_previously_checked:],
                                      "is_end": is_end
                                  }).json()

        index = result.get("index_of_hallucination")
        if index is not None:
            index = index + index_previously_checked

        return HallucinationDetectionResult(result["hallucination"], index)
