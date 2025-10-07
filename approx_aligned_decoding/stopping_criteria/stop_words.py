from approx_aligned_decoding.stopping_criteria.stopping_criteria import StoppingCriteria


class StopWords(StoppingCriteria):

    def __init__(self, stop_words):
        self.stop_words = stop_words

    def should_stop(self, left_context: str, right_context: str, generated: str) -> bool:
        for stop_word in self.stop_words:
            if stop_word in generated:
                # Stop generating
                return True
        return False
