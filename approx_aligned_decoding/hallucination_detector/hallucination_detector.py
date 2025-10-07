import abc

from dataclasses import dataclass
from typing import Optional


@dataclass
class HallucinationDetectionResult:
    hallucination: bool
    index_of_hallucination: Optional[int]  # Index within (prev_generated_text + new_generated_text) where hallucination starts

class HallucinationDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, left_context: str, right_context: str, generated_text: str, is_end: bool, index_previously_checked: int) -> HallucinationDetectionResult:
        """
        Note that the left context might contain the beginning of the hallucination.
        For example, it might contain "System.out.", and the generated text might be "printWord()" or such.
        index_of_hallucination should be 5, as the "W" is the first character for which there is no valid completion.
        If the index is 0, that would prevent "println"; if the index is 6, that would incorrectly allow "printWeekly".
        Generation might also be doomed from the start; i.e if a hallucination is in the left context, and the hallucinated
        identifier is continued into the generation. In that case, it is not necessary to detect a hallucination, but
        if one is detected, set hallucination=True, index_of_hallucination=None.
        Same if the hallucination is contained entirely within generated_text[:index_previously_checked].
        
        :param left_context: User-supplied left context
        :param right_context: User-supplied right context
        :param generated_text: Text the LLM generated
        :param is_end: Whether the generation is the last one (and the right-context must be taken into account)
        :param index_previously_checked: If calling this function multiple times on successively-built generated-text; 
        length of generated_text previously for which there was no hallucination 
        """""
        raise NotImplemented
