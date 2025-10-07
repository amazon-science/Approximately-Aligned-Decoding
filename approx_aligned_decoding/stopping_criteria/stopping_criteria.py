import abc


class StoppingCriteria(abc.ABC):
    @abc.abstractmethod
    def should_stop(self, left_context: str, right_context: str, generated: str) -> bool:
        pass
