from approx_aligned_decoding.backtracking_strategy import BacktrackingStrategy


class NoBacktrackingStrategy(BacktrackingStrategy):
    """
    Unconstrained generation
    """
    def do_whole_sample_hallu_check(self) -> bool:
        return False
