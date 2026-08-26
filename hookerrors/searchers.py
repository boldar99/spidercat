import itertools

class ExhaustiveSearcher:
    """
    Evaluates all possible splits up to max_split_size.
    Returns all safe splits found.
    """
    def __init__(self, max_split_size=None):
        self.max_split_size = max_split_size

    def search(self, support, evaluator):
        w = len(support)
        safe_splits = []
        max_size = w // 2 if self.max_split_size is None else min(w // 2, self.max_split_size)
        
        for split_size in range(1, max_size + 1):
            if split_size < 2:
                continue
            for subset in itertools.combinations(support, split_size):
                # Avoid redundant complementary splits if exactly at halfway
                if split_size == w // 2 and subset[0] != support[0]:
                    continue
                    
                if evaluator(subset):
                    safe_splits.append(subset)
                    
        return safe_splits

class EarlyExitSearcher(ExhaustiveSearcher):
    """
    Evaluates splits up to max_split_size but stops searching a generator
    as soon as it finds `max_splits` safe splits.
    """
    def __init__(self, max_splits=1, max_split_size=None):
        super().__init__(max_split_size)
        self.max_splits = max_splits

    def search(self, support, evaluator):
        w = len(support)
        safe_splits = []
        max_size = w // 2 if self.max_split_size is None else min(w // 2, self.max_split_size)
        
        for split_size in range(1, max_size + 1):
            if split_size < 2:
                continue
            for subset in itertools.combinations(support, split_size):
                if split_size == w // 2 and subset[0] != support[0]:
                    continue
                    
                if evaluator(subset):
                    safe_splits.append(subset)
                    if len(safe_splits) >= self.max_splits:
                        return safe_splits
                        
        return safe_splits
