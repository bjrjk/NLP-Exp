from typing import List, Tuple
import numpy as np

def CalculateEntropy(tokenFreq: List[Tuple[str, int]]) -> float:
    elemCount = 0
    entropy = 0.0
    for item in tokenFreq:
        elemCount += item[1]

    for item in tokenFreq:
        p = float(item[1]) / elemCount
        log2p = np.log2(p)
        entropy -= p * log2p

    return entropy