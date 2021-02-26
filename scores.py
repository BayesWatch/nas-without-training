import numpy as np
import torch




def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def random_score(jacob, label=None):
    return np.random.normal()


_scores = {
        'hook_logdet': hooklogdet,
        'random': random_score
        }

def get_score_func(score_name):
    return _scores[score_name]
