import numpy as np
from fedlora.data import dirichlet_partition

def test_dirichlet_partition_preserves_size_and_no_dupes():
    labels = np.array([0]*50 + [1]*50)
    splits = dirichlet_partition(labels, num_clients=5, alpha=0.1, seed=123)
    all_idx = np.concatenate(splits)
    assert len(all_idx) == len(labels)
    assert len(np.unique(all_idx)) == len(labels)

