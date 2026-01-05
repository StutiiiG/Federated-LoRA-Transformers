import numpy as np
from fl_lora.federated.aggregation import fedavg

def test_fedavg_weighted_average():
    w1 = np.array([1.0, 3.0], dtype=np.float32)
    w2 = np.array([5.0, 7.0], dtype=np.float32)

    out = fedavg([w1, w2], client_sizes=[1, 3])
    # expected = 0.25*w1 + 0.75*w2 = [4.0, 6.0]
    assert np.allclose(out, np.array([4.0, 6.0], dtype=np.float32))

def test_fedavg_shape_mismatch_raises():
    w1 = np.array([1.0, 2.0])
    w2 = np.array([1.0, 2.0, 3.0])
    try:
        fedavg([w1, w2], client_sizes=[1, 1])
        assert False, "expected ValueError"
    except ValueError:
        pass
