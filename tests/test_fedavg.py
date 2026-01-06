import torch
from fedlora.fl.aggregation import weighted_average_state

def test_weighted_average_state():
    s1 = {"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([0.0])}
    s2 = {"a": torch.tensor([3.0, 3.0]), "b": torch.tensor([2.0])}
    out = weighted_average_state([s1, s2], [1, 3])
    assert torch.allclose(out["a"], torch.tensor([2.5, 2.5]))
    assert torch.allclose(out["b"], torch.tensor([1.5]))

