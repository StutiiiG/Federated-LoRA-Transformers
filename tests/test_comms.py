import torch
from fedlora.comms import bytes_of_state

def test_bytes_of_state():
    sd = {"x": torch.zeros((10,), dtype=torch.float32)}  # 10 * 4 bytes
    assert bytes_of_state(sd) == 40

