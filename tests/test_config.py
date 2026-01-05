from fl_lora.config import load_config

def test_load_config(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("num_clients: 2\nnum_rounds: 1\nclient_frac: 1.0\nseed: 123\n")
    cfg = load_config(p)
    assert cfg.num_clients == 2
    assert cfg.seed == 123
