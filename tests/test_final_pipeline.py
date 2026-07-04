from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.training.validation import validate_training_config
from scripts.report import render_report


def test_training_config_validation_accepts_tiny_unit():
    cfg = DOPAConfig.from_yaml("configs/tiny_unit.yaml")
    errors = validate_training_config(cfg)
    assert errors == []


def test_report_renderer_outputs_markdown():
    md = render_report(
        {
            "config": "cfg.yaml",
            "out_dir": "runs/x",
            "stages": [{"stage": "stage1", "step": 1, "loss": 1.0, "lm_loss": 1.0}],
            "eval": {"loss": 1.0, "perplexity": 2.7, "samples": [{"prompt": "def", "completion": "def pass"}]},
            "writeback": {"enabled": True, "count": 2, "format": "int4"},
        }
    )
    assert "DOPA Training Report" in md
    assert "stage1" in md
    assert "Perplexity" in md
