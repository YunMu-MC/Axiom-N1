import pytest

torch = pytest.importorskip("torch")

from dopa_coder_n1.training.optim import CPUAdamW, build_optimizer


def test_cpu_adamw_keeps_state_on_cpu():
    model = torch.nn.Linear(4, 2)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.0, state_device="cpu")
    assert isinstance(opt, CPUAdamW)
    x = torch.randn(3, 4)
    loss = model(x).pow(2).mean()
    loss.backward()
    before = model.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, model.weight)
    state = opt.state[model.weight]
    assert state["exp_avg"].device.type == "cpu"
    assert state["master"].device.type == "cpu"
