from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.config import DOPAConfig
from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.dopa import DOPACoderN1
from dopa_coder_n1.model.skeleton import SkeletonBatch
from dopa_coder_n1.training.checkpoint import save_checkpoint


def main() -> None:
    cfg = DOPAConfig.from_yaml(ROOT / "configs" / "tiny.yaml")
    cfg.offload.enabled = False
    cfg.model.max_seq_len = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DOPACoderN1(cfg).to(device)
    tok = ByteTokenizer()
    skel_tok = SkeletonTokenizer(cfg.dopa.skeleton_vocab_size)
    text = "def solve():\n    print(1)\n"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    ids = ids[:32] + [tok.pad_id] * max(0, 32 - len(ids))
    x = torch.tensor([ids[:32]], dtype=torch.long, device=device)
    skel = SkeletonBatch(
        token_ids=torch.tensor(
            [skel_tok.encode({"name": "simple_function", "steps": [{"op": "emit_output"}]}, max_len=64)],
            dtype=torch.long,
            device=device,
        )
    )
    out = model(x, labels=x, skeleton=skel, return_aux=True)
    assert out.loss is not None
    out.loss.backward()
    generated = model.generate(x[:, :8], max_new_tokens=4, skeleton=skel, temperature=0.0)
    save_checkpoint(ROOT / "runs" / "smoke" / "last.pt", model, None, 1, cfg)
    print({"loss": float(out.loss.detach().cpu()), "generated_shape": list(generated.shape)})


if __name__ == "__main__":
    main()
