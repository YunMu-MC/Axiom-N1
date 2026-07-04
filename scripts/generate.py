from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.skeleton import SkeletonBatch
from dopa_coder_n1.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with DOPA Coder N1.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="def solve():\n    ")
    parser.add_argument("--skeleton", default=None, help="JSON skeleton string or path to JSON file.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, raw = load_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device)
    tokenizer = ByteTokenizer()
    skel_batch = None
    if args.skeleton:
        text = Path(args.skeleton).read_text(encoding="utf-8") if Path(args.skeleton).exists() else args.skeleton
        skel = json.loads(text)
        skel_tok = SkeletonTokenizer(model.cfg.dopa.skeleton_vocab_size)
        token_ids = torch.tensor([skel_tok.encode(skel)], dtype=torch.long, device=device)
        skel_batch = SkeletonBatch(token_ids=token_ids)
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        skeleton=skel_batch,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
        use_incremental=args.incremental,
    )
    print(tokenizer.decode(out[0].tolist()))
    if "step" in raw:
        print(f"\n[checkpoint_step={raw['step']}]")


if __name__ == "__main__":
    main()
