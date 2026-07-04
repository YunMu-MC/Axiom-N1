import json

from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer
from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.model.rust_core import RustCoreBackend


def test_rust_core_byte_tokenizer_matches_python_roundtrip():
    backend = RustCoreBackend.default()
    if not backend.available():
        return
    tok = ByteTokenizer(rust_backend=backend)
    text = "def solve():\n    print('hello')\n"

    ids = tok.encode(text, add_bos=True, add_eos=True)

    assert tok.backend_name == "rust"
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == text


def test_skeleton_tokenizer_rust_fast_path_matches_python_encoding():
    backend = RustCoreBackend.default()
    skeleton = {"name": "grid_bfs", "steps": [{"op": "graph_search"}]}
    expected = SkeletonTokenizer(1024, rust_backend=None).encode(skeleton, max_len=32)

    tok = SkeletonTokenizer(1024, rust_backend=backend)
    ids = tok.encode(skeleton, max_len=32)

    assert ids == expected
    if backend.available():
        assert tok.backend_name == "rust"


def test_rust_core_cli_skeleton_encode_keeps_json_sort_order(tmp_path):
    backend = RustCoreBackend.default()
    if not backend.available():
        return
    payload = {"b": 2, "a": 1}
    path = tmp_path / "skeleton.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    ids = backend.encode_skeleton_json(path, vocab_size=512, max_len=16)
    expected = SkeletonTokenizer(512, rust_backend=None).encode(payload, max_len=16)

    assert ids == expected


def test_rust_core_batch_byte_encode_matches_python_lines(tmp_path):
    backend = RustCoreBackend.default()
    if not backend.available():
        return
    lines = ["def a():", "", "    return '你好'"]
    path = tmp_path / "lines.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

    encoded = backend.encode_byte_lines(path, add_bos_first=True, add_eos_each=True)
    python_tok = ByteTokenizer(rust_backend=None)
    expected = [
        python_tok.encode(line, add_bos=(idx == 0), add_eos=True)
        for idx, line in enumerate(lines)
    ]

    assert encoded == expected
