from dopa_coder_n1.data.tokenizer import ByteTokenizer
from dopa_coder_n1.data.skeleton_tokenizer import SkeletonTokenizer


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    text = "def solve():\n    print('你好')"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == text


def test_skeleton_tokenizer_stable_length():
    tok = SkeletonTokenizer(1024)
    ids = tok.encode({"name": "grid_bfs", "steps": [{"op": "graph_search"}]}, max_len=32)
    assert len(ids) == 32
    assert ids[0] == 1
