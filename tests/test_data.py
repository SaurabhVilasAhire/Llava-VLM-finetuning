"""CPU-only tests for the JSON <-> token-sequence conversion that underlies
training and decoding. These need no GPU and can be run anywhere:

    python -m pytest tests/test_data.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import json2token, token2json  # noqa: E402


def test_flat_object_roundtrip():
    obj = {"name": "Coffee", "price": "3.50"}
    tokens = json2token(obj, sort_json_key=False)
    assert tokens == "<s_name>Coffee</s_name><s_price>3.50</s_price>"
    assert token2json(tokens) == obj


def test_nested_object_roundtrip():
    obj = {"menu": {"name": "Latte", "count": "2"}}
    tokens = json2token(obj, sort_json_key=False)
    assert token2json(tokens) == obj


def test_list_of_objects_roundtrip():
    obj = [{"name": "Tea"}, {"name": "Coffee"}]
    tokens = json2token(obj, sort_json_key=False)
    assert token2json(tokens) == obj


def test_sorted_keys_are_deterministic():
    obj = {"b": "2", "a": "1"}
    tokens = json2token(obj, sort_json_key=True)
    assert tokens == "<s_b>2</s_b><s_a>1</s_a>"  # reverse-sorted keys


def test_unmatched_opening_tag_is_dropped_not_raised():
    tokens = "<s_name>Tea<s_price>2.00</s_price>"
    result = token2json(tokens)
    assert result == {"price": "2.00"}


def test_empty_input_returns_text_sequence():
    assert token2json("") == {"text_sequence": ""}
