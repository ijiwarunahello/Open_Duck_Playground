import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip import lane_label, select_lanes  # noqa: E402


def test_select_lanes():
    assert select_lanes(1) == [0]
    assert select_lanes(3) == [0, 1, 2]
    assert select_lanes(15) == [0, 4, 7, 10, 14]  # round half to even at 3.5 -> 4, 10.5 -> 10
    s = select_lanes(6)
    assert s[0] == 0 and s[-1] == 5 and len(s) == len(set(s)) == 5


def test_lane_label():
    assert lane_label(0) == "0ぷん"
    assert lane_label(186) == "3じかん6ぷん"
