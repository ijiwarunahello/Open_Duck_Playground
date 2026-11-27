import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.mark.parametrize(
    "minutes,text",
    [(0, "れんしゅう 0ぷん"), (27, "れんしゅう 27ふん"), (40, "れんしゅう 40ぷん"), (97, "れんしゅう 1じかん37ふん"),
     (180, "れんしゅう 3じかん"), (61, "れんしゅう 1じかん1ぷん"), (65, "れんしゅう 1じかん5ふん"), (3, "れんしゅう 3ぷん")],
)
def test_clock_text(minutes, text):
    from crowd_video import clock_text
    assert clock_text(minutes) == text


def test_stages_and_timeline():
    import crowd_video as C
    assert [m for _, m in C.STAGES] == [0, 27, 40, 97, 180]
    assert C.T_TITLE == 4.0 and C.STAGE_S == 8.0 and C.T_END == 50.0
    assert C.T_TITLE + len(C.STAGES) * C.STAGE_S == 44.0
    assert C.TITLE == "みんなで れんしゅう"
    assert C.SUMMARY_TITLE == "ころんだ かず"
    assert C.SUMMARY_LINE == "8192たいで 3じかん = 69にちぶんの れんしゅう"


def test_clock_counts_up_between_stages():
    import crowd_video as C
    start = C.T_TITLE + C.STAGE_S  # second stage begins
    assert C.clock_minutes(start - 0.01) == 0
    assert 0 < C.clock_minutes(start + C.COUNT_UP_S / 2) < 27
    assert C.clock_minutes(start + C.COUNT_UP_S + 0.01) == 27


def test_bar_heights_follow_counts():
    from crowd_video import bar_heights
    assert bar_heights([171, 143, 100, 0], max_px=300) == [300, 251, 175, 0]
