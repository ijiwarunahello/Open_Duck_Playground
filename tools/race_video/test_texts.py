import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from texts import KANA, KANJI, clock_text, parse_ruby, plain  # noqa: E402

KANJI_RE = re.compile(r"[一-鿿]")
# All taught by the end of junior high school (常用漢字).
ALLOWED = set("練習分時間日体一緒転動途中隣歩数")


def _all_strings(t):
    out = []
    for v in t.values():
        out.extend(v if isinstance(v, (list, tuple)) else [v])
    return out


def test_parse_ruby():
    assert parse_ruby("{練習|れんしゅう} 0{分|ぷん}") == [("練習", "れんしゅう"), (" 0", None), ("分", "ぷん")]
    assert parse_ruby("よーい、ドン!") == [("よーい、ドン!", None)]
    assert plain("{隣|となり}の duck も、") == "隣の duck も、"


def test_every_kanji_has_ruby_and_is_allowed():
    strings = _all_strings(KANJI) + [clock_text(m, "kanji") for m in (0, 27, 40, 97, 180, 13)]
    for s in strings:
        for base, ruby in parse_ruby(s):
            ks = KANJI_RE.findall(base)
            if ks:
                assert ruby, f"kanji without ruby in {s!r}"
                assert set(ks) <= ALLOWED, s


def test_kana_version_has_no_kanji():
    for s in _all_strings(KANA) + [clock_text(m, "kana") for m in (0, 97)]:
        assert not KANJI_RE.search(s), s


def test_kanji_texts():
    assert plain(KANJI["end"][0]) == "3時間で 69日分 練習したよ"
    assert plain(KANJI["end"][1]) + plain(KANJI["end"][2]) == "隣の duck も、こうして 練習した AI で 歩いています"
    assert plain(KANJI["crowd"]) == "8192体で 一緒に 練習"
    assert [plain(s) for s in KANJI["lanes"]] == ["練習 0分", "27分", "40分", "54分", "1時間37分", "3時間"]
    assert plain(KANJI["summary_line"]) == "8192体で 3時間 = 69日分の 練習"


@pytest.mark.parametrize(
    "minutes,text",
    [(0, "{練習|れんしゅう} 0{分|ぷん}"), (27, "{練習|れんしゅう} 27{分|ふん}"), (97, "{練習|れんしゅう} 1{時間|じかん}37{分|ふん}"),
     (180, "{練習|れんしゅう} 3{時間|じかん}")],
)
def test_clock_text_kanji(minutes, text):
    assert clock_text(minutes, "kanji") == text


def test_kana_matches_existing_strings():
    import render_race as R
    assert KANA["end"] == R.END_LINES and KANA["crowd"] == R.CROWD_TEXT
    assert clock_text(97, "kana") == "れんしゅう 1じかん37ふん"
