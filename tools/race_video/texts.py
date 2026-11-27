"""On-screen text for both videos, in two scripts.

kana:  all hiragana (the first version).
kanji: kanji taught by the end of junior high school, every kanji with ruby.
Ruby markup: {base|ruby}, e.g. "{練習|れんしゅう} 0{分|ぷん}".
"""
import re

_RUBY = re.compile(r"\{([^|}]+)\|([^}]+)\}")


def parse_ruby(s):
    """-> [(base, ruby or None)]"""
    out, pos = [], 0
    for m in _RUBY.finditer(s):
        if m.start() > pos:
            out.append((s[pos:m.start()], None))
        out.append((m.group(1), m.group(2)))
        pos = m.end()
    if pos < len(s):
        out.append((s[pos:], None))
    return out


def plain(s):
    return "".join(b for b, _ in parse_ruby(s))


def _fun(m):
    return "ふん" if m % 10 in (2, 5, 7, 9) else "ぷん"


def clock_text(minutes, script="kana"):
    h, m = divmod(int(minutes), 60)
    if script == "kana":
        if m == 0 and h:
            return f"れんしゅう {h}じかん"
        return f"れんしゅう {h}じかん{m}{_fun(m)}" if h else f"れんしゅう {m}{_fun(m)}"
    hours = f"{h}{{時間|じかん}}" if h else ""
    mins = f"{m}{{分|{_fun(m)}}}" if (m or not h) else ""
    return f"{{練習|れんしゅう}} {hours}{mins}"


KANA = {
    "title": "アヒルの かけっこ",
    "go": "よーい、ドン!",
    "push": "えいっ!",
    "slow": "スロー",
    "lanes": ["れんしゅう 0ぷん", "27ふん", "40ぷん", "54ふん", "1じかん37ふん", "3じかん"],
    "fallen": "ころんだ",
    "still": "うごかない",
    "goal": "ゴール!",
    "midway": "とちゅう",
    "end": ("3じかんで 69にちぶん れんしゅうしたよ", "となりの duck も、", "こうして れんしゅうした AI で あるいています"),
    "crowd": "8192たいで いっしょに れんしゅう",
    "crowd_title": "みんなで れんしゅう",
    "fall_count": "ころんだ",
    "summary_title": "ころんだ かず",
    "summary_line": "8192たいで 3じかん = 69にちぶんの れんしゅう",
}

KANJI = {
    "title": "アヒルの かけっこ",
    "go": "よーい、ドン!",
    "push": "えいっ!",
    "slow": "スロー",
    "lanes": ["{練習|れんしゅう} 0{分|ぷん}", "27{分|ふん}", "40{分|ぷん}", "54{分|ふん}", "1{時間|じかん}37{分|ふん}", "3{時間|じかん}"],
    "fallen": "{転|ころ}んだ",
    "still": "{動|うご}かない",
    "goal": "ゴール!",
    "midway": "{途中|とちゅう}",
    "end": ("3{時間|じかん}で 69{日分|にちぶん} {練習|れんしゅう}したよ", "{隣|となり}の duck も、",
            "こうして {練習|れんしゅう}した AI で {歩|ある}いています"),
    "crowd": "8192{体|たい}で {一緒|いっしょ}に {練習|れんしゅう}",
    "crowd_title": "みんなで {練習|れんしゅう}",
    "fall_count": "{転|ころ}んだ",
    "summary_title": "{転|ころ}んだ {数|かず}",
    "summary_line": "8192{体|たい}で 3{時間|じかん} = 69{日分|にちぶん}の {練習|れんしゅう}",
}

SCRIPTS = {"kana": KANA, "kanji": KANJI}
