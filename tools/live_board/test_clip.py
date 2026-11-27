import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip import H, W, select_lanes, sign_tracks  # noqa: E402

import numpy as np  # noqa: E402


def test_select_lanes():
    assert select_lanes(1) == [0]
    assert select_lanes(3) == [0, 1, 2]
    assert select_lanes(15) == [0, 4, 7, 10, 14]  # round half to even at 3.5 -> 4, 10.5 -> 10
    s = select_lanes(6)
    assert s[0] == 0 and s[-1] == 5 and len(s) == len(set(s)) == 5


def test_sign_tracks_fractions_and_offscreen():
    frames = np.zeros((2, 2, 7))
    frames[1, 1, 0] = 99.0  # second lane leaves the screen in frame 1

    def project(p, cam, w, h, fovy):
        return (w / 2 + p[1] * 100, h / 2 - p[2] * 100) if p[0] < 50 else (-1, -1)

    t = sign_tracks(frames, 2, None, 45.0, project)
    assert t[0][0] == [0.5, round((H / 2 - 28) / H, 4)]
    assert t[0][1] == [0.5, round((H / 2 - 46) / H, 4)]  # odd lane sits higher
    assert t[1][1] is None
