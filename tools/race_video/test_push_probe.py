import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(REPO)


def test_probe_reproduces_chosen_push_in_render_timeline():
    from lanes import PUSH_TIME_S, PUSH_VEL
    import push_probe
    rows = push_probe.trial(PUSH_VEL[1], PUSH_TIME_S)
    assert [fallen for _, fallen, _ in rows] == [True, False, False, False, False, False]
