import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from status import build_status, clip_name, list_checkpoints, practice_days, read_curve  # noqa: E402

RUN = Path(os.environ.get(
    "LIVE_TEST_RUN",
    "/home/iri/Workspaces/src/github.com/ijiwarunahello/Open_Duck_Playground/checkpoints/backlash_20261127_0141",
))
needs_run = pytest.mark.skipif(not RUN.is_dir(), reason="reference run not on this machine")


def test_practice_days():
    assert practice_days(300_000_000) == pytest.approx(69.44, abs=0.01)


def test_ignores_other_files(tmp_path):
    (tmp_path / "2026_11_27_014427_0.onnx").touch()
    (tmp_path / "ONNX.onnx").touch()
    (tmp_path / "2026_11_27_015834_21463040").mkdir()
    assert [s for s, _, _ in list_checkpoints(tmp_path)] == [0]


def test_empty_run(tmp_path):
    st = build_status(tmp_path, tmp_path, curve=[])
    assert st["latest_step"] == 0 and st["clips"] == [] and st["started_at"] is None


@needs_run
def test_reference_run(tmp_path):
    ck = list_checkpoints(RUN)
    assert len(ck) == 15 and ck[-1][0] == 300482560
    (tmp_path / clip_name(0)).touch()
    (tmp_path / clip_name(300482560)).touch()
    st = build_status(RUN, tmp_path, curve=[(0, 1.0)])
    assert st["practice_days"] == pytest.approx(69.6, abs=0.1)
    assert [c["step"] for c in st["clips"]] == [0, 300482560]
    assert st["clips"][-1]["minutes"] == 186  # 01:44:27 -> 04:50:05
    assert st["latest_minutes"] == 186


@needs_run
def test_reference_curve():
    c = read_curve(RUN)
    assert len(c) == 15 and c[0][0] == 0 and c[-1] == (300482560, pytest.approx(277.96, abs=0.01))
