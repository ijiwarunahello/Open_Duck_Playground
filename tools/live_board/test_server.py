import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import make_app  # noqa: E402


def test_routes(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "clip_0.mp4").write_bytes(b"0123456789")
    c = make_app(tmp_path).test_client()
    assert c.get("/status.json").status_code == 404
    (tmp_path / "status.json").write_text('{"latest_step": 0}')
    r = c.get("/status.json")
    assert r.json == {"latest_step": 0}
    assert r.headers["Access-Control-Allow-Origin"] == "*" and r.headers["Cache-Control"] == "no-store"
    r = c.get("/clips/clip_0.mp4", headers={"Range": "bytes=2-4"})
    assert r.status_code == 206 and r.data == b"234"  # Safari needs range requests for video
    assert c.get("/clips/../status.json").status_code == 404
    (tmp_path / "clips" / "clip_0.json").write_text('{"fps": 30}')
    r = c.get("/clips/clip_0.json")
    assert r.json == {"fps": 30} and r.headers["Access-Control-Allow-Origin"] == "*"
