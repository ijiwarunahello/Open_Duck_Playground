"""Serve the live board data on 127.0.0.1 only (Windows `tailscale serve` exposes it to the tailnet).

    python tools/live_board/server.py --dir .tmp/live_board [--port 8765]
GET /status.json, /clips/<file>.mp4 and / (board.html, for opening the board from here).
"""
import argparse
from pathlib import Path

from flask import Flask, abort, send_from_directory

HERE = Path(__file__).resolve().parent


def make_app(data_dir):
    data_dir = Path(data_dir).absolute()  # not resolved: "current" is a symlink that start_live.sh repoints
    app = Flask(__name__)

    @app.after_request
    def headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/status.json")
    def status():
        if not (data_dir / "status.json").is_file():
            abort(404)
        return send_from_directory(data_dir, "status.json", mimetype="application/json")

    @app.get("/clips/<name>")
    def clip(name):
        return send_from_directory(data_dir / "clips", name, mimetype="video/mp4", conditional=True)

    @app.get("/")
    def board():
        return send_from_directory(HERE, "board.html")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".tmp/live_board")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    make_app(a.dir).run(host="127.0.0.1", port=a.port, threaded=True)
