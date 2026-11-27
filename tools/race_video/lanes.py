"""Race lanes: checkpoints of run backlash_20261127_0141 (flat_terrain_backlash, 300M steps)."""
import os
from pathlib import Path

CKPT = Path(os.environ.get(
    "RACE_CKPT_DIR",
    "/home/iri/Workspaces/src/github.com/ijiwarunahello/Open_Duck_Playground/checkpoints/backlash_20261127_0141",
))

# (onnx, lane sign text). Times are wall-clock minutes since training start (01:44).
LANES = [
    (str(CKPT / "2026_11_27_014427_0.onnx"), "れんしゅう 0ぷん"),
    (str(CKPT / "2026_11_27_021146_42926080.onnx"), "27ふん"),
    (str(CKPT / "2026_11_27_022458_64389120.onnx"), "40ぷん"),
    (str(CKPT / "2026_11_27_023811_85852160.onnx"), "54ふん"),
    (str(CKPT / "2026_11_27_031742_150241280.onnx"), "1じかん37ふん"),
    (str(CKPT / "2026_11_27_045005_300482560.onnx"), "3じかん"),
]

# Chosen with push_probe.py (2026-11-27): at vy=0.8 only the 0-min lane falls and every
# walking lane still covers > 1.2 m in 20 s; at 1.0 the 3-hour lane drops to 0.73 m.
PUSH_TIME_S = 5.0  # seconds after the start signal
PUSH_VEL = (0.0, 0.8)  # m/s added to the base, same as training pushes (range 0.1-1.0)
GOAL_X = 1.2  # metres from the start line
