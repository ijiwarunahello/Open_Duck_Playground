#!/usr/bin/env bash
# Start a new training run and point the live board at it.
#   tools/live_board/start_live.sh [NUM_TIMESTEPS]
# Training runs in the main Playground clone (PLAYGROUND), the watcher from this checkout.
# The board server serves $BOARD/current, a symlink this script repoints to the new run's data.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
PLAYGROUND="${PLAYGROUND:-$HOME/Workspaces/src/github.com/ijiwarunahello/Open_Duck_Playground}"
BOARD="${BOARD:-$HERE/.tmp/live_board}"
STEPS="${1:-300000000}"
RUN="live_$(date +%Y%m%d_%H%M)"
mkdir -p "$PLAYGROUND/checkpoints/$RUN" "$BOARD/$RUN"
ln -sfn "$BOARD/$RUN" "$BOARD/current"
cd "$PLAYGROUND"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 TF_FORCE_GPU_ALLOW_GROWTH=true \
  setsid nohup uv run --no-sync playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash \
  --num_timesteps "$STEPS" --output_dir "checkpoints/$RUN" > "checkpoints/$RUN/train.log" 2>&1 &
echo "training $RUN (pid $!)"
cd "$HERE"
# The watcher only reads events; keep TensorFlow (imported by tensorboard) off the GPU.
exec env CUDA_VISIBLE_DEVICES= uv run --no-sync --with imageio-ffmpeg --with mediapy \
  python tools/live_board/watcher.py --run "$PLAYGROUND/checkpoints/$RUN" --out "$BOARD/$RUN" --total-steps "$STEPS"
