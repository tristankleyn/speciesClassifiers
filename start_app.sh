#!/bin/bash
# speciesClassifiers app launcher (macOS / Linux): run ./start_app.sh (on macOS, double-click start_app.command).
# The first start downloads Python and the app's packages into this folder (a few minutes);
# nothing is installed system-wide apart from the small "uv" tool, if you don't have it yet.
cd "$(dirname "$0")" || exit 1
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "Installing uv, a small tool that sets up Python just for this app..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
echo "Starting speciesClassifiers. The first start takes a few minutes; the app opens in your browser."
echo "Keep this window open while you use the app."
exec uv run --project python --extra app --python 3.12 speciesclassifiers app "$@"
