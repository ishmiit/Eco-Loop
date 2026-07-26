#!/usr/bin/env bash
# One-command setup: virtualenv, dependencies, EnergyPlus, LLM, weather.
#
#   ./scripts/setup.sh
#
# Every step is skippable and idempotent — re-running it is safe. Nothing here
# needs sudo, and nothing is installed outside the repo except the Ollama
# binary (via Homebrew) if you ask for it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
MODEL="${ECOLOOP_LLM_MODEL:-qwen2.5:3b}"

step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

step "1/5  Python environment"
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
  echo "  created .venv with $("$PYTHON" --version)"
else
  echo "  .venv already exists"
fi
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt
echo "  dependencies installed"

step "2/5  EnergyPlus"
if compgen -G "vendor/EnergyPlus-*/pyenergyplus/api.py" > /dev/null; then
  echo "  already present: $(ls -d vendor/EnergyPlus-*/ | head -1)"
elif [ -n "${ECOLOOP_ENERGYPLUS_DIR:-}" ]; then
  echo "  using ECOLOOP_ENERGYPLUS_DIR=$ECOLOOP_ENERGYPLUS_DIR"
else
  ./scripts/install_energyplus.sh
fi

step "3/5  Weather"
if compgen -G "models/weather/*.epw" > /dev/null; then
  echo "  already present: $(ls models/weather/*.epw | head -1 | xargs basename)"
else
  mkdir -p models/weather
  URL="https://climate.onebuilding.org/WMO_Region_2_Asia/IND_India/TN_Tamil_Nadu/IND_TN_Chennai.Intl.AP.432790_TMYx.2009-2023.zip"
  echo "  fetching Chennai TMYx ..."
  if curl -fsSL --retry 2 -o /tmp/ecoloop_epw.zip "$URL"; then
    ( cd models/weather && unzip -o -q /tmp/ecoloop_epw.zip "*.epw" ) && rm -f /tmp/ecoloop_epw.zip
    echo "  installed $(ls models/weather/*.epw | head -1 | xargs basename)"
  else
    echo "  could not download; a weather file bundled with EnergyPlus will be used instead"
  fi
fi

step "4/5  Open-source LLM"
if command -v ollama > /dev/null 2>&1; then
  echo "  ollama found: $(ollama --version 2>/dev/null | head -1)"
else
  if command -v brew > /dev/null 2>&1; then
    echo "  installing ollama via Homebrew ..."
    brew install ollama
  else
    echo "  ollama not installed. Install it from https://ollama.com/download,"
    echo "  or point ecoloop at any OpenAI-compatible server:"
    echo "    export ECOLOOP_LLM_PROVIDER=openai_compat"
    echo "    export ECOLOOP_LLM_URL=http://localhost:8000"
  fi
fi

if command -v ollama > /dev/null 2>&1; then
  if ! curl -fsS -m 3 http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "  starting ollama serve in the background ..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      curl -fsS -m 2 http://127.0.0.1:11434/api/tags > /dev/null 2>&1 && break
      sleep 1
    done
  fi
  if curl -fsS -m 3 http://127.0.0.1:11434/api/tags 2>/dev/null | grep -q "${MODEL%%:*}"; then
    echo "  model $MODEL already pulled"
  else
    echo "  pulling $MODEL (~2 GB) ..."
    ollama pull "$MODEL"
  fi
fi

step "5/5  Check"
PYTHONPATH=src ./.venv/bin/python -m ecoloop doctor

cat <<'EOF'

Run it:
  PYTHONPATH=src ./.venv/bin/python -m ecoloop run --days 3 --ecm-pass
  PYTHONPATH=src ./.venv/bin/python -m ecoloop serve      # http://127.0.0.1:8765

Or use the Makefile:  make run    make serve    make test
EOF
