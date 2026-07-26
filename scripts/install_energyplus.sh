#!/usr/bin/env bash
# Download an official NREL EnergyPlus build into ./vendor/.
#
# EnergyPlus is not a pip package: the pyenergyplus Python API ships inside the
# install directory next to libenergyplusapi. This script fetches a release
# archive and unpacks it locally, so nothing is installed system-wide and
# nothing needs sudo. ecoloop finds it automatically (see
# src/ecoloop/energyplus_locate.py).
#
#   ./scripts/install_energyplus.sh              # default version, auto platform
#   ./scripts/install_energyplus.sh 25.1.0       # a specific version
#
# Already have EnergyPlus? Skip this and point ecoloop at it instead:
#   export ECOLOOP_ENERGYPLUS_DIR=/Applications/EnergyPlus-25-2-0

set -euo pipefail

VERSION="${1:-25.2.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$REPO_ROOT/vendor"

# The release tags carry a build hash, so the exact asset name is resolved from
# the releases API rather than guessed.
API="https://api.github.com/repositories/14620185/releases"

case "$(uname -s)" in
  Darwin) PLATFORM="Darwin" ;;
  Linux)  PLATFORM="Linux" ;;
  *)      echo "unsupported platform $(uname -s) — install EnergyPlus manually and set ECOLOOP_ENERGYPLUS_DIR" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  arm64|aarch64) ARCH="arm64" ;;
  x86_64|amd64)  ARCH="x86_64" ;;
  *)             echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

if [ -d "$VENDOR" ] && compgen -G "$VENDOR/EnergyPlus-*/pyenergyplus/api.py" > /dev/null; then
  echo "EnergyPlus is already installed:"
  ls -d "$VENDOR"/EnergyPlus-*/ | sed 's/^/  /'
  echo "Delete that directory first if you want to reinstall."
  exit 0
fi

echo "Looking up EnergyPlus $VERSION for $PLATFORM/$ARCH ..."
URL="$(curl -fsSL "$API?per_page=100" \
  | python3 -c "
import json, sys
version = '$VERSION'
platform, arch = '$PLATFORM', '$ARCH'
releases = json.load(sys.stdin)
for release in releases:
    if release.get('tag_name') != 'v' + version:
        continue
    for asset in release.get('assets', []):
        name = asset['name']
        if platform in name and arch in name and name.endswith('.tar.gz'):
            print(asset['browser_download_url'])
            sys.exit(0)
sys.exit('no $PLATFORM/$ARCH .tar.gz asset found for v' + version)
")"

ARCHIVE="$VENDOR/energyplus-download.tar.gz"
mkdir -p "$VENDOR"
echo "Downloading $(basename "$URL") (~200 MB) ..."
curl -fL --progress-bar --retry 3 -o "$ARCHIVE" "$URL"

echo "Extracting ..."
tar xzf "$ARCHIVE" -C "$VENDOR"
rm -f "$ARCHIVE"

ROOT="$(ls -d "$VENDOR"/EnergyPlus-*/ | head -1)"
if [ ! -f "$ROOT/pyenergyplus/api.py" ]; then
  echo "extraction did not produce a pyenergyplus package — check $ROOT" >&2
  exit 1
fi

echo
echo "EnergyPlus installed: $ROOT"
echo "Weather files bundled with it: $(ls "$ROOT/WeatherData"/*.epw 2>/dev/null | wc -l | tr -d ' ')"
echo
echo "Next:  python -m ecoloop doctor"
