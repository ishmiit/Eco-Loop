"""Find an EnergyPlus installation and make ``pyenergyplus`` importable.

``pyenergyplus`` is not on PyPI — it ships inside the EnergyPlus install
directory next to ``libenergyplusapi``. Rather than requiring the user to set
PYTHONPATH, we locate the install and inject it into ``sys.path`` at import
time. Search order (first hit wins):

1. ``$ECOLOOP_ENERGYPLUS_DIR``
2. ``<repo>/vendor/EnergyPlus-*``          (scripts/install_energyplus.sh)
3. ``/Applications/EnergyPlus-*``          (macOS installer default)
4. ``/usr/local/EnergyPlus-*``, ``/opt/EnergyPlus-*``, ``C:/EnergyPlusV*``
5. the directory containing an ``energyplus`` binary on ``$PATH``
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import REPO_ROOT


@dataclass(frozen=True)
class EnergyPlusInstall:
    root: Path
    version: str

    @property
    def python_package(self) -> Path:
        return self.root / "pyenergyplus"

    @property
    def executable(self) -> Path:
        exe = self.root / ("energyplus.exe" if os.name == "nt" else "energyplus")
        return exe

    @property
    def weather_dir(self) -> Path:
        return self.root / "WeatherData"

    @property
    def example_dir(self) -> Path:
        return self.root / "ExampleFiles"


def _version_of(root: Path) -> str:
    name = root.name
    for token in ("EnergyPlus-", "EnergyPlusV"):
        if name.startswith(token):
            return name[len(token):].split("-")[0]
    # Fall back to the IDD header. A malformed IDD degrades to "unknown"
    # rather than crashing the locator.
    idd = root / "Energy+.idd"
    if idd.exists():
        try:
            with idd.open(encoding="latin-1") as fh:
                for line in fh:
                    if line.startswith("!IDD_Version"):
                        return line.split()[1].strip()
        except (OSError, IndexError):
            pass
    return "unknown"


def _candidates() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("ECOLOOP_ENERGYPLUS_DIR", "").strip()
    if env:
        out.append(Path(env).expanduser())

    globs = [
        (REPO_ROOT / "vendor", "EnergyPlus-*"),
        (Path("/Applications"), "EnergyPlus-*"),
        (Path("/usr/local"), "EnergyPlus-*"),
        (Path("/opt"), "EnergyPlus-*"),
        (Path("C:/"), "EnergyPlusV*"),
        (Path.home(), "EnergyPlus-*"),
    ]
    for parent, pattern in globs:
        try:
            if parent.is_dir():
                out.extend(sorted(parent.glob(pattern), reverse=True))
        except OSError:
            continue

    which = shutil.which("energyplus")
    if which:
        out.append(Path(which).resolve().parent)
    return out


def find_energyplus() -> EnergyPlusInstall | None:
    for root in _candidates():
        if (root / "pyenergyplus" / "api.py").exists():
            return EnergyPlusInstall(root=root, version=_version_of(root))
    return None


def ensure_importable() -> EnergyPlusInstall | None:
    """Put ``pyenergyplus`` on ``sys.path``. Returns the install, or None."""
    install = find_energyplus()
    if install is None:
        return None
    root = str(install.root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return install


def describe() -> str:
    install = find_energyplus()
    if install is None:
        return (
            "EnergyPlus: NOT FOUND — the surrogate engine will be used.\n"
            "  Install with: ./scripts/install_energyplus.sh\n"
            "  Or point ECOLOOP_ENERGYPLUS_DIR at an existing install."
        )
    return f"EnergyPlus {install.version} at {install.root}"
