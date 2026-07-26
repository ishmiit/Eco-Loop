"""Weather file resolution and a minimal EPW reader.

The EPW reader exists for the surrogate engine (and for tests): it needs
outdoor dry bulb, relative humidity, global horizontal irradiance and wind
speed per hour, which are columns 7, 9, 14 and 22 of the EPW data records.
EnergyPlus reads the file itself; it never uses this parser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .config import MODELS_DIR
from .energyplus_locate import find_energyplus

# EPW data-record column indices (0-based) per the EnergyPlus Aux Programs doc.
_COL_YEAR, _COL_MONTH, _COL_DAY, _COL_HOUR = 0, 1, 2, 3
_COL_DRYBULB = 6
_COL_RELHUM = 8
_COL_GHI = 13
_COL_WIND = 21

PREFERRED_EPW = "IND_TN_Chennai.Intl.AP.432790_TMYx.2009-2023.epw"


def resolve_epw(explicit: str | Path = "") -> Path | None:
    """Pick a weather file: explicit path, then the bundled Chennai TMYx, then
    anything in ``models/weather``, then the EnergyPlus WeatherData folder."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"weather file not found: {p}")

    local = MODELS_DIR / "weather"
    preferred = local / PREFERRED_EPW
    if preferred.exists():
        return preferred
    if local.is_dir():
        found = sorted(local.glob("*.epw"))
        if found:
            return found[0]

    install = find_energyplus()
    if install is not None and install.weather_dir.is_dir():
        found = sorted(install.weather_dir.glob("*.epw"))
        if found:
            return found[0]
    return None


@dataclass
class WeatherRecord:
    month: int
    day: int
    hour: int          # 1..24 as stored in the EPW
    drybulb_c: float
    rh_pct: float
    ghi_w_m2: float
    wind_m_s: float


class EPW:
    """Hourly weather with linear interpolation to arbitrary sub-hourly steps."""

    def __init__(self, records: list[WeatherRecord], location: str = "") -> None:
        self.records = records
        self.location = location
        self._index: dict[tuple[int, int, int], WeatherRecord] = {
            (r.month, r.day, r.hour): r for r in records
        }

    @classmethod
    def load(cls, path: str | Path) -> "EPW":
        path = Path(path)
        records: list[WeatherRecord] = []
        location = ""
        with path.open(encoding="latin-1") as fh:
            for lineno, raw in enumerate(fh):
                line = raw.strip()
                if not line:
                    continue
                if lineno == 0 and line.upper().startswith("LOCATION"):
                    parts = line.split(",")
                    location = ",".join(parts[1:4]).strip() if len(parts) > 3 else ""
                    continue
                if lineno < 8:
                    continue  # header block: DESIGN CONDITIONS ... DATA PERIODS
                parts = line.split(",")
                if len(parts) <= _COL_WIND:
                    continue
                try:
                    records.append(
                        WeatherRecord(
                            month=int(parts[_COL_MONTH]),
                            day=int(parts[_COL_DAY]),
                            hour=int(parts[_COL_HOUR]),
                            drybulb_c=float(parts[_COL_DRYBULB]),
                            rh_pct=float(parts[_COL_RELHUM]),
                            ghi_w_m2=max(0.0, float(parts[_COL_GHI])),
                            wind_m_s=max(0.0, float(parts[_COL_WIND])),
                        )
                    )
                except ValueError:
                    continue
        if not records:
            raise ValueError(f"no usable data records in {path}")
        return cls(records, location=location)

    def at(self, month: int, day: int, hour: int, minute: int = 0) -> WeatherRecord:
        """EPW hour *h* holds the average over the hour ending at *h*:00, so a
        reading at 14:30 sits between records 15 and 16 at 50%."""
        base_hour = hour + 1
        rec = self._index.get((month, day, base_hour))
        if rec is None:
            rec = self._nearest(month, day, base_hour)
        nxt = self._index.get((month, day, base_hour + 1)) or rec
        f = minute / 60.0
        return WeatherRecord(
            month=month,
            day=day,
            hour=base_hour,
            drybulb_c=rec.drybulb_c + (nxt.drybulb_c - rec.drybulb_c) * f,
            rh_pct=rec.rh_pct + (nxt.rh_pct - rec.rh_pct) * f,
            ghi_w_m2=rec.ghi_w_m2 + (nxt.ghi_w_m2 - rec.ghi_w_m2) * f,
            wind_m_s=rec.wind_m_s + (nxt.wind_m_s - rec.wind_m_s) * f,
        )

    def _nearest(self, month: int, day: int, hour: int) -> WeatherRecord:
        same_day = [r for r in self.records if r.month == month and r.day == day]
        pool = same_day or self.records
        return min(pool, key=lambda r: abs(r.hour - hour))

    def daily_mean(self, month: int, day: int) -> float:
        vals = [r.drybulb_c for r in self.records if r.month == month and r.day == day]
        return sum(vals) / len(vals) if vals else 25.0


def synthetic_record(month: int, day: int, hour: int, minute: int = 0) -> WeatherRecord:
    """Fallback weather when no EPW exists at all: a hot-humid diurnal cycle
    (Chennai May). Keeps the surrogate engine runnable in a bare CI container."""
    t = hour + minute / 60.0
    drybulb = 30.5 + 5.0 * math.sin(math.pi * (t - 9.0) / 12.0)
    rh = 70.0 - 18.0 * math.sin(math.pi * (t - 9.0) / 12.0)
    solar = max(0.0, 900.0 * math.sin(math.pi * (t - 6.2) / 12.2))
    return WeatherRecord(month, day, hour + 1, drybulb, max(30.0, rh), solar, 2.5)
