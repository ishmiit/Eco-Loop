"""The building's operating schedule, in one place.

These profiles mirror the ``Schedule:Compact`` objects in
``models/baseline.idf`` exactly. They exist as Python data for two reasons:

1. the surrogate engine needs them to drive internal gains;
2. the agent needs **occupancy foresight** — "the production shift starts in
   40 minutes" — to do optimum start.

Point 2 is what closes the last comfort gap in the control strategy. A
purely reactive setback controller leaves the building at its 30 C setback
until the first person walks in, and then spends the pull-down period with
occupants in an uncomfortable space; that is how a naive energy optimiser
"saves" energy by quietly spending comfort. Knowing the shift pattern, the
agent can pre-cool just in time and get the saving without the transient.
"""

from __future__ import annotations

# (until_hour, value) — the value applies up to but not including until_hour.
OCCUPANCY_WEEKDAY: dict[str, list[tuple[float, float]]] = {
    "OFFICE": [(9, 0.0), (13, 0.95), (14, 0.5), (18, 0.9), (19, 0.25), (24, 0.0)],
    "PROD_HALL": [(6, 0.0), (10, 1.0), (13, 0.9), (14, 0.4), (18, 0.95), (19, 0.3), (24, 0.0)],
    "PACK_STORE": [(7, 0.0), (12, 0.6), (14, 0.3), (18, 0.7), (24, 0.0)],
}

OCCUPANCY_SATURDAY: dict[str, list[tuple[float, float]]] = {
    "OFFICE": [(9, 0.0), (14, 0.4), (24, 0.0)],
    "PROD_HALL": [(6, 0.0), (14, 0.7), (24, 0.0)],
    "PACK_STORE": [(8, 0.0), (14, 0.4), (24, 0.0)],
}

LIGHTING: list[tuple[float, float]] = [(6, 0.05), (19, 0.9), (21, 0.2), (24, 0.05)]
EQUIPMENT: list[tuple[float, float]] = [(6, 0.15), (18, 0.85), (20, 0.35), (24, 0.15)]

PEOPLE_COUNT: dict[str, int] = {"OFFICE": 6, "PROD_HALL": 12, "PACK_STORE": 4}
METABOLIC_W: dict[str, float] = {"OFFICE": 120.0, "PROD_HALL": 200.0, "PACK_STORE": 165.0}

OCCUPIED_EPS = 0.05
FORECAST_HORIZON_MIN = 240


def fraction(profile: list[tuple[float, float]], clock: float) -> float:
    for until, value in profile:
        if clock < until:
            return value
    return profile[-1][1]


def occupancy_fraction(zone: str, clock: float, weekday: int) -> float:
    """``weekday`` is ISO: 1 = Monday ... 7 = Sunday."""
    if weekday >= 7:
        return 0.0
    table = OCCUPANCY_SATURDAY if weekday == 6 else OCCUPANCY_WEEKDAY
    return fraction(table.get(zone, OCCUPANCY_WEEKDAY["OFFICE"]), clock)


def _next_weekday(weekday: int) -> int:
    return weekday % 7 + 1


def minutes_until_occupied(
    zone: str, hour: int, minute: int, weekday: int, horizon_min: int = FORECAST_HORIZON_MIN
) -> float | None:
    """Minutes until this zone is next occupied, or None beyond the horizon.

    Returns 0.0 when it is occupied right now. Steps in 15-minute increments and
    rolls over midnight, so a Friday-evening query correctly skips the weekend.
    """
    clock = hour + minute / 60.0
    if occupancy_fraction(zone, clock, weekday) > OCCUPIED_EPS:
        return 0.0
    step = 15.0
    elapsed = 0.0
    probe_weekday = weekday
    probe_clock = clock
    while elapsed <= horizon_min:
        probe_clock += step / 60.0
        elapsed += step
        if probe_clock >= 24.0:
            probe_clock -= 24.0
            probe_weekday = _next_weekday(probe_weekday)
        if occupancy_fraction(zone, probe_clock, probe_weekday) > OCCUPIED_EPS:
            return elapsed
    return None


def minutes_until_vacant(
    zone: str, hour: int, minute: int, weekday: int, horizon_min: int = FORECAST_HORIZON_MIN
) -> float | None:
    clock = hour + minute / 60.0
    if occupancy_fraction(zone, clock, weekday) <= OCCUPIED_EPS:
        return 0.0
    step = 15.0
    elapsed = 0.0
    probe_weekday = weekday
    probe_clock = clock
    while elapsed <= horizon_min:
        probe_clock += step / 60.0
        elapsed += step
        if probe_clock >= 24.0:
            probe_clock -= 24.0
            probe_weekday = _next_weekday(probe_weekday)
        if occupancy_fraction(zone, probe_clock, probe_weekday) <= OCCUPIED_EPS:
            return elapsed
    return None


def occupancy_forecast(hour: int, minute: int, weekday: int) -> dict[str, dict[str, float | None]]:
    return {
        zone: {
            "minutes_until_occupied": minutes_until_occupied(zone, hour, minute, weekday),
            "minutes_until_vacant": minutes_until_vacant(zone, hour, minute, weekday),
        }
        for zone in OCCUPANCY_WEEKDAY
    }
