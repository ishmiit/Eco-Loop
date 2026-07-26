"""A dependency-free IDF reader/writer, plus the ECM library.

Why not eppy: eppy needs the 40 MB ``Energy+.idd`` parsed at import and pins
itself to IDD versions. The closed loop only needs to (a) rewrite the run
period / timestep before a run and (b) let the agent apply well-defined ECMs
and write out a new ``.idf``. That is a few hundred lines of exact,
version-independent text handling — and round-tripping is lossless, so the
generated variants stay readable next to the baseline in the submission.

The parser keeps every comment and the original field spelling, so a diff
between ``baseline.idf`` and an agent-generated variant shows only what the
agent actually changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class IDFObject:
    """One IDF object: a class name plus its fields, with per-field comments."""

    obj_class: str
    fields: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)   # trailing !- comment per field
    preamble: list[str] = field(default_factory=list)   # standalone comment lines above

    @property
    def name(self) -> str:
        return self.fields[0] if self.fields else ""

    def field(self, index: int, default: str = "") -> str:
        return self.fields[index] if 0 <= index < len(self.fields) else default

    def set_field(self, index: int, value: Any) -> None:
        while len(self.fields) <= index:
            self.fields.append("")
            self.comments.append("")
        self.fields[index] = str(value)

    def field_index_by_comment(self, needle: str) -> int:
        needle = needle.lower()
        for i, c in enumerate(self.comments):
            if needle in c.lower():
                return i
        return -1

    def to_text(self) -> str:
        lines = list(self.preamble)
        if not self.fields:
            return "\n".join(lines + [f"{self.obj_class};", ""])
        width = 24
        head = f"  {self.obj_class},"
        lines.append(head)
        for i, value in enumerate(self.fields):
            terminator = ";" if i == len(self.fields) - 1 else ","
            body = f"    {value}{terminator}"
            comment = self.comments[i] if i < len(self.comments) else ""
            if comment:
                pad = max(1, width - len(body) + 4)
                lines.append(f"{body}{' ' * pad}!- {comment}")
            else:
                lines.append(body)
        lines.append("")
        return "\n".join(lines)


class IDF:
    """An ordered collection of :class:`IDFObject`."""

    def __init__(self, objects: list[IDFObject], header: str = "") -> None:
        self.objects = objects
        self.header = header

    # -- io -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "IDF":
        return cls.parse(Path(path).read_text(encoding="latin-1"))

    @classmethod
    def parse(cls, text: str) -> "IDF":
        objects: list[IDFObject] = []
        header_lines: list[str] = []
        pending_comments: list[str] = []
        tokens: list[tuple[str, str]] = []   # (value, comment)
        seen_first_object = False

        for raw in text.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                if not tokens:
                    (pending_comments if seen_first_object else header_lines).append(line)
                continue
            if stripped.startswith("!"):
                if not tokens:
                    (pending_comments if seen_first_object else header_lines).append(line)
                continue

            body, _, comment = stripped.partition("!")
            comment = comment.lstrip("- ").strip()
            # A physical line can hold several fields: "Timestep, 4;" or
            # "ScheduleTypeLimits, Fraction, 0.0, 1.0, CONTINUOUS;". Note the
            # `.strip()` on the loop guard — trailing padding before the `!-`
            # comment must not be mistaken for another (empty) field.
            while body.strip():
                match = re.search(r"[,;]", body)
                if match is None:
                    tokens.append((body.strip(), comment))
                    body = ""
                    break
                piece, sep, body = body[: match.start()], body[match.start()], body[match.start() + 1:]
                is_last_on_line = not body.strip()
                tokens.append((piece.strip(), comment if is_last_on_line else ""))
                if sep == ";":
                    seen_first_object = True
                    objects.append(_build_object(tokens, pending_comments))
                    tokens = []
                    pending_comments = []
                    body = ""
                    break
        if tokens:  # unterminated final object — keep it rather than lose data
            objects.append(_build_object(tokens, pending_comments))
        return cls(objects, header="\n".join(header_lines).rstrip() + "\n" if header_lines else "")

    def to_text(self) -> str:
        parts = [self.header] if self.header else []
        parts.extend(obj.to_text() for obj in self.objects)
        return "\n".join(parts)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_text(), encoding="utf-8")
        return path

    # -- query --------------------------------------------------------------
    def of_class(self, obj_class: str) -> list[IDFObject]:
        want = obj_class.lower()
        return [o for o in self.objects if o.obj_class.lower() == want]

    def get(self, obj_class: str, name: str) -> IDFObject | None:
        want = name.strip().lower()
        for o in self.of_class(obj_class):
            if o.name.strip().lower() == want:
                return o
        return None

    def require(self, obj_class: str, name: str) -> IDFObject:
        obj = self.get(obj_class, name)
        if obj is None:
            raise KeyError(f"{obj_class} named {name!r} not found")
        return obj

    def classes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.objects:
            counts[o.obj_class] = counts.get(o.obj_class, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self, limit: int = 40) -> list[dict[str, Any]]:
        out = []
        for o in self.objects[:limit] if limit else self.objects:
            out.append({"class": o.obj_class, "name": o.name, "n_fields": len(o.fields)})
        return out

    def add(self, obj: IDFObject) -> IDFObject:
        self.objects.append(obj)
        return obj

    def remove(self, obj: IDFObject) -> None:
        self.objects.remove(obj)

    def copy(self) -> "IDF":
        return IDF.parse(self.to_text())

    # -- run-level rewrites -------------------------------------------------
    def set_run_period(
        self,
        start_month: int,
        start_day: int,
        end_month: int,
        end_day: int,
    ) -> None:
        for rp in self.of_class("RunPeriod"):
            rp.set_field(1, start_month)
            rp.set_field(2, start_day)
            rp.set_field(4, end_month)
            rp.set_field(5, end_day)

    def set_timestep(self, per_hour: int) -> None:
        objs = self.of_class("Timestep")
        if objs:
            objs[0].set_field(0, int(per_hour))
        else:
            self.add(IDFObject("Timestep", [str(int(per_hour))], [""]))

    def set_schedule_constant(self, name: str, value: float) -> bool:
        obj = self.get("Schedule:Constant", name)
        if obj is None:
            return False
        obj.set_field(2, f"{value:g}")
        return True

    def schedule_constants(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for obj in self.of_class("Schedule:Constant"):
            try:
                out[obj.name] = float(obj.field(2, "0"))
            except ValueError:
                continue
        return out


def _build_object(tokens: list[tuple[str, str]], preamble: list[str]) -> IDFObject:
    obj_class = tokens[0][0]
    fields = [t[0] for t in tokens[1:]]
    comments = [t[1] for t in tokens[1:]]
    keep = [ln for ln in preamble if ln.strip()]
    return IDFObject(obj_class=obj_class, fields=fields, comments=comments, preamble=keep)


# ==========================================================================
# ECM library — the vocabulary of retrofit measures the agent can apply.
# ==========================================================================


@dataclass
class ECMResult:
    ok: bool
    ecm: str
    detail: str
    changes: list[str] = field(default_factory=list)


ECMFunc = Callable[[IDF, dict[str, Any]], ECMResult]


def _num(params: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    try:
        val = float(params.get(key, default))
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))


def ecm_roof_insulation(idf: IDF, params: dict[str, Any]) -> ECMResult:
    """Add 50 mm insulation under the RCC roof deck (over-deck retrofit)."""
    con = idf.get("Construction", "ROOF_UNINSULATED")
    if con is None:
        return ECMResult(False, "roof_insulation", "Construction ROOF_UNINSULATED not found")
    if "ROOF_INSUL_50MM" in [f.upper() for f in con.fields]:
        return ECMResult(False, "roof_insulation", "insulation layer already present")
    if idf.get("Material", "ROOF_INSUL_50MM") is None:
        return ECMResult(False, "roof_insulation", "Material ROOF_INSUL_50MM not defined")
    # Insert after the outermost layer (screed) so it sits above the slab.
    con.fields.insert(2, "ROOF_INSUL_50MM")
    con.comments.insert(2, "Layer 2 (ECM: over-deck insulation)")
    return ECMResult(True, "roof_insulation", "50 mm XPS added to ROOF_UNINSULATED",
                     ["Construction ROOF_UNINSULATED += ROOF_INSUL_50MM"])


def ecm_cool_roof(idf: IDF, params: dict[str, Any]) -> ECMResult:
    """High-albedo roof coating: drop solar absorptance of the outer layer."""
    absorptance = _num(params, "solar_absorptance", 0.30, 0.15, 0.75)
    mat = idf.get("Material", "ROOF_SCREED_40MM")
    if mat is None:
        return ECMResult(False, "cool_roof", "Material ROOF_SCREED_40MM not found")
    before = mat.field(7, "?")
    mat.set_field(7, f"{absorptance:g}")   # Solar Absorptance
    mat.set_field(8, f"{absorptance:g}")   # Visible Absorptance
    return ECMResult(True, "cool_roof", f"solar absorptance {before} -> {absorptance:g}",
                     [f"Material ROOF_SCREED_40MM solar absorptance = {absorptance:g}"])


def ecm_glazing_upgrade(idf: IDF, params: dict[str, Any]) -> ECMResult:
    """Swap single clear glazing for the pre-defined low-e DGU."""
    con = idf.get("Construction", "WINDOW_CLEAR")
    if con is None:
        return ECMResult(False, "glazing_upgrade", "Construction WINDOW_CLEAR not found")
    if idf.get("WindowMaterial:SimpleGlazingSystem", "GLAZING_DGU_LOWE") is None:
        return ECMResult(False, "glazing_upgrade", "GLAZING_DGU_LOWE not defined")
    if con.field(1).upper() == "GLAZING_DGU_LOWE":
        return ECMResult(False, "glazing_upgrade", "already upgraded")
    con.set_field(1, "GLAZING_DGU_LOWE")
    return ECMResult(True, "glazing_upgrade", "WINDOW_CLEAR now uses GLAZING_DGU_LOWE",
                     ["Construction WINDOW_CLEAR outside layer = GLAZING_DGU_LOWE"])


def ecm_led_retrofit(idf: IDF, params: dict[str, Any]) -> ECMResult:
    """Reduce lighting power density by a fraction (LED retrofit)."""
    reduction = _num(params, "reduction", 0.35, 0.05, 0.6)
    changes = []
    for lights in idf.of_class("Lights"):
        # Watts/Area is field index 5 for the Watts/Area calculation method.
        if lights.field(3).strip().lower() != "watts/area":
            continue
        try:
            lpd = float(lights.field(5))
        except ValueError:
            continue
        new = round(lpd * (1.0 - reduction), 2)
        lights.set_field(5, f"{new:g}")
        changes.append(f"{lights.name}: {lpd:g} -> {new:g} W/m2")
    if not changes:
        return ECMResult(False, "led_retrofit", "no Watts/Area Lights objects found")
    return ECMResult(True, "led_retrofit", f"LPD reduced {reduction * 100:.0f}%", changes)


def ecm_window_shading(idf: IDF, params: dict[str, Any]) -> ECMResult:
    """Add external overhangs to the south-facing windows."""
    depth = _num(params, "depth_m", 0.9, 0.3, 2.0)
    added = []
    for win in idf.of_class("Window"):
        if not win.name.upper().endswith("_S"):
            continue
        shade_name = f"{win.name}_OVERHANG"
        if idf.get("Shading:Overhang", shade_name) is not None:
            continue
        idf.add(
            IDFObject(
                "Shading:Overhang",
                [shade_name, win.name, "0.2", "90", "0.3", "0.3", f"{depth:g}"],
                [
                    "Name",
                    "Window or Door Name",
                    "Height above Window or Door {m}",
                    "Tilt Angle from Window/Door {deg}",
                    "Left extension from Window/Door Width {m}",
                    "Right extension from Window/Door Width {m}",
                    "Depth {m} (ECM: external shading)",
                ],
                preamble=[f"!- ECM window_shading: overhang on {win.name}"],
            )
        )
        added.append(shade_name)
    if not added:
        return ECMResult(False, "window_shading", "no south windows without overhangs")
    return ECMResult(True, "window_shading", f"{len(added)} overhang(s) at {depth:g} m depth", added)


def ecm_infiltration_sealing(idf: IDF, params: dict[str, Any]) -> ECMResult:
    """Air-sealing / door curtains: cut infiltration air changes."""
    reduction = _num(params, "reduction", 0.3, 0.05, 0.6)
    changes = []
    for infil in idf.of_class("ZoneInfiltration:DesignFlowRate"):
        if infil.field(3).strip().lower() != "airchanges/hour":
            continue
        try:
            ach = float(infil.field(7))
        except ValueError:
            continue
        new = round(ach * (1.0 - reduction), 3)
        infil.set_field(7, f"{new:g}")
        changes.append(f"{infil.name}: {ach:g} -> {new:g} ACH")
    if not changes:
        return ECMResult(False, "infiltration_sealing", "no AirChanges/Hour infiltration objects")
    return ECMResult(True, "infiltration_sealing", f"infiltration cut {reduction * 100:.0f}%", changes)


def ecm_heat_recovery(idf: IDF, params: dict[str, Any]) -> ECMResult:
    """Enable sensible+latent heat recovery on the ventilation air."""
    sensible = _num(params, "sensible_effectiveness", 0.7, 0.3, 0.85)
    latent = _num(params, "latent_effectiveness", 0.55, 0.0, 0.8)
    changes = []
    for il in idf.of_class("ZoneHVAC:IdealLoadsAirSystem"):
        il.set_field(24, "Enthalpy")
        il.set_field(25, f"{sensible:g}")
        il.set_field(26, f"{latent:g}")
        changes.append(f"{il.name}: enthalpy HR {sensible:g}/{latent:g}")
    if not changes:
        return ECMResult(False, "heat_recovery", "no ideal loads systems found")
    return ECMResult(True, "heat_recovery", "enthalpy heat recovery enabled", changes)


def ecm_demand_controlled_ventilation(idf: IDF, params: dict[str, Any]) -> ECMResult:
    """Let EnergyPlus scale ventilation with occupancy (DCV)."""
    changes = []
    for il in idf.of_class("ZoneHVAC:IdealLoadsAirSystem"):
        il.set_field(22, "OccupancySchedule")
        changes.append(f"{il.name}: DCV = OccupancySchedule")
    if not changes:
        return ECMResult(False, "demand_controlled_ventilation", "no ideal loads systems found")
    return ECMResult(True, "demand_controlled_ventilation", "occupancy-based DCV enabled", changes)


ECM_LIBRARY: dict[str, ECMFunc] = {
    "roof_insulation": ecm_roof_insulation,
    "cool_roof": ecm_cool_roof,
    "glazing_upgrade": ecm_glazing_upgrade,
    "led_retrofit": ecm_led_retrofit,
    "window_shading": ecm_window_shading,
    "infiltration_sealing": ecm_infiltration_sealing,
    "heat_recovery": ecm_heat_recovery,
    "demand_controlled_ventilation": ecm_demand_controlled_ventilation,
}

ECM_DOCS: dict[str, str] = {
    "roof_insulation": "Add 50 mm XPS over the RCC roof deck. Cuts conduction gain through the largest exposed surface.",
    "cool_roof": "High-albedo roof coating. params: solar_absorptance (0.15-0.75, default 0.30).",
    "glazing_upgrade": "Replace single clear glazing with low-e DGU (U 5.8 -> 1.8, SHGC 0.82 -> 0.28).",
    "led_retrofit": "Reduce lighting power density. params: reduction (0.05-0.6, default 0.35).",
    "window_shading": "Add external overhangs to south windows. params: depth_m (0.3-2.0, default 0.9).",
    "infiltration_sealing": "Reduce infiltration ACH. params: reduction (0.05-0.6, default 0.3).",
    "heat_recovery": "Enthalpy heat recovery on ventilation air. params: sensible_effectiveness, latent_effectiveness.",
    "demand_controlled_ventilation": "Occupancy-based ventilation scaling inside EnergyPlus.",
}


def apply_ecms(idf: IDF, measures: Iterable[dict[str, Any]]) -> list[ECMResult]:
    """Apply a list of ``{"ecm": name, "params": {...}}`` to a copy-in-place IDF."""
    results: list[ECMResult] = []
    for measure in measures:
        name = str(measure.get("ecm", "")).strip().lower()
        params = measure.get("params") or {}
        func = ECM_LIBRARY.get(name)
        if func is None:
            results.append(
                ECMResult(False, name or "<empty>", f"unknown ECM; valid: {', '.join(sorted(ECM_LIBRARY))}")
            )
            continue
        try:
            results.append(func(idf, params))
        except Exception as exc:  # a bad ECM must never abort the pass
            results.append(ECMResult(False, name, f"{type(exc).__name__}: {exc}"))
    return results
