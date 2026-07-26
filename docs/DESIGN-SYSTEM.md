# Design System

The dashboard follows the **FoodRaksha CRM** design system: Apple iOS visual
language, titanium palette, SF Pro type scale, 980px capsule buttons, inset
grouped lists, 44pt minimum tap targets.

`src/ecoloop/server/static/styles.css` is the implementation; every token below
appears there as a CSS custom property.

## 1. Colour

The interface is titanium and graphite. **Colour appears in exactly one place —
status** — because a pipeline is unusable if every state looks the same.

### Titanium palette

| Token | Hex | Use |
|---|---|---|
| Natural Titanium | `#C3BCB1` | secondary buttons, step icons, comfort bands |
| Titanium Mid | `#A8A197` | hover, set-point traces |
| Titanium Deep | `#6E6960` | focus rings, baseline chart series |
| Graphite | `#1D1D1F` | primary actions, the one dark surface, AI series |

### Surfaces & text

| Token | Hex |
|---|---|
| Background | `#EFEDE8` |
| Surface | `#FFFFFF` |
| Surface Sunk | `#E8E5DF` |
| White Titanium | `#F0EEE9` |
| White Titanium Light | `#F7F6F3` |
| Label | `#1D1D1F` |
| Label 2 | `rgba(60,60,67,.60)` |
| Label 3 | `rgba(60,60,67,.32)` |
| Separator | `rgba(60,60,67,.16)` |

### Status tones — the only colour in the product

| State | Foreground | Background | Used for |
|---|---|---|---|
| ok | `#34785A` | `#E4EFE9` | comfort preserved, verified ECM, EnergyPlus present |
| wait | `#9A7B3F` | `#F5EEDF` | surrogate engine, peak-tariff window, external override |
| stop | `#A2453C` | `#F6E5E3` | comfort degraded, failed ECM, LLM offline, limit lines |

### Dark mode

A deliberate extension, not an inversion. The graphite card is already the
darkest surface in the palette, so dark mode promotes graphite to the ground and
titanium to the text; the graphite card goes true black and gains a hairline so
it still reads as elevated. Status tones are re-picked for contrast on a dark
ground rather than reused. Implemented at token level, so
`prefers-color-scheme` and the in-page toggle (`data-theme`) both work in both
directions. Demos get run in dim rooms.

## 2. Type — SF Pro

The system font stack renders genuine SF Pro on Apple hardware and Inter
elsewhere. **Negative tracking is what makes it read as SF — never omit it.**

| Style | Size | Weight | Tracking |
|---|---|---|---|
| Large Title | 34 | Bold | −0.026em |
| Title 1 | 28 | Bold | −0.022em |
| Title 2 | 22 | Bold | −0.018em |
| Title 3 | 20 | Semibold | −0.014em |
| Headline | 17 | Semibold | −0.012em |
| Body | 17 | Regular | −0.011em |
| Callout | 16 | Regular | −0.010em |
| Subhead | 15 | Regular | −0.008em |
| Footnote | 13 | Regular | −0.004em |
| Caption | 12 | Medium | 0, uppercase |

Numbers use `font-variant-numeric: tabular-nums` wherever they line up in
columns, and SF Mono for data readouts.

## 3. Buttons — 980px capsule

`border-radius: 980px`, `min-height: 50px`, `active: scale(.965)`.
One primary button per screen; everything else secondary or quiet.

* primary — graphite fill
* secondary — natural titanium fill
* quiet — surface-sunk fill
* small 44px, extra-small 36px, disabled surface-sunk with Label 3 text

## 4. Inset grouped list

The default container for run lists, decision feeds, ECM attempts and artifacts.
0.5px separators, omitted on the last row, 44pt minimum row height, chevron for
navigable rows. Rows carry an optional step icon: `✓` green when done, numbered
graphite when active, titanium when idle, `!` in wait/stop tones for problems.

Because a whole row can be one `<button>` or `<a>`, the label stack is built from
spans — which must be `display: block`, or the primary and secondary lines run
together on one line.

## 5. Form controls

17px inputs (prevents iOS zoom-on-focus), label above, never
placeholder-as-label. Radius 12, padding 14/16, focus is a titanium-deep border
plus a 3.5px ring.

## 6. Cards, progress, the one dark surface

* Surface card — white, 18px radius, shadow 1. Never nested inside another card.
* Graphite card — used sparingly, one per screen at most, for the single thing
  that matters. Here: the headline saving percentage.
* Progress — 6px track, surface-sunk, graphite fill.

## 7. Slide-over

560px, right-anchored, scrim with 3px blur. Escape or scrim click closes it,
focus returns to the trigger. Used for decision detail — the model's rationale,
the raw request, what the safety layer changed, and the full model response — and
for ECM attempt detail.

## 8. Charts

Charts are hand-rolled SVG: no CDN, because the dashboard has to work on an
air-gapped laptop.

The design system reserves colour for status, so the two data series are
separated by **lightness, dash pattern and a direct end label**, never by hue:

* baseline — Titanium Deep `#6E6960`, 2px, dashed `6 3`
* AI closed loop — Graphite `#1D1D1F`, 2px, solid
* set-point trace — Titanium Mid `#A8A197`, 1.5px

This pair was validated with the data-viz palette validator: adjacent OKLab ΔE
18.9 for protan and tritan simulation, comfortably above the ΔE 8 target. It
fails the validator's *chroma floor* — by design, because the design system
mandates a near-neutral interface — so the relief the validator requires is
provided: a legend is always present, series are direct-labelled at the line
end, and the full per-timestep data is downloadable as CSV from every run.

Other rules in use: one y-axis per chart (never a dual axis — two measures of
different scale get two charts); comfort bands and the peak-tariff window as
faint fills *behind* the data, so a breach reads as geometry rather than needing
a colour alarm; per-zone PMV limits as dotted `stop`-tone reference lines,
stacked in the right gutter so several limits do not print on top of each other;
recessive grid; thin marks.

## 9. Accessibility

44pt tap targets, visible `:focus-visible` outlines in Titanium Deep, `role` and
`aria-label` on charts, `prefers-reduced-motion` honoured, status never conveyed
by colour alone (every pill carries a text label, every series a direct label).
