/**
 * Viridis: a perceptually-uniform, colorblind-safe sequential palette
 * (Matplotlib/BIDS, public domain). Perceptual uniformity means equal steps
 * in data map to equal-looking steps in color — important for a data layer
 * a user will read quantitatively, not just admire.
 *
 * Implemented as a closed-form polynomial fit (Jamie Wong / Google's
 * commonly-published GLSL approximation of the reference LUT) rather than a
 * 256-entry lookup table, so the palette carries zero data and zero extra
 * dependencies — appropriate for a client whose whole job is being thin.
 */

export interface Rgb {
  r: number;
  g: number;
  b: number;
}

// prettier-ignore
const C0: Rgb = { r: 0.2777273272234177, g: 0.005407344544966578, b: 0.3340998053353061 };
// prettier-ignore
const C1: Rgb = { r: 0.1050930431085774, g: 1.404613529898575, b: 1.384590162594685 };
// prettier-ignore
const C2: Rgb = { r: -0.3308618287255563, g: 0.214847559468213, b: 0.09509516302823659 };
// prettier-ignore
const C3: Rgb = { r: -4.634230498983486, g: -5.799100973351585, b: -19.33244095627987 };
// prettier-ignore
const C4: Rgb = { r: 6.228269936347081, g: 14.17993336680509, b: 56.69055260068105 };
// prettier-ignore
const C5: Rgb = { r: 4.776384997670288, g: -13.74514537774601, b: -65.35303263337234 };
// prettier-ignore
const C6: Rgb = { r: -5.435455855934631, g: 4.645852612178535, b: 26.3124352495832 };

function add(a: Rgb, b: Rgb): Rgb {
  return { r: a.r + b.r, g: a.g + b.g, b: a.b + b.b };
}

function scale(a: Rgb, t: number): Rgb {
  return { r: a.r * t, g: a.g * t, b: a.b * t };
}

/** Sample Viridis at `t` in [0, 1]; returns linear-ish RGB in [0, 1]. */
export function viridis(t: number): Rgb {
  const clamped = Math.min(1, Math.max(0, t));
  // Horner's method over the six coefficient vectors, evaluated coarsest
  // (highest-degree) term first: poly = C0 + t*(C1 + t*(C2 + ... + t*C6)).
  const coefficients = [C6, C5, C4, C3, C2, C1, C0];
  let poly: Rgb = { r: 0, g: 0, b: 0 };
  for (const coefficient of coefficients) {
    poly = add(coefficient, scale(poly, clamped));
  }
  return { r: clamp01(poly.r), g: clamp01(poly.g), b: clamp01(poly.b) };
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

export function rgbToHex({ r, g, b }: Rgb): string {
  const toByte = (c: number) => Math.round(c * 255);
  return `#${[toByte(r), toByte(g), toByte(b)].map((b8) => b8.toString(16).padStart(2, "0")).join("")}`;
}

/** One labeled tick on a legend gradient. */
export interface LegendStop {
  t: number;
  value: number;
  color: string;
}

/** Structured legend data for a data layer: domain, unit, and color stops. */
export interface Legend {
  title: string;
  unit: string;
  min: number;
  max: number;
  /** Evenly-spaced stops across the domain, colored with the same palette
   *  function used to shade the layer, so the legend never drifts from the
   *  rendering it explains. */
  stops: LegendStop[];
}

/** Build a legend by sampling `colorAt` at `stepCount` evenly-spaced points. */
export function buildLegend(
  title: string,
  unit: string,
  min: number,
  max: number,
  colorAt: (t: number) => Rgb,
  stepCount = 6,
): Legend {
  const stops: LegendStop[] = [];
  for (let i = 0; i < stepCount; i += 1) {
    const t = i / (stepCount - 1);
    stops.push({ t, value: min + t * (max - min), color: rgbToHex(colorAt(t)) });
  }
  return { title, unit, min, max, stops };
}
