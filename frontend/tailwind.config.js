/**
 * Design tokens.
 *
 * The palette encodes model direction rather than decorating: `lift` is a
 * feature raising win probability, `drag` one lowering it. They are a
 * colourblind-safe diverging pair (teal / rust) rather than green / red, which
 * is both the cliche and the worst choice for deuteranopia.
 *
 * `paper` / `ink` are cool slate rather than warm cream — this is a measurement
 * instrument, not an editorial page.
 */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: {
          DEFAULT: "#eef1f5",
          raised: "#f7f9fb",
          sunken: "#e4e9ef",
        },
        ink: {
          DEFAULT: "#16202b",
          muted: "#5a6a7a",
          // 4.54:1 on paper, 4.87:1 on raised. #8b9aa8 looked right but came
          // in at 2.54:1, which fails AA for the footer and every caption.
          faint: "#636f7a",
        },
        rule: {
          DEFAULT: "#d3dae2",
          strong: "#b3bec9",
        },
        // Diverging semantic axis, used for SHAP contributions and score moves.
        lift: {
          DEFAULT: "#0f766e",
          soft: "#99f6e4",
          wash: "#e6fffb",
        },
        drag: {
          DEFAULT: "#b4501f",
          soft: "#fed7aa",
          wash: "#fff4ec",
        },
        // Priority is ordinal, so it gets a single-hue ramp, not new hues.
        alarm: "#9f1239",
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      fontSize: {
        micro: ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.01em" }],
      },
      borderRadius: {
        // Deliberately shallow: hairline structure, not a card kit.
        DEFAULT: "2px",
        sm: "1px",
      },
      transitionTimingFunction: {
        meter: "cubic-bezier(0.22, 0.61, 0.36, 1)",
      },
    },
  },
  plugins: [],
};
