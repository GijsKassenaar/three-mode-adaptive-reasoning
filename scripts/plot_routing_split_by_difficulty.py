#!/usr/bin/env python3
"""Stacked-bar plots of the NOTHINK/SHORT/LONG routing split per MATH difficulty level,
from the `val-difficulties/` metrics logged during validation.

Reads the LAST validation block of each given training/eval log and plots one panel
per log: x = difficulty level 1..5, stacked bar = mode fractions (sum to 1),
per-level accuracy annotated above each bar.

Usage:
    python scripts/plot_routing_split_by_difficulty.py LABEL=LOGFILE [LABEL=LOGFILE ...] \
        [--out figures/routing_split_by_difficulty.png]
e.g.
    python scripts/plot_routing_split_by_difficulty.py \
        "three-mode routing (step 90)=three_mode_routing_math_phase2.log"
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("logs", nargs="+", help="LABEL=LOGFILE pairs (or bare log paths)")
parser.add_argument("--out", default="figures/routing_split_by_difficulty.png")
args = parser.parse_args()

LOGS = {}
for spec in args.logs:
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        label, path = os.path.basename(spec), spec
    LOGS[label] = path
LEVELS = ["1", "2", "3", "4", "5"]
MODES = ["nothink", "short", "long"]
COLORS = {"nothink": "#66bb6a", "short": "#ffca28", "long": "#5c6bc0"}  # green / amber / indigo


def parse_last_val(path):
    """Return {level: {acc, count, nothink, short, long}} from the LAST val block."""
    text = open(path).read()
    out = {}
    for lv in LEVELS:
        d = {}
        for key in ("acc", "count", "nothink_frac", "short_frac", "long_frac"):
            # last occurrence = most recent validation
            matches = re.findall(rf"val-difficulties/level{lv}/{key}:([0-9.]+)", text)
            if matches:
                d[key.replace("_frac", "")] = float(matches[-1])
        if d:
            out[lv] = d
    return out


fig, axes = plt.subplots(1, len(LOGS), figsize=(12, 5.2), sharey=True)
if len(LOGS) == 1:
    axes = [axes]

for ax, (title, path) in zip(axes, LOGS.items()):
    data = parse_last_val(path)
    x = np.arange(len(LEVELS))
    bottoms = np.zeros(len(LEVELS))
    for mode in MODES:
        vals = np.array([data.get(lv, {}).get(mode, 0.0) for lv in LEVELS])
        ax.bar(x, vals, bottom=bottoms, color=COLORS[mode], label=mode.upper(),
               edgecolor="white", linewidth=0.6)
        # label segment % when big enough to read
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.07:
                ax.text(xi, b + v / 2, f"{v*100:.0f}", ha="center", va="center",
                        fontsize=8, color="black")
        bottoms += vals
    # accuracy annotated above each bar
    for xi, lv in enumerate(LEVELS):
        acc = data.get(lv, {}).get("acc")
        if acc is not None:
            ax.text(xi, 1.02, f"acc {acc*100:.0f}%", ha="center", va="bottom",
                    fontsize=8, color="#37474f")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{lv}" for lv in LEVELS])
    ax.set_xlabel("MATH difficulty level")
    ax.set_ylim(0, 1.12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)

axes[0].set_ylabel("Routing-mode fraction (free val)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("NOTHINK / SHORT / LONG routing split by difficulty  (MATH-500)", y=1.06, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out = args.out
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# also print the parsed tables for the record
for title, path in LOGS.items():
    print(f"\n{title}")
    data = parse_last_val(path)
    for lv in LEVELS:
        d = data.get(lv, {})
        print(f"  L{lv}: acc={d.get('acc',0)*100:.0f}%  NT={d.get('nothink',0)*100:.0f}  "
              f"SH={d.get('short',0)*100:.0f}  LG={d.get('long',0)*100:.0f}")
