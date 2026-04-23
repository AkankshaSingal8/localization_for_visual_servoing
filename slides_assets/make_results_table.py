"""Render the 4-object x 3-pipeline results table as a slide-ready PNG.

Mirrors final.tex Table 1 (tab:exp1_main), excluding the tissue row (whose
FP reference run failed to register). Tints Pipeline B's depth-error cells
to steer the viewer's eye during a 20s narration.
"""
from statistics import median
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


OBJECTS = [
    ("Cheez-It",    {"A_px": 40.3, "B_lat": 2.38, "B_dep": 9.48,  "A_pe": 0.770, "B_pe": 0.867, "C_pe": 0.847}),
    ("Cardboard",   {"A_px": 33.0, "B_lat": 0.95, "B_dep": 18.12, "A_pe": 0.617, "B_pe": 0.870, "C_pe": 0.862}),
    ("Protein bar", {"A_px": 28.9, "B_lat": 3.04, "B_dep": 7.81,  "A_pe": 0.757, "B_pe": 0.883, "C_pe": 0.853}),
    ("Cake mix",    {"A_px": 39.1, "B_lat": 0.29, "B_dep": 12.24, "A_pe": 0.543, "B_pe": 0.849, "C_pe": 0.847}),
]

# Per-pipeline medians across the 4 objects.
def med(k): return median([d[k] for _, d in OBJECTS])
MEDIAN = {
    "A_px": med("A_px"),
    "B_lat": med("B_lat"),
    "B_dep": med("B_dep"),
    "A_pe": med("A_pe"),
    "B_pe": med("B_pe"),
    "C_pe": med("C_pe"),
}

# Runtime from final.tex Table 2.
RUNTIME_MS = {"A": 495, "B": 699, "C": 16}


def build_rows():
    rows = []
    for name, d in OBJECTS:
        rows.append([name, "A (IBVS)",          f"{d['A_px']:.1f} px",  "—",                f"{d['A_pe']:.3f}"])
        rows.append(["",   "B (EKF + DINOv2)",  f"{d['B_lat']:.2f} cm", f"{d['B_dep']:.2f}", f"{d['B_pe']:.3f}"])
        rows.append(["",   "C (EKF + FP, ref)", "—",                    "—",                f"{d['C_pe']:.3f}"])
    # Aggregate median block.
    rows.append(["Median", "A",       f"{MEDIAN['A_px']:.1f} px",  "—",                       f"{MEDIAN['A_pe']:.3f}"])
    rows.append(["",       "B",       f"{MEDIAN['B_lat']:.2f} cm", f"{MEDIAN['B_dep']:.2f}",  f"{MEDIAN['B_pe']:.3f}"])
    rows.append(["",       "C (ref)", "—",                         "—",                       f"{MEDIAN['C_pe']:.3f}"])
    return rows


def render(path):
    headers = ["Object", "Pipeline", "Lat. / Px err.", "Depth err. (cm)", "Path eff."]
    rows = build_rows()

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=200)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.16, 0.26, 0.20, 0.20, 0.14],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 1.6)

    # Header styling.
    for col_idx in range(len(headers)):
        c = tbl[(0, col_idx)]
        c.set_facecolor("#1f3b5a")
        c.set_text_props(color="white", weight="bold")

    # Body styling: tint B-depth column, bold median block, zebra by object.
    n_body = len(rows)
    median_start = n_body - 3
    for r in range(1, n_body + 1):
        body_idx = r - 1
        is_median = body_idx >= median_start
        for c_idx in range(len(headers)):
            cell = tbl[(r, c_idx)]
            if is_median:
                cell.set_facecolor("#fff4d6")
                cell.set_text_props(weight="bold")
            else:
                object_block = body_idx // 3
                cell.set_facecolor("#f7f9fc" if object_block % 2 == 0 else "#ffffff")
            # Tint the B-row depth-err cell (col 3 is "Depth err").
            pipeline_within_block = body_idx % 3
            if c_idx == 3 and pipeline_within_block == 1:
                cell.set_facecolor("#fde0c3")
                cell.set_text_props(weight="bold")

    ax.set_title(
        "Per-frame localization error vs FoundationPose reference\n"
        "(P1 start pose, 4 objects × 3 pipelines, medians across servo-active frames)",
        fontsize=15, weight="bold", pad=18,
    )

    # Footer: runtime line.
    footer = (
        f"Median iteration runtime — A: {RUNTIME_MS['A']} ms   "
        f"B: {RUNTIME_MS['B']} ms   C: {RUNTIME_MS['C']} ms    "
        "|    FP serves as reference; its error cells are by construction —"
    )
    fig.text(0.5, 0.04, footer, ha="center", fontsize=11, color="#333333")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    render("/Users/siddvoh/localization_for_visual_servoing/slides_assets/results_table.png")
