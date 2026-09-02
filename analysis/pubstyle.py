"""Shared publication figure style for the scMitoATAC manuscript.
Figures are authored at final print size (inches), point-based fonts, bold lowercase panel letters top-left,
300 DPI, tight bbox. Import and call apply() at the top of every figure generator."""
import os
import matplotlib as mpl

# Genome Biology submission mode: when GB_SUB is set, save() (and save_submission) also write a vector
# PDF + a 300-dpi PNG of each main figure into GB_DIR, renamed fig1..fig6. Builders additionally drop
# the in-graphic overall suptitle in this mode (the title moves to the figure legend, per GB figure rules).
GB_SUB = bool(os.environ.get("GB_SUB"))
GB_DIR = os.environ.get("GB_DIR", "")
_GB_NAMEMAP = {
    "main_r2_method_envelope.png": "fig1",
    "main_r2_specificity.png": "fig2",
    "main_v3_fig6_lowvaf_pooling.png": "fig3",
    "main_r2_coverage.png": "fig4",
    "main_r2_m9438.png": "fig5",
    "main_r2_luad.png": "fig6",
}

COL = 6.9      # full text-column width (inches) for a 2 cm-margin letter page
HALF = 3.35    # half-column width
# validated CVD-safe palette used across the paper
BLUE, RED, GREEN, GREY, INK, FILL = "#2166ac", "#b2182b", "#2e7d32", "#8c8c8c", "#1a2b3c", "#eef4fa"


def apply():
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.titlesize": 9.5, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.constrained_layout.use": False,
    })


def panel(ax, letter, x=-0.16, y=1.04, fontsize=10):
    """Bold lowercase panel letter, top-left, outside the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
            va="bottom", ha="right", color=INK)


def save_submission(fig, path):
    """In GB submission mode, also emit a vector PDF + 300-dpi PNG (renamed fig1..fig6) into GB_DIR."""
    if not (GB_SUB and GB_DIR):
        return
    name = _GB_NAMEMAP.get(os.path.basename(path))
    if not name:
        return
    fig.savefig(f"{GB_DIR}/{name}.pdf", bbox_inches="tight")           # vector, for journal upload
    fig.savefig(f"{GB_DIR}/{name}.png", dpi=300, bbox_inches="tight")  # raster preview for the assembled file


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    save_submission(fig, path)
