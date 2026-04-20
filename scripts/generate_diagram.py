"""Generates the ASAG Azure deployment architecture diagram as a PNG."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")

C = {
    "zone_dev":    "#161b22",
    "zone_azure":  "#0f2027",
    "zone_data":   "#0a1628",
    "border_dev":  "#30363d",
    "border_az":   "#1f6feb",
    "border_data": "#388bfd",
    "box_blue":    "#1f6feb",
    "box_green":   "#238636",
    "box_purple":  "#8957e5",
    "box_orange":  "#d29922",
    "box_red":     "#da3633",
    "box_teal":    "#1a7f64",
    "box_gray":    "#30363d",
    "text_bright": "#e6edf3",
    "text_muted":  "#8b949e",
    "arrow":       "#58a6ff",
    "arrow_ci":    "#3fb950",
    "arrow_data":  "#bc8cff",
}


def zone(x, y, w, h, label, border_color, bg_color, label_color="#8b949e"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05",
                          linewidth=1.5, edgecolor=border_color,
                          facecolor=bg_color, zorder=1)
    ax.add_patch(rect)
    ax.text(x + 0.18, y + h - 0.22, label,
            fontsize=8, color=label_color, fontweight="bold",
            va="top", ha="left", zorder=5, fontfamily="monospace")


def box(x, y, w, h, title, subtitle="", color="#1f6feb"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.06",
                          linewidth=1.2, edgecolor=color,
                          facecolor=color + "22", zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2 + (0.13 if subtitle else 0),
            title, fontsize=7.5, color=C["text_bright"],
            fontweight="bold", ha="center", va="center", zorder=5)
    if subtitle:
        ax.text(x + w / 2, y + h / 2 - 0.22,
                subtitle, fontsize=6.2, color=C["text_muted"],
                ha="center", va="center", zorder=5)


def arrow(x1, y1, x2, y2, color=None, label="", lw=1.4, rad=0.0):
    color = color or C["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}",
                                mutation_scale=10))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.06, my + 0.08, label,
                fontsize=6.0, color=color, ha="center", zorder=6)


# ── Zones ───────────────────────────────────────────────────────────────────
zone(0.3,  12.5, 6.8,  3.2,  "Developer / CI",                C["border_dev"],  C["zone_dev"])
zone(0.3,   0.8, 21.2, 11.4, "Microsoft Azure  (eastus)",     C["border_az"],   C["zone_azure"], "#388bfd")
zone(0.6,   1.1,  5.5,  5.8, "Data Layer",                    C["border_data"], C["zone_data"],  "#388bfd")
zone(14.9,  8.2,  2.5,  3.7, "Monitoring",                    C["box_orange"],  "#1a1200",       "#d29922")
zone(17.8,  0.3,  3.5,  1.5, "End Users",                     C["border_dev"],  C["zone_dev"])

# ── Developer / CI ──────────────────────────────────────────────────────────
box(0.6,  13.9, 2.0, 0.9, "GitHub",         "main branch",    C["box_gray"])
box(3.0,  13.9, 1.8, 0.9, "GitHub Actions", "CI/CD pipeline", C["box_green"])
box(5.5,  13.4, 1.3, 1.8, "Azure Container\nRegistry",        "asagregistry",   C["box_blue"])

arrow(2.6,  14.35, 3.0,  14.35, C["arrow_ci"],  "push")
arrow(4.8,  14.35, 5.5,  14.35, C["arrow_ci"],  "push images")
arrow(6.15, 13.4,  6.15, 11.25, C["box_blue"],  "pull", lw=1.2)

# ── App Service Plan ────────────────────────────────────────────────────────
zone(7.1, 8.5, 7.6, 3.2, "Azure App Service Plan  (Linux B2)", C["border_az"], "#0d1f35", "#388bfd")

box(7.3,  10.3, 2.2, 1.1, "React Frontend",  "Nginx  port 80",  C["box_teal"])
box(9.8,  10.3, 2.2, 1.1, "FastAPI",         "port 8000",       C["box_blue"])
box(12.3, 10.3, 2.1, 1.1, "Celery Worker",   "async tasks",     C["box_purple"])

box(7.3,  8.7,  7.1, 1.3,
    "Azure Key Vault  (asag-kv)",
    "DATABASE_URL  |  REDIS_URL  |  SECRET_KEY  |  ACR_PASSWORD",
    C["box_orange"])

arrow(10.9, 8.7,  10.9, 10.3, C["box_orange"], "secrets", lw=1.1)
arrow(13.35, 8.7, 13.35, 10.3, C["box_orange"], "secrets", lw=1.1)

# ── Data Layer ──────────────────────────────────────────────────────────────
box(0.8, 5.4,  2.2, 1.1, "PostgreSQL",        "Flexible Server B2ms", C["box_blue"])
box(3.4, 5.4,  2.2, 1.1, "Azure Cache Redis", "C1 Standard",          C["box_red"])
box(0.8, 3.9,  2.2, 1.1, "Blob Storage",      "model artifacts",      C["box_gray"])
box(3.4, 3.9,  2.2, 1.1, "Azure Databricks",  "Premium  2-10 nodes",  C["box_purple"])
box(0.8, 2.2,  4.8, 1.2, "App Insights  +  Log Analytics",            "", C["box_orange"])

# ── Monitoring sub-zone ─────────────────────────────────────────────────────
box(15.05, 10.5, 2.1, 1.0, "App Insights",  "telemetry",  C["box_orange"])
box(15.05,  9.1, 2.1, 1.0, "Alert Rules",   "5xx > 1%",   C["box_red"])
box(15.05,  8.4, 2.1, 0.5, "Email notify",  "",            C["box_gray"])

arrow(11.0, 10.85, 15.05, 10.85, C["box_orange"], "telemetry", lw=1.1)
arrow(16.1,  9.1,  16.1,   8.9,  C["box_red"],    "alert",     lw=1.1)

# ── Data-flow arrows ────────────────────────────────────────────────────────
# API <-> PostgreSQL
arrow(9.8,  10.3, 2.0,  6.5,  C["arrow_data"], "reads/writes", rad=0.2)
# Worker -> Redis (queue)
arrow(12.3, 10.85, 5.6, 5.95, C["arrow_data"], "enqueue/dequeue", rad=-0.1)
# API -> Redis (enqueue jobs)
arrow(10.9, 10.3, 4.5,  5.9,  C["arrow_data"], "", rad=0.05)
# Worker -> Blob Storage (load model)
arrow(12.3, 10.3, 2.0,  4.45, C["arrow_data"], "load model", rad=0.3)
# Databricks <-> Blob Storage
arrow(4.45, 4.45, 3.4,  4.45, C["box_purple"], "mount /mnt/asag-models", lw=1.0)
# Telemetry to App Insights data layer
arrow(12.3, 10.3, 1.7,  3.4,  C["box_orange"], "", rad=0.35, lw=1.0)

# ── User flow ───────────────────────────────────────────────────────────────
box(18.0, 0.5, 2.8, 0.9, "Browser / Teacher", "HTTPS", C["box_teal"])
arrow(18.0, 0.95, 8.4,  10.3, C["arrow"], "HTTPS", rad=-0.12)

# ── Title + legend ──────────────────────────────────────────────────────────
ax.text(11.0, 15.6, "ASAG — Azure Deployment Architecture",
        fontsize=16, fontweight="bold", color=C["text_bright"],
        ha="center", va="center")
ax.text(11.0, 15.15,
        "Peer-Aware Automated Short Answer Grading  |  FastAPI + Celery + Azure Databricks",
        fontsize=9, color=C["text_muted"], ha="center", va="center")

legend = [
    (C["arrow"],      "User / API traffic"),
    (C["arrow_ci"],   "CI/CD build & deploy"),
    (C["arrow_data"], "Internal data flow"),
    (C["box_orange"], "Secrets / monitoring"),
]
for i, (color, label) in enumerate(legend):
    lx = 7.2 + i * 3.5
    ax.plot([lx, lx + 0.55], [15.3, 15.3], color=color, lw=2.2)
    ax.text(lx + 0.7, 15.3, label, fontsize=7, color=C["text_muted"], va="center")

plt.tight_layout(pad=0.3)
plt.savefig("architecture.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: architecture.png")
