
import argparse, csv, os
from collections import defaultdict
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["baseline", "fm", "ng", "full", "fm_paper", "baseline_tile", "fm_tile", "fm_paper_tile"]
GRID_COLS = ["baseline", "fm", "ng", "full", "fm_paper"]                  # main grid
TILE_COLS = ["baseline", "baseline_tile", "fm_tile", "fm_paper_tile"]     # tile-only grid
PAPER = {  # Table 2 of the paper: ArtFID, FID, LPIPS
    "baseline": (26.683, 16.205, 0.5509),
    "fm":       (24.112, 15.103, 0.4973),
    "ng":       (26.542, 16.128, 0.5496),
    "full":     (21.499, 13.448, 0.4881),
}

def load_metrics(path):
    by_cfg = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_cfg[r["config"]].append((float(r["artfid"]), float(r["fid"]), float(r["lpips"])))
    return by_cfg

def make_table(by_cfg, out_path):
    lines = [r"\begin{tabular}{l|ccc|ccc}", r"\hline",
             r" & \multicolumn{3}{c|}{Paper (800 images)} & \multicolumn{3}{c}{Ours (100 images, mean $\pm$ half-range over seeds)} \\",
             r"Config & ArtFID & FID & LPIPS & ArtFID & FID & LPIPS \\", r"\hline"]
    for cfg in ORDER:
        if cfg not in by_cfg:
            continue
        rows = by_cfg[cfg]
        ours = []
        for i in range(3):
            vals = [r[i] for r in rows]
            mean, half = sum(vals) / len(vals), (max(vals) - min(vals)) / 2
            fmt = "{:.4f}" if i == 2 else "{:.2f}"
            ours.append(fmt.format(mean) + (r" $\pm$ " + fmt.format(half) if len(vals) > 1 else ""))
        paper = PAPER.get(cfg)
        p = [f"{paper[0]:.2f}", f"{paper[1]:.2f}", f"{paper[2]:.4f}"] if paper else ["--"] * 3
        lines.append(f"{cfg.replace('_', '-')} & " + " & ".join(p + ours) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


GRID_PAIRS = ["c00_s00", "c01_s01","c02_s02", "c03_s03"]   # diagonal pairs chosen

def make_grid(runs, data, seed, out_path, configs, size=256):
    cols = ["content", "style"] + [c for c in configs if os.path.isdir(os.path.join(runs, f"seed{seed}", "images", c))]
    header = 28
    grid = Image.new("RGB", (size * len(cols), header + size * len(GRID_PAIRS)), "white")
    draw = ImageDraw.Draw(grid)
    for j, col in enumerate(cols):
        draw.text((j * size + 8, 8), col, fill="black")

    for i, pair in enumerate(GRID_PAIRS):
        c, s = pair.split("_")
        for j, col in enumerate(cols):
            if col == "content":
                path = os.path.join(data, "content", c + ".jpg")
            elif col == "style":
                path = os.path.join(data, "style", s + ".jpg")
            else:
                path = os.path.join(runs, f"seed{seed}", "images", col, pair + ".png")
            img = Image.open(path).convert("RGB").resize((size, size))
            grid.paste(img, (j * size, header + i * size))
    grid.save(out_path)
    print(f"Saved grid with columns {cols} to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default=os.path.join(ROOT, "results", "metrics.csv"))
    p.add_argument("--runs", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--seed", type=int, default=1234, help="which seed's images to show in the grid")
    p.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    make_table(load_metrics(args.metrics), os.path.join(args.out, "table2.tex"))
    make_grid(args.runs, args.data, args.seed, os.path.join(args.out, "qualitative_grid.png"), GRID_COLS)
    if os.path.isdir(os.path.join(args.runs, f"seed{args.seed}", "images", "baseline_tile")):
        make_grid(args.runs, args.data, args.seed, os.path.join(args.out, "qualitative_grid_tile.png"), TILE_COLS)

if __name__ == "__main__":
    main()