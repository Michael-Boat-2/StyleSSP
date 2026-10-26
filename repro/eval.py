
import argparse, csv, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "evaluation")
CONFIGS = ["baseline", "fm", "ng", "full"]
PAIR_RE = re.compile(r"^(c\d+)_(s\d+)\.png$")


def build_eval_folders(img_dir, data_dir, work_dir, pairs_filter=None):
    """For one config: make tar/, cnt/, sty/ folders with one file per stylised
    image, all named identically so sorted order pairs them up."""
    tar, cnt, sty = (os.path.join(work_dir, d) for d in ("tar", "cnt", "sty"))
    for d in (tar, cnt, sty):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    names = sorted(f for f in os.listdir(img_dir) if PAIR_RE.match(f))
    if pairs_filter is not None:
        names = [n for n in names if n[:-4] in pairs_filter]

    for name in names:
        c, s = PAIR_RE.match(name).groups()
        os.symlink(os.path.join(img_dir, name), os.path.join(tar, name))
        os.symlink(os.path.join(data_dir, "content", c + ".jpg"), os.path.join(cnt, name[:-4] + ".jpg"))
        os.symlink(os.path.join(data_dir, "style", s + ".jpg"), os.path.join(sty, name[:-4] + ".jpg"))
    return len(names)




def run_artfid(work_dir, python=sys.executable):
    cmd = [python, "eval_artfid.py",
           "--sty", os.path.join(work_dir, "sty"),
           "--cnt", os.path.join(work_dir, "cnt"),
           "--tar", os.path.join(work_dir, "tar"),
           "--mode", "art_fid_inf", "--batch_size", "16", "--num_workers", "2"]
    out = subprocess.run(cmd, cwd=EVAL_DIR, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr[-3000:])
        raise RuntimeError("eval_artfid.py failed")
    line = [l for l in out.stdout.splitlines() if l.startswith("ArtFID:")][-1]
    nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+", line)
    artfid, fid, lpips, lpips_gray = map(float, nums[:4])
    return {"artfid": artfid, "fid": fid, "lpips": lpips, "lpips_gray": lpips_gray}




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="folder containing seed*/images/<config>/")
    p.add_argument("--data", required=True, help="folder with content/ and style/")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "metrics.csv"))
    p.add_argument("--work", default="/content/eval_tmp")
    args = p.parse_args()

    seeds = sorted(d for d in os.listdir(args.runs) if d.startswith("seed"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)


    rows = []
    for seed in seeds:
        for config in CONFIGS:
            img_dir = os.path.join(args.runs, seed, "images", config)
            if not os.path.isdir(img_dir):
                continue
            n = build_eval_folders(img_dir, args.data, args.work)
            print(f"Evaluating {seed} / {config} on {n} images...", flush=True)
            m = run_artfid(args.work)
            rows.append({"seed": seed[4:], "config": config, "n_images": n, **m})
            print(f"  ArtFID {m['artfid']:.3f}  FID {m['fid']:.3f}  LPIPS {m['lpips']:.4f}", flush=True)

            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")



if __name__ == "__main__":
    main()