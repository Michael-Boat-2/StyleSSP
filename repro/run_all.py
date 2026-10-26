import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def run(args):
    cmd = [sys.executable] + args
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser(description="Reproduce write-up data.")
    p.add_argument("--work", required=True, help="persistent folder for data/ and runs/ ")
    args = p.parse_args()
    data, runs = os.path.join(args.work, "data"), os.path.join(args.work, "runs")

    run([f"{HERE}/sample_data.py", "--out", data, "--cache", os.path.join(data, "cache")])
    for seed in ["1234", "42"]:
        run([f"{HERE}/run_ablation.py", "--data", data, "--out", runs, "--seed", seed])
    run([f"{HERE}/run_ablation.py", "--data", data, "--out", runs, "--seed", "1234",
         "--phase", "sample", "--configs", "fm_paper"])
    run([f"{HERE}/eval.py", "--runs", runs, "--data", data])
    run([f"{HERE}/make_figures.py", "--runs", runs, "--data", data])

if __name__ == "__main__":
    main()