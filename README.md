# Reproducing StyleSSP (CVPR 2025): FM and NG ablation

Reproduction of *StyleSSP: Sampling StartPoint Enhancement for Training-free Diffusion-based
Method for Style Transfer* (Xu et al., CVPR 2025), built on the authors' released code.
The original README is kept as [README_original.md](README_original.md).

## Claim tested
Table 2 of the paper: frequency manipulation (FM) and negative guidance via inversion (NG)
each improve style transfer, and together give the best ArtFID / FID / LPIPS.

## Result (short)
- NG has a negligible effect, consistent with the paper.
- FM worsens LPIPS, FID and ArtFID in the released code, the opposite of the paper.
- A paper-faithful FM (`repro/fm_paper.py`) removes ~60% of that harm but still does not beat the baseline.
- With the paper's tile-only ControlNet (canny off), the baseline matches the paper's LPIPS (0.562 vs 0.551), but FM still worsens every metric.

Numbers: [`results/metrics.csv`](results/metrics.csv), table: [`results/table2.tex`](results/table2.tex),
figure: [`results/qualitative_grid.png`](results/qualitative_grid.png).

another figure : [`results/readme_grid.png`](results/readme_grid.jpg)


## Reproduce everything
Requires an NVIDIA GPU with ≥40 GB memory (we used a Colab A100 40 GB) and ~40 GB of disk for model weights.
Full run time ≈ 7 hours.

    conda env create -f environment.yml && conda activate stylessp-repro
    # download checkpoints: see repro/setup_colab.ipynb (cell "Downloading Checkpoints")
    python repro/run_all.py --work <persistent folder, e.g. /content/drive/MyDrive/stylessp>

`repro/setup_colab.ipynb` runs the same steps on Google Colab. Every stage resumes from
what already exists in `--work`, so a re-run only redoes missing work.

## What is ours
| File | Purpose |
|---|---|
| `repro/sample_data.py` | Seeded sampling of 10 COCO (no `person`) and 10 WikiArt (no nude paintings) images; writes `manifest.json` |
| `repro/run_ablation.py` | Two-phase runner: inversion (NG on/off) → saved latents → sampling (FM variants); seeded and resumable |
| `repro/fm_paper.py` | FM as written in the paper (Eq. 5, 7, 9) |
| `repro/eval.py` | Builds paired folders and calls the authors' `evaluation/eval_artfid.py` |
| `repro/make_figures.py` | Results table and qualitative grid |
| `repro/run_all.py` | The one command |
| `repro/setup_colab.ipynb` | Colab environment and checkpoint setup |

No code files from the original repository were modified. Original README and environment file were renamed, however.

## Deviations from the paper's setup
100 pairs (paper: 800), 10 distinct styles (paper: 40), two seeds (1234, 42). The released code
differs from the paper in several places (ReNoise inversion, band-pass FM filter, tile + canny
ControlNets, a 0.2× content-latent mix at the sampling start, and more); see the write-up.

## Data
`repro/data/manifest.json` lists every image used (COCO IDs and licences, WikiArt artist/style/genre).
Images are not committed; `sample_data.py` re-downloads them.

## Credits
Original code © 2025 Bytedance Ltd., Apache-2.0 (see [LICENCE.txt](LICENCE.txt)). Metric code from StyleID.