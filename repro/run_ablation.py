
import argparse, json, os, sys, time
import torch
from PIL import Image


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CONFIGS = {
    "baseline": {"ng": False, "fm": False},
    "fm":       {"ng": False, "fm": True},
    "ng":       {"ng": True,  "fm": False},
    "full":     {"ng": True,  "fm": True},
    "fm_paper": {"ng": False, "fm": "paper"},   # paper-faithful FM (repro/fm_paper.py)
    # tile-only ControlNet, as described in the paper (the code's default is tile + canny)
    "baseline_tile":  {"ng": False, "fm": False,   "control": "tile"},
    "fm_tile":        {"ng": False, "fm": True,    "control": "tile"},
    "fm_paper_tile":  {"ng": False, "fm": "paper", "control": "tile"},
}

# ControlNet scales [tile, canny]. The authors' "tile_canny" setting uses [0.25, 0.40]; their "tile"
# branch in infer_style.py uses the tile ControlNet alone at 0.25. Setting the canny scale to 0
# multiplies its residuals by zero, so "tile" here is the same as their tile-only branch.
CONTROL_SCALES = {"tile_canny": [0.25, 0.40], "tile": [0.25, 0.0]}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with content/, style/ and manifest.json")
    p.add_argument("--out", required=True, help="output folder (put it on Drive)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n_content", type=int, default=None, help="use only the first N content images")
    p.add_argument("--n_style", type=int, default=None, help="use only the first N style images")
    p.add_argument("--phase", choices=["invert", "sample", "both"], default="both")
    p.add_argument("--configs", nargs="+", default=["baseline", "fm", "ng", "full"],
                   help="which configs to sample")
    return p.parse_args()

def get_pairs(data_dir, n_content, n_style):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    contents = [c["file"] for c in manifest["content"]][:n_content]
    styles = [s["file"] for s in manifest["style"]][:n_style]
    return [(c, s) for c in contents for s in styles]

def pair_name(content_file, style_file):
    return f"{content_file[:-4]}_{style_file[:-4]}"


def log_time(log_path, stage, name, seconds):
    with open(log_path, "a") as f:
        f.write(f"{stage},{name},{seconds:.1f}\n")
    print(f"[{stage}] {name}: {seconds:.1f}s", flush=True)


# inversion steps

def caption(processor, blip, image):
    # same decoding settings as the authors' generate_caption() in infer_style.py
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
    ids = blip.generate(pixel_values=inputs.pixel_values, do_sample=True, temperature=1.0,
                        length_penalty=1.0, repetition_penalty=1.5, max_length=50,
                        min_length=1, num_beams=5, top_p=0.9)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()



def phase_invert(args, pairs, lat_dir, log_path):
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    from src.config import RunConfig
    from src.eunms import Model_Type, Scheduler_Type
    from src.utils.enums_utils import get_pipes
    from inversion import run as invert
    from infer_style import init_models

    todo = [p for p in pairs if not os.path.exists(os.path.join(lat_dir, pair_name(*p) + ".pt"))]
    print(f"Phase A: {len(todo)} of {len(pairs)} pairs still to invert")
    if not todo:
        return


    # same settings infer_style.py uses
    cfg = RunConfig(model_type=Model_Type.SDXL, scheduler_type=Scheduler_Type.DDIM,
                    num_inference_steps=50, num_inversion_steps=50, num_renoise_steps=1,
                    perform_noise_correction=False, seed=args.seed)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", device_map="cuda", torch_dtype=torch.float16).eval()
    ip_instruct = init_models(cfg)
    pipe_inversion, pipe_inference = get_pipes(cfg.model_type, cfg.scheduler_type,
                                               device=cfg.device, model_name=cfg.base_model_path)
    log_time(log_path, "load", "phase_A_models", time.time() - t0)

    STYLE_Q, CONTENT_Q = "use the style from the image", "use the composition from the image"
    res = cfg.resolution


    for content_file, style_file in todo:
        name = pair_name(content_file, style_file)
        t0 = time.time()
        content = Image.open(os.path.join(args.data, "content", content_file)).convert("RGB").resize((res, res))
        style = Image.open(os.path.join(args.data, "style", style_file)).convert("RGB").resize((res, res))

        torch.manual_seed(args.seed)
        prompt = caption(processor, blip, content)

        emb = dict(
            style_embedding=ip_instruct.get_decouple_embeds(pil_image=style, prompt="", query=STYLE_Q),
            content_embedding=ip_instruct.get_decouple_embeds(pil_image=content, prompt="", query=CONTENT_Q),
            neg_style_embedding=ip_instruct.get_decouple_embeds(pil_image=content, prompt="", query=STYLE_Q),
            neg_content_embedding=ip_instruct.get_decouple_embeds(pil_image=style, prompt="", query=CONTENT_Q),
        )

        latents = {}
        for ng in (True, False):
            _, inv_latent, _, _ = invert(content, prompt, cfg,
                                            pipe_inversion=pipe_inversion, pipe_inference=pipe_inference,
                                            do_reconstruction=False, feature_extractor=ip_instruct,
                                            enable_guidance=False, used_NPI_guidance=ng, **emb)
            latents["ng" if ng else "no_ng"] = inv_latent.detach().cpu()

        torch.save({"prompt": prompt, **latents}, os.path.join(lat_dir, name + ".pt"))
        log_time(log_path, "invert", name, time.time() - t0)


    del processor, blip, ip_instruct, pipe_inversion, pipe_inference
    torch.cuda.empty_cache()



# sampling
def phase_sample(args, pairs, lat_dir, img_dir, log_path):
    import cv2
    import numpy as np
    from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
    from transformers import CLIPVisionModelWithProjection
    from src.config import RunConfig
    from src.frequency_utils import freq_exp
    from infer_style import get_canny_map
    from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline

    cfg = RunConfig(seed=args.seed)
    configs = {c: CONFIGS[c] for c in args.configs}
    res, dev, dt = cfg.resolution, cfg.device, torch.float16

    def out_path(config, name):
        return os.path.join(img_dir, config, name + ".png")

    todo = [p for p in pairs
            if not all(os.path.exists(out_path(c, pair_name(*p))) for c in configs)]
    print(f"Phase B: {len(todo)} of {len(pairs)} pairs still to sample")
    if not todo:
        return
    for c in configs:
        os.makedirs(os.path.join(img_dir, c), exist_ok=True)


      # --- load once (same components and settings as infer_style.py, control_type = "tile_canny") ---
    t0 = time.time()
    controlnet = [
        ControlNetModel.from_pretrained(cfg.tile_controlnet_path, torch_dtype=dt, use_safetensors=True).to(dev),
        ControlNetModel.from_pretrained(cfg.canny_controlnet_path, torch_dtype=dt, variant="fp16").to(dev),
    ]
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=dt).to(dev)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dt).to(dev)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        cfg.base_model_path, controlnet=controlnet, vae=vae, image_encoder=image_encoder,
        torch_dtype=dt, use_safetensors=True, variant="fp16").to(dev)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.enable_gradient_checkpointing()
    pipe.load_ip_adapter(cfg.IP_path, subfolder="sdxl_models",
                         weight_name="ip-adapter_sdxl_vit-h.safetensors", image_encoder_folder=None)
    pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, 2.5, 0.0]}})
    log_time(log_path, "load", "phase_B_models", time.time() - t0)

    full_mask = Image.new("RGB", (res, res), color=(255, 255, 255))
    neg_prompt = ("watermark, lowres, low quality, worst quality, deformed, glitch, "
                  "low contrast, noisy, saturation, blurry")

    for content_file, style_file in todo:
        name = pair_name(content_file, style_file)
        saved = torch.load(os.path.join(lat_dir, name + ".pt"))

        content_path = os.path.join(args.data, "content", content_file)
        content = Image.open(content_path).convert("RGB").resize((res, res))
        style = Image.open(os.path.join(args.data, "style", style_file)).convert("RGB").resize((res, res))
        canny = get_canny_map(np.array(cv2.imread(content_path))).resize((res, res))

        for config, sw in configs.items():
            if os.path.exists(out_path(config, name)):
                continue
            t0 = time.time()
            inv_latent = saved["ng" if sw["ng"] else "no_ng"].to(dev)

            if sw["fm"] == "paper":
                from fm_paper import freq_paper
                start = freq_paper(inv_latent, alpha=0.7, cutoff=0.3, noise_sigma=0.3, seed=args.seed)
            elif sw["fm"]:
                torch.manual_seed(args.seed)
                _, latent_l, _ = freq_exp(inv_latent, d_s=0.3, d_t=0.9, alpha=0.7, filter_type="gaussian_b")
                start = latent_l.to(inv_latent.dtype)
            else:
                start = inv_latent

            out = pipe(
                prompt=saved["prompt"], negative_prompt=neg_prompt,
                num_inference_steps=50, eta=1.0,
                mask_image=full_mask, image=content,
                control_image=[content, canny], controlnet_conditioning_scale=CONTROL_SCALES[sw.get("control", "tile_canny")],
                ip_adapter_image=style, latents=start,
                guidance_scale=cfg.guidance_scale, npi_interp=0.5,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
            ).images[0]
            out.save(out_path(config, name))
            log_time(log_path, f"sample_{config}", name, time.time() - t0)


    del pipe, controlnet, image_encoder, vae
    torch.cuda.empty_cache()



def main():
    args = parse_args()
    pairs = get_pairs(args.data, args.n_content, args.n_style)

    lat_dir = os.path.join(args.out, f"seed{args.seed}", "latents")
    img_dir = os.path.join(args.out, f"seed{args.seed}", "images")
    os.makedirs(lat_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    log_path = os.path.join(args.out, f"seed{args.seed}", "timings.csv")

    print(f"{len(pairs)} pairs x {len(args.configs)} configs, seed {args.seed}")
    if args.phase in ("invert", "both"):
        phase_invert(args, pairs, lat_dir, log_path)
    if args.phase in ("sample", "both"):
        phase_sample(args, pairs, lat_dir, img_dir, log_path)
    print("Done.")

if __name__ == "__main__":
    main()



