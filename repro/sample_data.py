

import argparse, json, os, random, urllib.request, zipfile
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

from datasets import load_dataset

def sample_coco(n, seed, out_dir, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "annotations.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(COCO_ANN_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        ann = json.load(z.open("annotations/instances_val2017.json"))


    person_ids = {c["id"] for c in ann["categories"] if c["name"] == "person"}
    has_person = {a["image_id"] for a in ann["annotations"] if a["category_id"] in person_ids}
    candidates = sorted([im for im in ann["images"] if im["id"] not in has_person],
                        key=lambda im: im["id"])

    
    rng = random.Random(seed)
    chosen = rng.sample(candidates, n)  


    os.makedirs(out_dir, exist_ok=True)
    picked = []
    for i, im in enumerate(chosen):
        name = f"c{i:02d}.jpg"
        urllib.request.urlretrieve(im["coco_url"], os.path.join(out_dir, name))
        picked.append({"file": name, "coco_id": im["id"], "license": im["license"], "url": im["coco_url"]})
    return picked


# can exclude genres if needed 
EXCLUDED_GENRES = {"nude_painting"}

def sample_wikiart(m, seed, out_dir, buffer_size=2000):
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    genre_names = ds.features["genre"].names
    artist_names = ds.features["artist"].names
    style_names = ds.features["style"].names


    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    os.makedirs(out_dir, exist_ok=True)
    picked = []
    for ex in ds:
        if genre_names[ex["genre"]] in EXCLUDED_GENRES:
            continue
        name = f"s{len(picked):02d}.jpg"
        ex["image"].convert("RGB").save(os.path.join(out_dir, name), quality=95)
        picked.append({"file": name,
                       "artist": artist_names[ex["artist"]],
                       "style": style_names[ex["style"]],
                       "genre": genre_names[ex["genre"]]})
        if len(picked) == m:
            break

    return picked



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_content", type=int, default=10)
    p.add_argument("--n_style", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="repro/data")
    p.add_argument("--cache", default="repro/data/cache")
    args = p.parse_args()

    if os.path.exists(os.path.join(args.out, "manifest.json")):
        print("manifest.json exists; skipping sampling.")
        return

    content = sample_coco(args.n_content, args.seed, os.path.join(args.out, "content"), args.cache)
    style = sample_wikiart(args.n_style, args.seed, os.path.join(args.out, "style"))

    manifest = {"seed": args.seed, "content": content, "style": style}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(content)} content and {len(style)} style images.")

if __name__ == "__main__":
    main()





