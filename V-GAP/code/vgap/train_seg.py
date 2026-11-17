#!/usr/bin/env python3
"""
YOLO segmentation training for the 6D pose estimation / disassembly pipeline.

Four classes (gearbox, hemisphere, L-shape, T-shape), Roboflow-labeled,
frames extracted from short orbit videos. Target deployment: NVIDIA Jetson
with TensorRT, mask feeds a point-cloud crop for registration.

Modes
-----
  resplit   Rebuild train/valid/test: whole video clips stay in one split,
            single images are split randomly.
  train     Train a segmentation model.
  compare   Train yolo26s-seg and yolo11s-seg back to back and diff them.
  export    Export a trained .pt to TensorRT. MUST be run on the Jetson.

  doctor    Find bbox-format labels; --fix quarantines them. Also counts null
            (background) images and flags hard examples whose filename class
            disagrees with the label (L/T mix-ups).

Video frames are split clip-wise. Single images (hard examples named
..._lowconf_c46 / _missed_ / _manual_ / _as_<class>_ by collect_hard_examples.py,
and null background images with an empty/missing label) get a normal random
split, stratified by class.

Typical run:
  python train_seg.py resplit --src export --dst dataset
  python train_seg.py doctor --root dataset          # read the report
  python train_seg.py doctor --root dataset --fix    # quarantine bbox labels
  python train_seg.py train --data dataset/data.yaml --name yolo26s_v2
  # then, on the Jetson:
  python train_seg.py export --weights runs/segment/yolo26s/weights/best.pt
"""

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# --------------------------------------------------------------------------
# Augmentation / training config.
#
# Rationale for the non-default values, since these are the ones that matter
# for this dataset:
#
#   degrees=20      Ultralytics defaults this to 0. In-plane rotation is the
#                   cheapest orientation robustness available and the dataset
#                   only has 3 fixed elevation angles. Raise toward 180 if the
#                   wrist camera can roll arbitrarily.
#   scale=0.6       Above the 0.5 default. Working distance varies between the
#                   wall-mounted and wrist-mounted RealSense.
#   fliplr=0.0      OFF deliberately. The L-shape is chiral; a mirrored L is a
#                   silhouette that only occurs if the part can physically rest
#                   flipped over. If it can, set this to 0.5 and gain free data.
#   copy_paste=0.0  OFF. It pastes instances into other images from this same
#                   dataset, which all share one background. It does not
#                   synthesize viewpoints or fix background overfit.
#   hsv_v=0.5       Above default. Black PLA under varying light is the known
#                   weak point; value jitter is the direct lever on that.
#   mask_ratio=2    Finer mask supervision than the default 4. Mask boundary
#                   quality propagates straight into the point-cloud crop.
#   close_mosaic=15 Last 15 epochs train on clean, un-mosaicked images.
# --------------------------------------------------------------------------
TRAIN_CFG = dict(
    imgsz=640,
    epochs=150,
    batch=16,
    patience=40,
    optimizer="auto",
    seed=0,
    deterministic=True,
    val=True,
    plots=True,
    # geometric
    degrees=20.0,
    scale=0.6,
    translate=0.1,
    shear=0.0,
    perspective=0.0005,
    fliplr=0.0,
    flipud=0.0,
    # photometric
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.5,
    # composition
    mosaic=1.0,
    close_mosaic=15,
    mixup=0.0,
    copy_paste=0.0,
    # segmentation-specific
    mask_ratio=2,
    overlap_mask=True,
)

# Frame filenames from a Roboflow video export usually look like
#   gearbox_45deg_side1_mp4-0042_jpg.rf.a1b2c3d4.jpg
# so everything before the container-extension marker names the source clip.
CLIP_PATTERN = re.compile(r"_(?:mp4|mov|avi|mkv|MP4|MOV)-|\.rf\.")
VIDEO_MARKER = re.compile(r"_(?:mp4|mov|avi|mkv|MP4|MOV)-")

# Single images from collect_hard_examples.py are named
#   <class>_<color>_<YYYYmmdd_HHMMSS_mmm>_<reason>_c<conf>.png
# with reason in lowconf / missed / manual / as_<otherclass>. After a Roboflow
# export that becomes ..._lowconf_c46_png.rf.<hash>.jpg, so match the
# reason + confidence tail anywhere in the name.
SINGLE_PATTERN = r"_(?:lowconf|missed|manual|as_[A-Za-z0-9_]+?)_c\d{1,3}(?:_png|_jpg|\.|$)"


# --------------------------------------------------------------------------
# resplit
# --------------------------------------------------------------------------
def clip_of(name: str) -> str:
    """Source clip identifier for a frame filename."""
    return CLIP_PATTERN.split(name)[0]


def label_of(img: Path) -> Path:
    return img.parent.parent / "labels" / f"{img.stem}.txt"


def label_classes(img: Path) -> list:
    """Class ids in an image's label file; [] for a null (background) image."""
    lbl = label_of(img)
    if not lbl.exists():
        return []
    return [int(line.split()[0]) for line in lbl.read_text().splitlines() if line.strip()]


def split_groups(by_class: dict, ratios, rng) -> dict:
    """Shuffle each class's groups and cut them train/valid/test."""
    assignment = {"train": [], "valid": [], "test": []}
    for cls, groups in sorted(by_class.items()):
        groups = sorted(groups)
        rng.shuffle(groups)
        n = len(groups)
        a = int(round(ratios[0] * n))
        b = int(round((ratios[0] + ratios[1]) * n))
        # guarantee at least one group in valid when the class has >= 2
        if n >= 2 and a >= n:
            a = n - 1
        assignment["train"] += groups[:a]
        assignment["valid"] += groups[a:b]
        assignment["test"] += groups[b:]
    return assignment


def resplit(src: Path, dst: Path, ratios=(0.7, 0.2, 0.1), seed=0, stratify=True,
            single_pattern=SINGLE_PATTERN):
    """
    Rebuild the split.

    Video frames: every frame from a given clip lands in exactly one of
    train/valid/test. Frames from a 5-second orbit are near-duplicates; a
    random per-frame split puts near-identical images on both sides, and
    validation mAP becomes a measure of memorization rather than
    generalization.

    Single images (hard examples from collect_hard_examples.py, and null
    background images with an empty or missing label file): independent
    shots, so they get a normal random split, stratified by the class in
    their label (nulls are their own stratum). Roboflow augmented copies of
    one single image share a name prefix and are kept together.
    """
    images = [
        p
        for p in list(src.rglob("images/*.jpg")) + list(src.rglob("images/*.png"))
        if ".venv" not in p.parts and "site-packages" not in p.parts
    ]
    if not images:
        sys.exit(f"No images found under {src}. Expecting a Roboflow YOLO export.")

    single_re = re.compile(single_pattern)
    clips = defaultdict(list)    # video clip name -> frames
    singles = defaultdict(list)  # single image name -> itself + augmented copies
    n_hard = n_null = 0
    for img in images:
        # A single with an empty/missing label is a null image whatever its
        # name says (e.g. background shots saved as gearbox_white_..._lowconf).
        is_single = single_re.search(img.name) or not VIDEO_MARKER.search(img.name)
        if is_single and not label_classes(img):
            singles[clip_of(img.name)].append(img)
            n_null += 1
        elif single_re.search(img.name):
            singles[clip_of(img.name)].append(img)
            n_hard += 1
        else:
            clips[clip_of(img.name)].append(img)

    # Surface the grouping so a bad regex is visible, not silent.
    n_video = sum(len(v) for v in clips.values())
    print(f"{len(images)} images: {n_video} video frames in {len(clips)} clips, "
          f"{n_hard} hard-example singles, {n_null} null/background singles.")
    print("Sample clips:")
    for clip in sorted(clips)[:5]:
        print(f"  {clip:<40} {len(clips[clip]):4d} frames")
    print("Sample singles:")
    for name in sorted(singles)[:5]:
        print(f"  {name}")

    if clips and len(clips) < 10:
        print(
            f"WARNING: only {len(clips)} distinct clips detected. If that looks "
            f"wrong, your filenames may not encode the source video and this "
            f"split will not be clip-safe. Inspect a few names:\n  "
            + "\n  ".join(p.name for p in images[:3])
        )

    # Stratify by object class so a small test set cannot starve a class.
    # Clip names carry a color prefix (black_gearbox, gray_L-Shape,
    # reflective_sphere, ...), so match the shape token anywhere in the name
    # rather than assuming it comes first.
    def class_key(clip: str) -> str:
        if not stratify:
            return "_all"
        low = clip.lower().replace("-", "").replace("_", "")
        for token in ("gearbox", "lshape", "tshape", "sphere"):
            if token in low:
                return token
        return "_unknown"

    def single_key(name: str) -> str:
        if not stratify:
            return "_all"
        ids = label_classes(singles[name][0])
        if not ids:
            return "_null"
        return str(max(set(ids), key=ids.count))  # most common class in the image

    rng = random.Random(seed)
    clip_by_class = defaultdict(list)
    for clip in sorted(clips):
        clip_by_class[class_key(clip)].append(clip)
    single_by_class = defaultdict(list)
    for name in sorted(singles):
        single_by_class[single_key(name)].append(name)

    clip_split = split_groups(clip_by_class, ratios, rng)
    single_split = split_groups(single_by_class, ratios, rng)

    if dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "valid", "test"):
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        groups = [clips[c] for c in clip_split[split]] + [singles[n] for n in single_split[split]]
        for group in groups:
            for img in group:
                lbl = label_of(img)
                shutil.copy2(img, dst / split / "images" / img.name)
                if lbl.exists():
                    shutil.copy2(lbl, dst / split / "labels" / lbl.name)
                else:
                    # explicit empty label = background image, not "unlabeled"
                    (dst / split / "labels" / f"{img.stem}.txt").write_text("")

    # Carry class names over from the Roboflow data.yaml.
    src_yaml = next(src.rglob("data.yaml"), None)
    if src_yaml is None:
        sys.exit("No data.yaml in the export; cannot recover class names.")
    meta = yaml.safe_load(src_yaml.read_text())
    names = meta["names"]
    (dst / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dst.resolve()),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
        )
    )

    print(f"\nSplit written to {dst}")
    print(f"  {'split':5s}  {'clips':>5s} {'frames':>6s}  {'singles':>7s} {'(null)':>6s}  {'total':>5s}")
    for split in ("train", "valid", "test"):
        n_frames = sum(len(clips[c]) for c in clip_split[split])
        n_single = sum(len(singles[n]) for n in single_split[split])
        n_bg = sum(len(singles[n]) for n in single_split[split] if not label_classes(singles[n][0]))
        print(f"  {split:5s}  {len(clip_split[split]):5d} {n_frames:6d}  {n_single:7d} {n_bg:6d}  "
              f"{n_frames + n_single:5d}")
    verify_class_coverage(dst, names)
    print(
        "\nExpect mask mAP to drop relative to a random per-frame split. "
        "That lower number is the honest one; baseline against it."
    )


def verify_class_coverage(root: Path, names):
    """Warn if any class is missing from valid or test."""
    idx_to_name = names if isinstance(names, list) else [names[i] for i in sorted(names)]
    for split in ("train", "valid", "test"):
        seen = set()
        for lbl in (root / split / "labels").glob("*.txt"):
            for line in lbl.read_text().splitlines():
                if line.strip():
                    seen.add(int(line.split()[0]))
        missing = [idx_to_name[i] for i in range(len(idx_to_name)) if i not in seen]
        if missing:
            print(f"  WARNING: {split} is missing class(es): {', '.join(missing)}")


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def shape_of(name: str):
    """Normalize a class name or hard-example filename to l / t / gearbox / hemisphere."""
    low = name.lower().replace("-", "_")
    if "gear" in low:
        return "gearbox"
    if "sphere" in low or low.startswith(("hemi", "dome")):
        return "hemisphere"
    if low.startswith(("l_block", "l_shape", "lshape")) or low in ("l", "l_block", "l_shape"):
        return "l"
    if low.startswith(("t_block", "t_shape", "tshape")) or low in ("t", "t_block", "t_shape"):
        return "t"
    return None


def doctor(root: Path, fix: bool = False):
    """
    Find label files containing detection-format (bbox) lines in a dataset that
    is supposed to be all polygons.

    A YOLO segmentation line is `cls x1 y1 x2 y2 ... xn yn` — an odd token count
    of at least 7. A detection line is `cls cx cy w h` — exactly 5 tokens.

    Ultralytics' reaction to even one bbox line is to drop EVERY segment in the
    split and silently train detection instead, which then crashes YOLO26's
    semantic segmentation loss on an empty mask tensor. So this is fatal, not
    cosmetic.

    With --fix, the offending image/label pairs are moved to _quarantine/ rather
    than deleted, so nothing is lost and the dataset becomes pure-polygon.
    """
    names = None
    if (root / "data.yaml").exists():
        names = yaml.safe_load((root / "data.yaml").read_text()).get("names")
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names)]

    single_re = re.compile(SINGLE_PATTERN)
    total_bad = 0
    mismatches = []
    for split in ("train", "valid", "test"):
        lbl_dir = root / split / "labels"
        if not lbl_dir.is_dir():
            continue
        bad = []
        # Null/background images (empty or missing label file) are valid
        # negatives, not errors. Count them so you can see they made it in.
        n_null = sum(
            1 for img in (root / split / "images").glob("*")
            if not (lbl_dir / f"{img.stem}.txt").exists()
            or not (lbl_dir / f"{img.stem}.txt").read_text().strip()
        )
        for lbl in sorted(lbl_dir.glob("*.txt")):
            if names:
                # only the leading target of a hard-example name (not "as_gearbox")
                m = re.match(r"(l_block|t_block|hemisphere|gearbox)_", lbl.name)
                want = shape_of(m.group(1)) if m and single_re.search(lbl.name) else None
                got = {shape_of(names[int(l.split()[0])]) for l in lbl.read_text().splitlines()
                       if l.strip() and l.split()[0].isdigit() and int(l.split()[0]) < len(names)}
                if want and got and want not in got:
                    mismatches.append((split, lbl.name, want, sorted(got)))
            for i, line in enumerate(lbl.read_text().splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) == 5:
                    bad.append((lbl, i, "bbox format (5 tokens)"))
                elif len(parts) < 7 or len(parts) % 2 == 0:
                    bad.append((lbl, i, f"malformed ({len(parts)} tokens)"))

        print(f"\n{split}: {len(bad)} bad line(s) across {len({b[0] for b in bad})} file(s); "
              f"{n_null} null/background image(s) (kept)")
        for lbl, lineno, why in bad[:20]:
            print(f"  {lbl.name}:{lineno}  {why}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")
        total_bad += len(bad)

        if fix and bad:
            q_img = root / "_quarantine" / split / "images"
            q_lbl = root / "_quarantine" / split / "labels"
            q_img.mkdir(parents=True, exist_ok=True)
            q_lbl.mkdir(parents=True, exist_ok=True)
            for lbl in {b[0] for b in bad}:
                for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
                    img = root / split / "images" / f"{lbl.stem}{ext}"
                    if img.exists():
                        shutil.move(str(img), str(q_img / img.name))
                        break
                shutil.move(str(lbl), str(q_lbl / lbl.name))
            # stale caches will otherwise resurrect the removed entries
            for cache in (root / split).glob("labels.cache"):
                cache.unlink()
            print(f"  moved {len({b[0] for b in bad})} pair(s) to {root / '_quarantine' / split}")

    if mismatches:
        print(f"\nLABEL CHECK: {len(mismatches)} hard-example image(s) whose filename class "
              f"is not in its label (likely an L/T mix-up in labeling). Fix these in "
              f"Roboflow; --fix does not touch them:")
        for split, name, want, got in mismatches[:30]:
            print(f"  {split}/{name}: filename says {want}, label has {', '.join(got)}")
        if len(mismatches) > 30:
            print(f"  ... and {len(mismatches) - 30} more")
    elif names:
        print("\nLABEL CHECK: hard-example filenames agree with their labels.")

    if total_bad == 0:
        print("\nAll labels are polygon format. Nothing to fix.")
    elif not fix:
        print(f"\n{total_bad} bad line(s) total. Re-run with --fix to quarantine them.")
    else:
        print(
            "\nQuarantined. Re-annotate those instances in Roboflow with the polygon "
            "tool if you want them back; 13 or so images out of 1,400 is not worth "
            "blocking on."
        )


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def train(model_name: str, data: Path, run_name: str, overrides: dict):
    from ultralytics import YOLO

    cfg = {**TRAIN_CFG, **overrides}
    print(f"\n=== training {model_name} -> runs/segment/{run_name} ===")
    model = YOLO(model_name)
    model.train(data=str(data), name=run_name, **cfg)

    # Evaluate on the held-out clips. split="test" is the number that matters;
    # the per-epoch val numbers are used for checkpoint selection and are
    # mildly optimistic as a result.
    metrics = model.val(data=str(data), split="test", imgsz=cfg["imgsz"])
    result = {
        "model": model_name,
        "run": run_name,
        "mask_mAP50_95": float(metrics.seg.map),
        "mask_mAP50": float(metrics.seg.map50),
        "box_mAP50_95": float(metrics.box.map),
        "weights": f"runs/segment/{run_name}/weights/best.pt",
    }
    print(
        f"\n{model_name}  test mask mAP50-95={result['mask_mAP50_95']:.4f}  "
        f"mask mAP50={result['mask_mAP50']:.4f}"
    )
    return result


def compare(data: Path, overrides: dict):
    """Train both candidates on identical data and settings, then diff."""
    results = [
        train("yolo26s-seg.pt", data, "yolo26s", overrides),
        train("yolo11s-seg.pt", data, "yolo11s", overrides),
    ]
    print("\n" + "=" * 62)
    print(f"{'model':<18}{'mask mAP50-95':>16}{'mask mAP50':>14}")
    print("-" * 62)
    for r in results:
        print(f"{r['model']:<18}{r['mask_mAP50_95']:>16.4f}{r['mask_mAP50']:>14.4f}")
    best = max(results, key=lambda r: r["mask_mAP50_95"])
    print("=" * 62)
    print(f"Best on mask mAP50-95: {best['model']}  ({best['weights']})")
    print(
        "Accuracy is only half the decision. Export both to TensorRT on the "
        "Jetson and compare real latency before committing."
    )
    return results


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def export(weights: Path, imgsz: int, half: bool, int8: bool, data: Path | None):
    """
    Export to a TensorRT engine.

    Run this ON THE JETSON. TensorRT engines are built against the specific
    GPU architecture, TensorRT version and driver on the build machine. An
    engine built on a desktop RTX card will not load on a Jetson.
    """
    from ultralytics import YOLO

    print(f"Exporting {weights} to TensorRT (imgsz={imgsz}, half={half}, int8={int8})")
    if int8 and data is None:
        sys.exit("INT8 export needs --data for the calibration set.")

    model = YOLO(str(weights))
    path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half and not int8,
        int8=int8,
        data=str(data) if int8 else None,
        workspace=4,
        simplify=True,
    )
    print(f"\nEngine written to {path}")
    print(
        "Benchmark it before wiring into the pipeline, and profile your capture "
        "and mask-decode path too — those often cost more than the model does.\n"
        "At inference, retina_masks=True gives full-resolution masks instead of "
        "upsampled 160x160 prototypes, which is usually the real limit on "
        "boundary precision for a point-cloud crop."
    )


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("resplit", help="rebuild a clip-wise train/valid/test split")
    s.add_argument("--src", type=Path, required=True, help="unzipped Roboflow export")
    s.add_argument("--dst", type=Path, default=Path("dataset"))
    s.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.2, 0.1])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-stratify", action="store_true")
    s.add_argument("--single-pattern", default=SINGLE_PATTERN,
                   help="regex marking single (non-video) images; default matches "
                        "collect_hard_examples.py names (_lowconf_c46, _missed_c00, ...)")

    for name, helptext in (("train", "train one model"), ("compare", "train yolo26s and yolo11s")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--data", type=Path, default=Path("dataset/data.yaml"))
        s.add_argument("--epochs", type=int)
        s.add_argument("--batch", type=int)
        s.add_argument("--imgsz", type=int)
        s.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu")
        if name == "train":
            s.add_argument("--model", default="yolo26s-seg.pt")
            s.add_argument("--name", default="seg")

    s = sub.add_parser("doctor", help="find bbox-format lines in a polygon dataset")
    s.add_argument("--root", type=Path, default=Path("dataset"))
    s.add_argument("--fix", action="store_true", help="quarantine the offending pairs")

    s = sub.add_parser("export", help="export to TensorRT (run on the Jetson)")
    s.add_argument("--weights", type=Path, required=True)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--int8", action="store_true")
    s.add_argument("--fp32", action="store_true", help="disable FP16")
    s.add_argument("--data", type=Path, default=None, help="data.yaml, required for --int8")

    a = p.parse_args()

    if a.mode == "resplit":
        resplit(a.src, a.dst, tuple(a.ratios), a.seed, stratify=not a.no_stratify,
                single_pattern=a.single_pattern)
        return

    if a.mode == "doctor":
        doctor(a.root, fix=a.fix)
        return

    if a.mode == "export":
        export(a.weights, a.imgsz, half=not a.fp32, int8=a.int8, data=a.data)
        return

    if not a.data.exists():
        sys.exit(f"{a.data} not found. Run the resplit mode first.")
    overrides = {
        k: v
        for k, v in (("epochs", a.epochs), ("batch", a.batch), ("imgsz", a.imgsz), ("device", a.device))
        if v is not None
    }

    if a.mode == "train":
        train(a.model, a.data, a.name, overrides)
    else:
        compare(a.data, overrides)


if __name__ == "__main__":
    main()