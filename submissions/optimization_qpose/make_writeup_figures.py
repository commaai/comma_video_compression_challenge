from pathlib import Path
import json
import math
import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submissions" / "optimization_qpose"
ASSETS = SUB / "writeup_assets"
FRAMES_OUT = ASSETS / "frames"
CHARTS_OUT = ASSETS / "charts"
TABLES_OUT = ASSETS / "tables"

FRAMES_OUT.mkdir(parents=True, exist_ok=True)
CHARTS_OUT.mkdir(parents=True, exist_ok=True)
TABLES_OUT.mkdir(parents=True, exist_ok=True)

ORIG_VIDEO = ROOT / "videos" / "0.mkv"
RECON_DIR = Path("/tmp/optimization_qpose_writeup_frames")

FRAME_IDS = [0, 25, 75, 150, 225, 300, 375, 450, 525, 599]

FINAL = {
    "pose": 0.00061985,
    "seg": 0.00071020,
    "size": 277087,
    "original_size": 37545489,
    "rate": 0.00738003,
    "score": 0.33,
}

EXPERIMENTS = [
    {
        "name": "Final qpose/tile-action variant",
        "size": 277087,
        "pose": 0.00061985,
        "seg": 0.00071020,
        "score": 0.33,
        "decision": "kept",
        "lesson": "Stable temporal behavior; conservative tile corrections preserve PoseNet.",
    },
    {
        "name": "Lossless temporal-delta mask codec",
        "size": 648559,
        "pose": None,
        "seg": None,
        "score": None,
        "decision": "rejected",
        "lesson": "Simple temporal delta coding was substantially larger than the original mask stream.",
    },
    {
        "name": "Simple arithmetic mask codec",
        "size": 305334,
        "pose": None,
        "seg": None,
        "score": None,
        "decision": "rejected",
        "lesson": "Hand-written spatial/temporal contexts were still worse than the existing representation.",
    },
    {
        "name": "Lossy mask re-encode CRF50",
        "size": 273630,
        "pose": 0.00529962,
        "seg": 0.00082431,
        "score": 0.49,
        "decision": "rejected",
        "lesson": "Small byte savings were overwhelmed by temporal distortion.",
    },
    {
        "name": "Lossy mask re-encode CRF60",
        "size": 181142,
        "pose": 0.04886454,
        "seg": 0.00201768,
        "score": 1.02,
        "decision": "rejected",
        "lesson": "Excellent rate reduction, but catastrophic PoseNet degradation.",
    },
    {
        "name": "Aggressive keyframe/g=1 mask re-encode",
        "size": 340225,
        "pose": 0.26924199,
        "seg": 0.00188160,
        "score": 2.06,
        "decision": "rejected",
        "lesson": "Larger archive and severe temporal failure.",
    },
]


def score_terms(pose, seg, rate):
    pose_term = math.sqrt(10.0 * pose)
    seg_term = 100.0 * seg
    rate_term = 25.0 * rate
    return pose_term, seg_term, rate_term


def read_original_frame(video_path: Path, frame_id: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def list_recon_images():
    return sorted(
        list(RECON_DIR.rglob("*.png"))
        + list(RECON_DIR.rglob("*.jpg"))
        + list(RECON_DIR.rglob("*.jpeg"))
    )


def find_recon_frame(frame_id: int):
    candidates = [
        f"{frame_id}.png",
        f"{frame_id:03d}.png",
        f"{frame_id:04d}.png",
        f"{frame_id:05d}.png",
        f"{frame_id:06d}.png",
        f"{frame_id}.jpg",
        f"{frame_id:03d}.jpg",
        f"{frame_id:04d}.jpg",
        f"{frame_id:05d}.jpg",
        f"{frame_id:06d}.jpg",
    ]

    for name in candidates:
        for p in RECON_DIR.rglob(name):
            if p.exists():
                return p

    imgs = list_recon_images()
    if frame_id < len(imgs):
        return imgs[frame_id]

    return None


def read_recon_frame(frame_id: int):
    p = find_recon_frame(frame_id)
    if p is None:
        return None
    img = iio.imread(p)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


def resize_like(img, ref):
    if img.shape[:2] == ref.shape[:2]:
        return img
    return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)


def save_frame_comparisons():
    frame_rows = []

    for fid in FRAME_IDS:
        orig = read_original_frame(ORIG_VIDEO, fid)
        recon = read_recon_frame(fid)

        if orig is None or recon is None:
            print(f"Skipping frame {fid}: missing original or reconstruction")
            continue

        recon = resize_like(recon, orig)

        absdiff = np.abs(orig.astype(np.int16) - recon.astype(np.int16)).astype(np.uint8)
        diff_boost = np.clip(absdiff * 4, 0, 255).astype(np.uint8)

        error_gray = np.mean(absdiff.astype(np.float32), axis=-1)
        max_err = float(error_gray.max()) if error_gray.size else 1.0
        heat = error_gray / max(max_err, 1e-6)

        mae = float(error_gray.mean())
        p95 = float(np.percentile(error_gray, 95))

        fig = plt.figure(figsize=(15, 4))
        panels = [
            ("Original", orig),
            ("Reconstruction", recon),
            ("Absolute difference x4", diff_boost),
            ("Normalized error heatmap", heat),
        ]

        for i, (title, img) in enumerate(panels):
            ax = fig.add_subplot(1, 4, i + 1)
            if "heatmap" in title:
                ax.imshow(img, vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"Frame {fid}: mean absolute error {mae:.2f}, p95 error {p95:.2f}")
        fig.tight_layout()

        out = FRAMES_OUT / f"frame_compare_{fid:03d}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print("wrote", out)

        iio.imwrite(FRAMES_OUT / f"original_{fid:03d}.png", orig)
        iio.imwrite(FRAMES_OUT / f"reconstruction_{fid:03d}.png", recon)
        iio.imwrite(FRAMES_OUT / f"diff_x4_{fid:03d}.png", diff_boost)

        frame_rows.append(
            {
                "frame": fid,
                "mean_abs_error": mae,
                "p95_error": p95,
            }
        )

    pd.DataFrame(frame_rows).to_csv(TABLES_OUT / "frame_error_summary.csv", index=False)
    return frame_rows


def save_contact_sheet():
    rows = []

    for fid in FRAME_IDS:
        orig = read_original_frame(ORIG_VIDEO, fid)
        recon = read_recon_frame(fid)

        if orig is None or recon is None:
            continue

        recon = resize_like(recon, orig)
        absdiff = np.abs(orig.astype(np.int16) - recon.astype(np.int16)).astype(np.uint8)
        diff_boost = np.clip(absdiff * 4, 0, 255).astype(np.uint8)

        w = 256
        h = int(orig.shape[0] * w / orig.shape[1])

        orig_s = cv2.resize(orig, (w, h))
        recon_s = cv2.resize(recon, (w, h))
        diff_s = cv2.resize(diff_boost, (w, h))

        row = np.concatenate([orig_s, recon_s, diff_s], axis=1)
        rows.append(row)

    if not rows:
        print("No contact sheet generated: no reconstructed frames found.")
        return

    sheet = np.concatenate(rows, axis=0)
    out = FRAMES_OUT / "contact_sheet_original_recon_diff.png"
    iio.imwrite(out, sheet)
    print("wrote", out)


def save_original_contact_sheet_if_needed():
    imgs = []
    for fid in FRAME_IDS:
        img = read_original_frame(ORIG_VIDEO, fid)
        if img is None:
            continue
        w = 256
        h = int(img.shape[0] * w / img.shape[1])
        img = cv2.resize(img, (w, h))
        cv2.putText(img, f"frame {fid}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        imgs.append(img)

    if not imgs:
        return

    rows = []
    for i in range(0, len(imgs), 2):
        pair = imgs[i:i + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows.append(np.concatenate(pair, axis=1))

    sheet = np.concatenate(rows, axis=0)
    out = FRAMES_OUT / "original_frame_contact_sheet.png"
    iio.imwrite(out, sheet)
    print("wrote", out)


def save_score_breakdown():
    pose_term, seg_term, rate_term = score_terms(FINAL["pose"], FINAL["seg"], FINAL["rate"])

    labels = ["PoseNet term", "SegNet term", "Rate term"]
    values = [pose_term, seg_term, rate_term]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Score contribution")
    ax.set_title("Final score contribution breakdown")
    ax.set_ylim(0, max(values) * 1.25)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")

    fig.tight_layout()
    out = CHARTS_OUT / "score_breakdown.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("wrote", out)

    pd.DataFrame({"term": labels, "value": values}).to_csv(TABLES_OUT / "score_breakdown.csv", index=False)


def save_experiment_visuals():
    df = pd.DataFrame(EXPERIMENTS)
    df.to_csv(TABLES_OUT / "experiments.csv", index=False)

    table_rows = []
    for e in EXPERIMENTS:
        table_rows.append(
            [
                e["name"],
                f'{e["size"]:,}' if isinstance(e["size"], int) else str(e["size"]),
                "—" if e["pose"] is None else f'{e["pose"]:.8f}',
                "—" if e["seg"] is None else f'{e["seg"]:.8f}',
                "—" if e["score"] is None else f'{e["score"]:.2f}',
                e["decision"],
            ]
        )

    columns = ["Experiment", "Archive size", "PoseNet", "SegNet", "Score", "Decision"]

    fig, ax = plt.subplots(figsize=(16, 4.7))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    ax.set_title("Experiment log: successful and rejected approaches", pad=18)
    fig.tight_layout()

    out = TABLES_OUT / "experiment_table.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("wrote", out)

    scored = [e for e in EXPERIMENTS if e["score"] is not None]
    names = [e["name"] for e in scored]
    scores = [e["score"] for e in scored]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, scores)
    ax.invert_yaxis()
    ax.set_xlabel("Final score, lower is better")
    ax.set_title("Score impact of attempted approaches")
    for i, s in enumerate(scores):
        ax.text(s, i, f" {s:.2f}", va="center")
    fig.tight_layout()

    out = CHARTS_OUT / "experiment_scores.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("wrote", out)

    pose_rows = [e for e in EXPERIMENTS if e["pose"] is not None]
    names = [e["name"] for e in pose_rows]
    pose_vals = [e["pose"] for e in pose_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, pose_vals)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("PoseNet distortion, log scale")
    ax.set_title("PoseNet sensitivity across experiments")
    fig.tight_layout()

    out = CHARTS_OUT / "posenet_sensitivity.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("wrote", out)


def save_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    boxes = [
        ("Semantic mask\nstream", 0.10),
        ("Pose / latent\nside channel", 0.30),
        ("Neural frame\ngenerator", 0.52),
        ("Tile correction\nactions", 0.74),
        ("Reconstructed\nvideo", 0.92),
    ]

    for text, x in boxes:
        ax.text(
            x,
            0.5,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black"),
        )

    for x1, x2 in [(0.18, 0.23), (0.38, 0.45), (0.60, 0.68), (0.82, 0.87)]:
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5), arrowprops=dict(arrowstyle="->"))

    ax.set_title("optimization_qpose_josema reconstruction pipeline")
    fig.tight_layout()

    out = CHARTS_OUT / "pipeline_diagram.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("wrote", out)


def save_formula():
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    formula = r"$score = 100 \cdot SegNet + \sqrt{10 \cdot PoseNet} + 25 \cdot Rate$"
    ax.text(0.5, 0.55, formula, ha="center", va="center", fontsize=22)
    ax.set_title("Challenge metric")
    fig.tight_layout()

    out = CHARTS_OUT / "metric_formula.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("wrote", out)


def main():
    print("ROOT:", ROOT)
    print("Original video exists:", ORIG_VIDEO.exists())
    print("Recon dir exists:", RECON_DIR.exists())
    print("Recon image count:", len(list_recon_images()))

    frame_summary = save_frame_comparisons()
    save_contact_sheet()
    save_original_contact_sheet_if_needed()
    save_score_breakdown()
    save_experiment_visuals()
    save_pipeline_diagram()
    save_formula()

    summary = {
        "final": FINAL,
        "experiments": EXPERIMENTS,
        "frame_summary": frame_summary,
    }
    (ASSETS / "writeup_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", ASSETS / "writeup_summary.json")


if __name__ == "__main__":
    main()
