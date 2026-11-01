"""Train TGCN model on MediaPipe-extracted keypoints from WLASL videos."""

import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.tgcn_model import GCN_muti_att
from services.keypoint_service import keypoint_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Config
_BACKEND_DIR = Path(__file__).parent.parent
VIDEOS_DIR = Path(os.environ.get("WLASL_VIDEOS_DIR", str(_BACKEND_DIR / "../wlasl/videos")))
WLASL_JSON = Path(os.environ.get("WLASL_JSON_PATH", str(_BACKEND_DIR / "../wlasl/WLASL_v0.3.json")))
OUTPUT_DIR = Path(os.environ.get("TRAIN_OUTPUT_DIR", str(_BACKEND_DIR / "trained_model")))
NUM_CLASSES = 100
NUM_SAMPLES = 50  # frames per video
NUM_NODES = 55
HIDDEN_SIZE = 64
NUM_STAGES = 10   # reduced from 20 to limit overfitting on small data
DROPOUT = 0.5     # increased from 0.3 to limit overfitting
MIN_SAMPLES_PER_CLASS = 10  # only train on classes with enough data
BATCH_SIZE = 32
EPOCHS = 200
LR = 0.001
PATIENCE = 30
SEED = 42
K_FOLDS = 5  # for the cross-validated accuracy estimate on this small dataset

# Node pairs that swap under a left-right mirror flip: (right, left) —
# shoulder, elbow, wrist, hip, eye. Nose(0)/neck(1)/mid-hip(8) sit on the
# midline and keep their index. Node layout matches convert_to_tgcn() below.
MIRROR_SWAP_PAIRS = [(2, 5), (3, 6), (4, 7), (9, 10), (53, 54)]


# ── Step 1: Extract keypoints ──────────────────────────────────────────────

def extract_keypoints_from_video(video_path: str, max_frames: int = 150) -> list[list[float]]:
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    while cap.isOpened() and len(keypoints) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp = keypoint_service._extract(rgb)
        keypoints.append(kp)
    cap.release()
    return keypoints


def convert_to_tgcn(keypoints: list[list[float]]) -> np.ndarray:
    """Convert MediaPipe 225-dim frames to TGCN (55, 100) with correct format."""
    n_frames = len(keypoints)
    arr = np.array(keypoints, dtype=np.float32).reshape(n_frames, 75, 3)

    tgcn = np.zeros((n_frames, NUM_NODES, 3), dtype=np.float32)

    # OpenPose order mapping from MediaPipe
    tgcn[:, 0] = arr[:, 0]                          # Nose
    tgcn[:, 1] = (arr[:, 11] + arr[:, 12]) / 2      # Neck
    tgcn[:, 2] = arr[:, 12]                          # RShoulder
    tgcn[:, 3] = arr[:, 14]                          # RElbow
    tgcn[:, 4] = arr[:, 16]                          # RWrist
    tgcn[:, 5] = arr[:, 11]                          # LShoulder
    tgcn[:, 6] = arr[:, 13]                          # LElbow
    tgcn[:, 7] = arr[:, 15]                          # LWrist
    tgcn[:, 8] = (arr[:, 23] + arr[:, 24]) / 2      # MidHip
    tgcn[:, 9] = arr[:, 24]                          # RHip
    tgcn[:, 10] = arr[:, 23]                         # LHip
    tgcn[:, 11:32] = arr[:, 33:54]                   # Left Hand (21)
    tgcn[:, 32:53] = arr[:, 54:75]                   # Right Hand (21)
    tgcn[:, 53] = arr[:, 5]                          # REye
    tgcn[:, 54] = arr[:, 2]                          # LEye

    # x, y only
    tgcn_xy = tgcn[:, :, :2]

    # Normalize: center on nose, scale by max abs
    for t in range(n_frames):
        nose = tgcn_xy[t, 0].copy()
        if np.any(nose != 0):
            tgcn_xy[t] -= nose
        max_val = np.abs(tgcn_xy[t]).max()
        if max_val > 1e-6:
            tgcn_xy[t] /= max_val

    # Temporal sample to NUM_SAMPLES
    if n_frames >= NUM_SAMPLES:
        indices = np.linspace(0, n_frames - 1, NUM_SAMPLES, dtype=int)
    else:
        indices = list(range(n_frames)) + [n_frames - 1] * (NUM_SAMPLES - n_frames)
        indices = np.array(indices)

    sampled = tgcn_xy[indices]  # (50, 55, 2)

    # Channels-first: [x1..x50, y1..y50] per node
    x_coords = sampled[:, :, 0].T  # (55, 50)
    y_coords = sampled[:, :, 1].T  # (55, 50)
    result = np.concatenate([x_coords, y_coords], axis=1)  # (55, 100)

    return result


def extract_all():
    """Extract keypoints from all WLASL-100 videos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OUTPUT_DIR / "keypoints_cache.npz"

    if cache_file.exists():
        logger.info("Loading cached keypoints from %s", cache_file)
        data = np.load(cache_file)
        return data["X"], data["y"], data["labels"]

    with open(WLASL_JSON) as f:
        wlasl = json.load(f)

    vid_set = {v.stem for v in VIDEOS_DIR.glob("*.mp4")}

    X_all, y_all = [], []
    labels = []
    skipped = 0

    for class_idx, entry in enumerate(wlasl[:NUM_CLASSES]):
        gloss = entry["gloss"]
        labels.append(gloss)
        count = 0

        for inst in entry.get("instances", []):
            vid_id = inst["video_id"]
            if vid_id not in vid_set:
                continue

            video_path = str(VIDEOS_DIR / f"{vid_id}.mp4")
            try:
                kps = extract_keypoints_from_video(video_path)
                if len(kps) < 5:
                    skipped += 1
                    continue

                features = convert_to_tgcn(kps)
                X_all.append(features)
                y_all.append(class_idx)
                count += 1
            except Exception as e:
                logger.warning("Failed %s: %s", vid_id, e)
                skipped += 1

        logger.info("  [%d/%d] %s: %d videos", class_idx + 1, NUM_CLASSES, gloss, count)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)

    np.savez(cache_file, X=X, y=y, labels=np.array(labels))
    logger.info("Saved %d samples (%d skipped) to %s", len(X), skipped, cache_file)

    return X, y, labels


def mirror_sample(x: np.ndarray) -> np.ndarray:
    """Mirror a (55, NUM_SAMPLES*2) TGCN feature array left-right.

    Negates all x-coordinates and swaps symmetric node pairs (including the
    two 21-landmark hand blocks) so a sign performed with the other hand
    becomes a valid new training sample with the same label.
    """
    m = x.copy()
    m[:, :NUM_SAMPLES] = -m[:, :NUM_SAMPLES]
    for a, b in MIRROR_SWAP_PAIRS:
        m[[a, b]] = m[[b, a]]
    left_hand, right_hand = m[11:32].copy(), m[32:53].copy()
    m[11:32], m[32:53] = right_hand, left_hand
    return m


def augment_with_mirror(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Double a dataset by adding a mirrored twin of every sample."""
    mirrored = np.stack([mirror_sample(x) for x in X])
    return np.concatenate([X, mirrored]), np.concatenate([y, y])


def stratified_kfold_indices(
    y: np.ndarray, k: int, rng: np.random.RandomState
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return k (train_idx, val_idx) splits, stratified by class."""
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)
        for i, idx in enumerate(cls_idx):
            folds[i % k].append(int(idx))

    all_idx = set(range(len(y)))
    splits = []
    for i in range(k):
        val_idx = np.array(sorted(folds[i]))
        train_idx = np.array(sorted(all_idx - set(folds[i])))
        splits.append((train_idx, val_idx))
    return splits


# ── Step 2: Dataset ────────────────────────────────────────────────────────

class ASLDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Random noise
            x += torch.randn_like(x) * 0.01
            # Random temporal shift
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(1, 5, (1,)).item()
                x_part = x[:, :NUM_SAMPLES]
                y_part = x[:, NUM_SAMPLES:]
                x_part = torch.roll(x_part, shift, dims=1)
                y_part = torch.roll(y_part, shift, dims=1)
                x = torch.cat([x_part, y_part], dim=1)
        return x, self.y[idx]


# ── Step 3: Train ──────────────────────────────────────────────────────────

def train_model(
    train_ds: ASLDataset,
    val_ds: ASLDataset | None,
    num_classes: int,
    epochs: int,
    device: torch.device,
    verbose: bool = True,
) -> tuple[float, int, dict]:
    """Train a fresh model. Returns (best_val_acc, best_epoch, best_state_dict).

    If val_ds is None (final full-data run), there is nothing to select a
    "best" epoch on, so it simply trains for `epochs` and returns the last
    epoch's weights.
    """
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE) if val_ds is not None else None

    model = GCN_muti_att(
        input_feature=NUM_SAMPLES * 2,
        hidden_feature=HIDDEN_SIZE,
        num_class=num_classes,
        p_dropout=DROPOUT,
        num_stage=NUM_STAGES,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = epochs
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (out.argmax(1) == yb).sum().item()
            train_total += xb.size(0)
        scheduler.step()
        train_acc = train_correct / max(train_total, 1)

        if val_loader is not None:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_correct += (out.argmax(1) == yb).sum().item()
                    val_total += xb.size(0)
            val_acc = val_correct / max(val_total, 1)
        else:
            val_acc = train_acc  # no held-out data on the final full-data run

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            logger.info(
                "  Epoch %3d/%d | loss=%.4f | train_acc=%.1f%% | val_acc=%.1f%% | lr=%.6f",
                epoch + 1, epochs, train_loss / max(train_total, 1),
                train_acc * 100, val_acc * 100, scheduler.get_last_lr()[0],
            )

        if val_loader is not None:
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                if verbose:
                    logger.info("  Early stopping at epoch %d", epoch + 1)
                break
        else:
            # No val set to select on: keep training and take the last epoch.
            best_val_acc, best_epoch = train_acc, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_val_acc, best_epoch, best_state


def train():
    logger.info("=== Starting TGCN Training on MediaPipe Keypoints ===")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Extract
    logger.info("Step 1: Extracting keypoints...")
    start = time.time()
    X, y, labels = extract_all()
    logger.info("Extraction done in %.0fs. Dataset: %d samples, %d classes",
                time.time() - start, len(X), len(set(y)))

    # Filter to classes with enough samples
    counts = Counter(y.tolist())
    valid_indices = sorted(idx for idx, cnt in counts.items() if cnt >= MIN_SAMPLES_PER_CLASS)
    old_to_new = {old: new for new, old in enumerate(valid_indices)}
    mask = np.array([i for i, yi in enumerate(y) if yi in old_to_new])
    X = X[mask]
    y = np.array([old_to_new[yi] for yi in y[mask]])
    labels = [labels[i] for i in valid_indices]
    num_classes = len(labels)
    logger.info("Filtered to %d classes with >=%d samples: %d total samples",
                num_classes, MIN_SAMPLES_PER_CLASS, len(X))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(SEED)

    # Step 2: k-fold CV for an unbiased accuracy estimate. Mirror-augment
    # only each fold's train split — a sample's mirrored twin must never
    # land in that fold's validation split, or the estimate leaks.
    logger.info("Step 2: %d-fold cross-validation...", K_FOLDS)
    splits = stratified_kfold_indices(y, K_FOLDS, rng)
    fold_accs, fold_epochs = [], []
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = augment_with_mirror(X[train_idx], y[train_idx])
        train_ds = ASLDataset(X_train, y_train, augment=True)
        val_ds = ASLDataset(X[val_idx], y[val_idx], augment=False)

        val_acc, best_epoch, _ = train_model(train_ds, val_ds, num_classes, EPOCHS, device, verbose=False)
        fold_accs.append(val_acc)
        fold_epochs.append(best_epoch)
        logger.info("  Fold %d/%d: val_acc=%.1f%% (best epoch %d)",
                    fold_i + 1, K_FOLDS, val_acc * 100, best_epoch)

    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    final_epochs = int(round(float(np.median(fold_epochs))))
    logger.info("CV estimate: %.1f%% +/- %.1f%% (median best epoch: %d)",
                mean_acc * 100, std_acc * 100, final_epochs)

    # Step 3: train the deployed model on ALL data (original + mirror) — the
    # CV loop above already gives an unbiased accuracy estimate, so nothing
    # needs to be held out from the model we actually ship.
    logger.info("Step 3: training final model on 100%% of data for %d epochs...", final_epochs)
    X_all, y_all = augment_with_mirror(X, y)
    full_ds = ASLDataset(X_all, y_all, augment=True)
    _, _, final_state = train_model(full_ds, None, num_classes, final_epochs, device, verbose=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": final_state,
        "labels": list(labels),
        "val_acc": mean_acc,
        "epoch": final_epochs,
    }, OUTPUT_DIR / "best_model.pt")

    logger.info("=== Training Complete ===")
    logger.info("CV val accuracy estimate: %.1f%% (+/- %.1f%%)", mean_acc * 100, std_acc * 100)
    logger.info("Final model trained on %d samples (incl. mirror augmentation), saved to %s",
                len(X_all), OUTPUT_DIR / "best_model.pt")

    # Save labels
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(list(labels), f)


if __name__ == "__main__":
    keypoint_service.initialize()
    train()
