"""Parameters as vectors from the canonical points, compared with the cloud of
fluent signers -- no hand-made features or thresholds.

Handshape vector (degrees, 20 values), measured in 3D in true proportions
(points are fractions of width / height, so x, y, z are scaled back by the
clip's size first), median over the signing phase:
  - each finger's bend at its three joints (knuckle, middle, end): 15
  - thumb opening from the hand (thumb base direction vs the index bone): 1
  - spread between neighbouring fingers (index-middle, middle-ring,
    ring-little), plus thumb-index: 4
Bend 0 = straight; spread 0 = fingers together.

Scoring (unusualness): the learner's mean distance to the k nearest fluent
signers of the sign, compared with each fluent signer's own distance to the
others (leave one out). The share of fluent signers who are closer is the
"unusualness" (0 = typical, 1 = further than every fluent signer). Each
dimension is scaled by its spread across all signs so no joint dominates.
"""
import numpy as np

CHAINS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "little": (17, 18, 19, 20)}
JOINTS = ("knuckle", "middle joint", "end joint")
HS_NAMES = ([f"{f} {j}" for f in CHAINS for j in JOINTS] + ["thumb opening"]
            + ["spread thumb-index", "spread index-middle", "spread middle-ring", "spread ring-little"])
R = slice(54, 75)


def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    c = np.sum(u * v, axis=-1) / np.maximum(nu * nv, 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def signing_phase(present: np.ndarray) -> np.ndarray:
    """Middle of the time a hand is up (raise and drop trimmed)."""
    up = np.where(present)[0]
    if len(up) < 3:
        return present
    lo, hi = up[0], up[-1]
    t = int(round((hi - lo) * 0.2))
    core = present.copy()
    core[: lo + t] = False
    core[hi - t + 1:] = False
    return core if core.sum() >= 3 else present


def handshape_vector(seq, width: int, height: int, use_depth: bool = True) -> np.ndarray | None:
    """20 angles (degrees) for the dominant (right-slot) hand, or None."""
    a = np.asarray(seq, dtype=np.float32).reshape(len(seq), 75, 3)
    h = a[:, R].astype(np.float64)
    present = np.any(h[:, :, :2] != 0, axis=(1, 2))
    core = signing_phase(present)
    if core.sum() < 3:
        return None
    h = h[core] * np.array([width, height, width if use_depth else 0.0])   # true proportions (MediaPipe z uses width's scale)
    out = []
    for f, (b, m, d, t) in CHAINS.items():
        base = h[:, 0] if f != "thumb" else h[:, 0]
        out.append(_angle(h[:, b] - base, h[:, m] - h[:, b]))   # knuckle bend
        out.append(_angle(h[:, m] - h[:, b], h[:, d] - h[:, m]))  # middle joint
        out.append(_angle(h[:, d] - h[:, m], h[:, t] - h[:, d]))  # end joint
    out.append(_angle(h[:, 2] - h[:, 1], h[:, 5] - h[:, 0]))      # thumb opening vs the index bone
    tips = {"thumb": (2, 4), "index": (5, 8), "middle": (9, 12), "ring": (13, 16), "little": (17, 20)}
    order = ["thumb", "index", "middle", "ring", "little"]
    for x, y in zip(order, order[1:]):
        (bx, tx), (by, ty) = tips[x], tips[y]
        out.append(_angle(h[:, tx] - h[:, bx], h[:, ty] - h[:, by]))
    v = np.nanmedian(np.stack(out, axis=1), axis=0)
    return v if np.all(np.isfinite(v)) else None


class Cloud:
    """The fluent signers of one sign, for one parameter."""

    def __init__(self, vectors: list[np.ndarray], scale: np.ndarray, k: int = 3, own_weight: float = 0.0):
        # own_weight > 0 blends in this sign's own spread per dimension: joints
        # fluent signers agree on (the ones that define the sign) count more
        V = np.stack(vectors)
        if own_weight > 0:
            own = np.maximum(np.std(V, axis=0), 1.0)
            scale = np.sqrt((1 - own_weight) * scale ** 2 + own_weight * own ** 2)
        self.X = V / scale
        self.scale, self.k = scale, min(k, len(vectors) - 1)
        d = np.linalg.norm(self.X[:, None] - self.X[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        self.null = np.sort(np.sort(d, axis=1)[:, : self.k].mean(1))    # fluent signers' own distances
        self.median = np.median(np.stack(vectors), axis=0)
        self.p10 = np.percentile(np.stack(vectors), 10, axis=0)
        self.p90 = np.percentile(np.stack(vectors), 90, axis=0)

    def distance(self, v: np.ndarray) -> float:
        d = np.linalg.norm(self.X - v / self.scale, axis=1)
        return float(np.sort(d)[: self.k].mean())

    def unusualness(self, v: np.ndarray) -> float:
        """Share of fluent signers who are closer to the others than the learner is."""
        return float(np.searchsorted(self.null, self.distance(v), side="right") / len(self.null))

    def evidence(self, v: np.ndarray, names: list[str], top: int = 2) -> list[dict]:
        """The dimensions where the learner is furthest outside the fluent range."""
        spread = np.maximum((self.p90 - self.p10) / 2, self.scale * 0.25)
        out = []
        for i, n in enumerate(names):
            lo, hi = self.p10[i], self.p90[i]
            dev = (lo - v[i]) / spread[i] if v[i] < lo else (v[i] - hi) / spread[i] if v[i] > hi else 0.0
            if dev > 0:
                out.append({"dim": n, "yours": round(float(v[i]), 1), "fluent": [round(float(lo), 1), round(float(hi), 1)], "outside": round(float(dev), 2)})
        return sorted(out, key=lambda e: -e["outside"])[:top]


# ---------------------------------------------------------------------------
# Palm orientation and location as vectors (same cloud scoring as handshape).

PALM_NAMES = ["palm faces sideways", "palm faces up/down", "palm faces you/away",
              "fingers point sideways", "fingers point up/down", "fingers point you/away"]
LOC_NAMES = ["hand side to side", "hand height", "fingertip side to side", "fingertip height",
             "wrist side to side", "wrist height", "fingertip closest to nose", "fingertip closest to mouth",
             "hand closest to chest"]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def palm_vector(seq, width: int, height: int) -> np.ndarray | None:
    """Palm normal and finger direction (unit vectors, 3D, true proportions), median over the signing phase."""
    a = np.asarray(seq, dtype=np.float32).reshape(len(seq), 75, 3)
    h = a[:, R].astype(np.float64)
    present = np.any(h[:, :, :2] != 0, axis=(1, 2))
    core = signing_phase(present)
    if core.sum() < 3:
        return None
    h = h[core] * np.array([width, height, width])
    normal = _unit(np.cross(h[:, 5] - h[:, 0], h[:, 17] - h[:, 0]))
    point = _unit(h[:, 9] - h[:, 0])
    v = np.r_[_unit(np.median(normal, axis=0)), _unit(np.median(point, axis=0))]
    return v if np.all(np.isfinite(v)) else None


def location_vector(seq, width: int, height: int) -> np.ndarray | None:
    """Where the dominant hand is, relative to the face and body, in shoulder widths (true proportions)."""
    a = np.asarray(seq, dtype=np.float64).reshape(len(seq), 75, 3)[:, :, :2] * np.array([width, height])
    h = a[:, R]
    present = np.any(h != 0, axis=(1, 2))
    body = np.all(a[:, [0, 9, 10, 11, 12]] != 0, axis=(1, 2))
    core = signing_phase(present) & body
    if core.sum() < 3:
        return None
    a, h = a[core], h[core]
    sw = float(np.median(np.linalg.norm(a[:, 11] - a[:, 12], axis=1)))
    if sw <= 1e-6:
        return None
    nose, mouth, chest = a[:, 0], a[:, [9, 10]].mean(1), a[:, [11, 12]].mean(1)
    centre, tip, wrist = h[:, [0, 5, 9, 13, 17]].mean(1), h[:, 8], h[:, 0]
    rel = lambda p: np.median((p - nose) / sw, axis=0)  # noqa: E731
    close = lambda p, q: float(np.min(np.linalg.norm(p - q, axis=1)) / sw)  # noqa: E731
    v = np.r_[rel(centre), rel(tip), rel(wrist), close(tip, nose), close(tip, mouth), close(centre, chest)]
    return v if np.all(np.isfinite(v)) else None
