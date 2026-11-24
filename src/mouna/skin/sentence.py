"""Stitch several signs' 3D skin clips into one sentence clip for the avatar (v5 Practice).

Prototyped in the lab (mouna-lab, lab/stories/sequence.py).

Each practice sign's skin (data/references/skin/<sign>.json.gz) comes from a
different signer, framed differently, at 25-30 fps. To play a sentence as one
avatar:
  1. trim each clip to its signing part (a hand raised), plus a few frames;
  2. put every signer on the same body: shoulders centred and scaled to one width;
  3. same dominant hand throughout (a left-handed clip is mirrored), and the
     hand slots ordered the same way so blends join the right hands;
  4. one face: each clip contributes only its face MOVEMENT (frame minus that
     clip's own typical face), laid onto the first clip's face shape;
  5. resample to one frame rate and blend a short transition between signs.
Returns the app's SkinData format plus `segments` = which frames are which sign.
"""
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

SKIN = Path(__file__).resolve().parents[3] / "data/references/skin"
FPS = 30.0
SW = 200.0                  # shoulder width in the output, px
CENTRE = (400.0, 420.0)     # shoulder midpoint in the output
SIZE = (800, 800)
TRANSITION_S = 0.28
PAD = 3                     # frames kept before / after the signing part


@lru_cache(maxsize=128)
def load(sign: str) -> dict | None:
    p = SKIN / f"{sign}.json.gz"
    return json.loads(gzip.open(p).read()) if p.exists() else None


def available() -> set[str]:
    return {p.name[:-8] for p in SKIN.glob("*.json.gz")}


def _arr(x):
    return None if x is None else np.asarray(x, dtype=np.float64)


def _clip(sign: str) -> dict | None:
    d = load(sign)
    if d is None:
        return None
    F = [{"h": [_arr(h) for h in f["h"]], "w": [_arr(w) for w in (f.get("w") or [None, None])],
          "p": _arr(f["p"]), "f": _arr(f["f"])} for f in d["frames"]]
    for f in F:   # always two hand slots
        f["h"] = (f["h"] + [None, None])[:2]
        f["w"] = (f["w"] + [None, None])[:2]
    poses = [f["p"] for f in F if f["p"] is not None]
    if poses:
        P = np.stack(poses)
        mid = np.median((P[:, 11, :2] + P[:, 12, :2]) / 2, axis=0)
        sw = float(np.median(np.linalg.norm(P[:, 11, :2] - P[:, 12, :2], axis=1))) or 1.0
    else:   # no body: use the face (face width ~ 0.45 shoulder widths)
        faces = [f["f"] for f in F if f["f"] is not None]
        Fc = np.stack(faces)
        fw = float(np.median(Fc[:, :, 0].max(1) - Fc[:, :, 0].min(1)))
        sw = fw / 0.45
        mid = np.median(Fc[:, 152, :2], axis=0) + np.array([0, 0.55 * sw])
    # signing part: frames where a wrist is above the belly
    def raised(f):
        return any(h is not None and h[0][1] < mid[1] + 0.85 * sw for h in f["h"])
    up = [i for i, f in enumerate(F) if raised(f)]
    if up:
        F = F[max(0, up[0] - PAD): up[-1] + PAD + 1]
    k = SW / sw

    def norm(a):
        if a is None:
            return None
        b = a.copy()
        b[:, 0] = (a[:, 0] - mid[0]) * k + CENTRE[0]
        b[:, 1] = (a[:, 1] - mid[1]) * k + CENTRE[1]
        if b.shape[1] > 2:
            b[:, 2] = a[:, 2] * k
        return b
    for f in F:
        f["h"] = [norm(h) for h in f["h"]]
        f["p"] = norm(f["p"])
        f["f"] = norm(f["f"])
    return {"sign": sign, "fps": float(d["fps"]), "frames": F}


def _mirror(c: dict) -> None:
    """Flip a clip left-right (a left-handed signer becomes right-handed)."""
    def fx(a, world=False):
        if a is None:
            return None
        b = a.copy()
        b[:, 0] = -a[:, 0] if world else 2 * CENTRE[0] - a[:, 0]
        return b
    for f in c["frames"]:
        f["h"] = [fx(h) for h in f["h"]]
        f["w"] = [fx(w, world=True) for w in f["w"]]
        f["p"] = fx(f["p"])
        f["f"] = fx(f["f"])


def _dominant_x(c: dict) -> float | None:
    """Mean x of the hand that moves most (the dominant hand), relative to the body centre."""
    best, bx = -1.0, None
    for k in range(2):
        W = [f["h"][k][0] for f in c["frames"] if f["h"][k] is not None]
        if len(W) < 2:
            continue
        W = np.stack(W)
        path = float(np.linalg.norm(np.diff(W[:, :2], axis=0), axis=1).sum())
        if path > best:
            best, bx = path, float(W[:, 0].mean() - CENTRE[0])
    return bx


def _order_slots(c: dict) -> None:
    """Slot 0 = the hand on the image's left (the signer's right) in every clip."""
    xs = []
    for k in range(2):
        v = [f["h"][k][0][0] for f in c["frames"] if f["h"][k] is not None]
        xs.append(np.mean(v) if v else (CENTRE[0] if k == 0 else CENTRE[0] + 1))
    if xs[0] > xs[1]:
        for f in c["frames"]:
            f["h"] = f["h"][::-1]
            f["w"] = f["w"][::-1]


def _lerp(a, b, t):
    if a is None or b is None:
        return a if t < 0.5 else b
    return a + (b - a) * t


def _blend(fa: dict, fb: dict, t: float) -> dict:
    return {"h": [_lerp(fa["h"][k], fb["h"][k], t) for k in range(2)],
            "w": [_lerp(fa["w"][k], fb["w"][k], t) for k in range(2)],
            "p": _lerp(fa["p"], fb["p"], t), "f": _lerp(fa["f"], fb["f"], t)}


def _resample(c: dict) -> list[dict]:
    F, n = c["frames"], len(c["frames"])
    m = max(2, int(round(n / c["fps"] * FPS)))
    out = []
    for j in range(m):
        x = j * (n - 1) / (m - 1)
        i = int(np.floor(x))
        out.append(_blend(F[i], F[min(n - 1, i + 1)], x - i))
    return out


def _fill(frames: list[dict], key: str) -> None:
    """A part missing in a frame is held from the nearest frame that has it."""
    for k in (range(2) if key in ("h", "w") else [None]):
        get = (lambda f: f[key][k]) if k is not None else (lambda f: f[key])
        have = [i for i, f in enumerate(frames) if get(f) is not None]
        if not have:
            continue
        for i, f in enumerate(frames):
            if get(f) is None:
                j = min(have, key=lambda h: abs(h - i))
                if abs(j - i) <= 6:
                    if k is None:
                        f[key] = frames[j][key]
                    else:
                        f[key][k] = frames[j][key][k]


def sentence(signs: list[str]) -> dict:
    clips = [c for c in (_clip(s) for s in signs) if c is not None and c["frames"]]
    if not clips:
        return {"width": SIZE[0], "height": SIZE[1], "fps": FPS, "n_frames": 0, "frames": [], "segments": []}
    for c in clips:
        dx = _dominant_x(c)
        if dx is not None and dx > 0:      # dominant hand on the image's right = signer's left
            _mirror(c)
        _order_slots(c)
    # one face: the first clip's typical face, moved by each clip's own face motion
    base = None
    for c in clips:
        faces = [f["f"] for f in c["frames"] if f["f"] is not None]
        if not faces:
            continue
        own = np.median(np.stack(faces), axis=0)
        if base is None:
            base = own
        for f in c["frames"]:
            if f["f"] is not None:
                f["f"] = base + (f["f"] - own)
    frames, segments = [], []
    nt = int(round(TRANSITION_S * FPS))
    for c in clips:
        r = _resample(c)
        for key in ("h", "w", "p", "f"):
            _fill(r, key)
        if frames:
            a, b = frames[-1], r[0]
            frames += [_blend(a, b, (t + 1) / (nt + 1)) for t in range(nt)]
        segments.append({"sign": c["sign"], "start": len(frames), "end": len(frames) + len(r) - 1})
        frames += r
    rd = lambda a: None if a is None else np.round(a, 1).tolist()  # noqa: E731
    out = [{"h": [rd(h) for h in f["h"]], "w": [rd(w) for w in f["w"]], "p": rd(f["p"]), "f": rd(f["f"])} for f in frames]
    return {"width": SIZE[0], "height": SIZE[1], "fps": FPS, "n_frames": len(out), "frames": out, "segments": segments}


if __name__ == "__main__":
    import sys
    s = sentence(sys.argv[1:] or ["mother", "cook", "now"])
    print(s["n_frames"], "frames", s["segments"])
