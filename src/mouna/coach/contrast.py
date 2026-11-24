"""What exactly separates two signs, in plain words, from ASL-LEX 2.0.

Used when an attempt looks more like another sign (the recognizer's guess):
instead of "needs work", the coach can say what the target does differently,
e.g. PAPER vs MEDICINE -> "In PAPER all fingers stay straight (in MEDICINE
the middle finger bends forward)". Everything comes from ASL-LEX's expert
phonological coding -- nothing is invented.
"""
import csv
import re
from functools import lru_cache
from pathlib import Path

LEX = Path(__file__).resolve().parents[3] / "data/external/asl_lex/signdata.csv"

HANDSHAPE = {
    "1": "the index finger pointing", "bent_1": "the index finger bent", "flat_1": "the index finger flat",
    "5": "all fingers spread open", "curved_5": "fingers spread and curved", "flatspread_5": "fingers spread flat",
    "open_b": "a flat open hand, thumb out", "flat_b": "a flat hand, fingers together", "closed_b": "a flat hand, thumb tucked",
    "a": "a fist with the thumb at the side", "s": "a fist with the thumb across", "e": "fingertips curled down to the thumb",
    "c": "a curved C shape", "o": "fingers and thumb touching in an O", "flat_o": "fingertips together in a flat O",
    "baby_o": "the index finger touching the thumb", "f": "thumb and index finger touching, other fingers up",
    "open_8": "the middle finger bent forward", "8": "the middle finger touching the thumb",
    "l": "thumb and index finger out in an L", "curved_l": "a curved L", "bent_l": "a bent L",
    "v": "two fingers up in a V", "bent_v": "two fingers bent in a V", "curved_v": "two fingers curved in a V", "flat_v": "two fingers flat in a V",
    "h": "two fingers together, pointing sideways", "flat_h": "two fingers together, flat", "open_h": "two fingers together, apart from the thumb",
    "y": "thumb and little finger out", "i": "the little finger up", "w": "three fingers up", "3": "thumb and two fingers up",
    "4": "four fingers up, thumb tucked", "d": "the index finger up, others touching the thumb", "g": "thumb and index finger pointing sideways",
    "p": "a P shape, pointing down", "r": "two fingers crossed", "horns": "index and little finger up",
    "spread_open_e": "fingers bent at the knuckles", "x": "the index finger hooked",
}
MOVEMENT = {
    "Straight": "moves in a straight line", "Curved": "brushes in a curved arc", "None": "stays still",
    "Circular": "moves in circles", "Z-shaped": "moves in a zig-zag", "BackAndForth": "moves back and forth",
    "X-shaped": "moves in a crossing X", "Other": "has its own movement",
}
LOCATION = {
    "Neutral": "in front of the body", "Palm": "on the other hand's palm", "HeadAway": "near the side of the head",
    "Mouth": "at the mouth", "Forehead": "at the forehead", "Chin": "at the chin", "CheekNose": "at the cheek",
    "HandAway": "near the other hand", "TorsoTop": "at the upper chest", "Eye": "near the eye", "Clavicle": "at the collarbone",
    "Shoulder": "at the shoulder", "UnderChin": "under the chin", "Neck": "at the neck", "TorsoMid": "at the middle of the chest",
    "HeadTop": "at the top of the head", "UpperLip": "at the upper lip", "FingerTip": "at the other hand's fingertips",
    "PalmBack": "on the back of the other hand", "Heel": "at the heel of the other hand", "Waist": "at the waist", "Hips": "at the hips",
}
HANDS = {"OneHanded": "one hand", "SymmetricalOrAlternating": "two hands moving together",
         "AsymmetricalSameHandshape": "two hands, one moving on the other", "AsymmetricalDifferentHandshape": "two hands, one moving on the other"}


@lru_cache(maxsize=1)
def _lex() -> dict[str, dict]:
    out = {}
    try:
        for r in csv.DictReader(open(LEX, encoding="utf-8", errors="replace")):
            out.setdefault(re.sub(r"_\d+$", "", r["EntryID"].lower()), r)
    except OSError:
        pass
    return out


def _v(row: dict, col: str) -> str | None:
    v = (row.get(col) or "").strip()
    return None if not v or v == "NA" else v


def contrast(target: str, other: str, max_points: int = 2) -> list[str]:
    """Plain sentences on what the target does that `other` doesn't, most
    important first (number of hands, handshape, movement, location, the
    other hand's shape). Empty when either sign isn't coded."""
    lex = _lex()
    a, b = lex.get(target.lower()), lex.get(other.lower())
    if not a or not b:
        return []
    T, O = target.upper(), other.upper()
    out = []
    ha, hb = _v(a, "SignType.2.0"), _v(b, "SignType.2.0")
    if ha and hb and (ha == "OneHanded") != (hb == "OneHanded"):
        out.append(f"{T} uses {HANDS.get(ha, 'two hands')} ({O} uses {HANDS.get(hb, 'one hand')}).")
    sa, sb = _v(a, "Handshape.2.0"), _v(b, "Handshape.2.0")
    if sa and sb and sa != sb and sa in HANDSHAPE and sb in HANDSHAPE:
        out.append(f"In {T} the hand shows {HANDSHAPE[sa]} (in {O}, {HANDSHAPE[sb]}).")
    ma, mb = _v(a, "Movement.2.0"), _v(b, "Movement.2.0")
    if ma and mb and ma != mb and ma in MOVEMENT and mb in MOVEMENT:
        out.append(f"In {T} the hand {MOVEMENT[ma]} ({O} {MOVEMENT[mb]}).")
    la, lb = _v(a, "MinorLocation.2.0"), _v(b, "MinorLocation.2.0")
    if la and lb and la != lb and la in LOCATION and lb in LOCATION:
        out.append(f"{T} is signed {LOCATION[la]} ({O} {LOCATION[lb]}).")
    na, nb = _v(a, "NonDominantHandshape.2.0"), _v(b, "NonDominantHandshape.2.0")
    if na and nb and na != nb and na.lower() in HANDSHAPE and nb.lower() in HANDSHAPE:
        out.append(f"In {T} the other hand shows {HANDSHAPE[na.lower()]} (in {O}, {HANDSHAPE[nb.lower()]}).")
    return out[:max_points]
