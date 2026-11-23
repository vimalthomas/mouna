#!/usr/bin/env python3
"""After build_sign_references_v2.py rewrites the reference file, put back
what a rebuild can't produce, from the copy saved just before it:
  - signs the rebuild dropped (e.g. auto-screened `how`, no human approvals)
  - face data (`representative_face_signal`) attached by
    auto_screen_references.py, carried to signs that lost it
Usage: merge_reference_extras.py <saved_copy.pkl>
"""
import pickle
import sys
from pathlib import Path

REFS = Path(__file__).resolve().parents[1] / "data/references/sign_references.pkl"


def main() -> None:
    old = pickle.load(open(sys.argv[1], "rb"))
    new = pickle.load(open(REFS, "rb"))
    kept, face = [], []
    for w, e in old.items():
        if w not in new:
            new[w] = e
            kept.append(w)
        elif e.get("representative_face_signal") is not None and new[w].get("representative_face_signal") is None:
            new[w]["representative_face_signal"] = e["representative_face_signal"]
            new[w]["face_signal_source"] = e.get("face_signal_source", "carried_over")
            face.append(w)
    pickle.dump(new, open(REFS, "wb"))
    print(f"{len(new)} references; kept from before: {kept}; face data carried: {face}")


if __name__ == "__main__":
    main()
