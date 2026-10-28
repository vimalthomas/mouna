import sys
sys.path.insert(0, "scripts")
import mouna.data.preprocessing as prep
prep.KeypointExtractor.normalize_keypoints = lambda self, x: x  # identity — save raw
import extract_keypoints
extract_keypoints.main()
