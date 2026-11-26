"""Check that an interrupted line is repaired only with two agreeing scans."""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Resources/Tools'))
from scan_line_source import repair_lines


def main():
    dpi = 300
    source = np.full((300, 510, 3), 232, np.uint8)
    source[125:130, 40:470] = 122
    other = np.clip(source.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    damaged = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
    damaged[125:130, 190:277, :3] = 232
    scans = [source, other]
    alignments = [np.eye(3)[:2]] * 2
    result = damaged.copy()
    edits = repair_lines(result, scans, alignments, dpi)
    assert len(edits) == 1, edits
    assert result[127, 200:265, 0].max() < 140
    assert np.array_equal(result[:, :, 3], damaged[:, :, 3])
    assert np.array_equal(result[:100], damaged[:100])
    assert np.array_equal(result[:, :155], damaged[:, :155])

    flat_scan = np.full_like(source, 232)
    untouched = damaged.copy()
    assert not repair_lines(untouched, [source, flat_scan], alignments, dpi)
    assert np.array_equal(untouched, damaged)

    already_good = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
    assert not repair_lines(already_good, scans, alignments, dpi)
    assert np.array_equal(already_good[:, :, :3], source)


if __name__ == '__main__':
    main()
