"""Melding bij alleen snel lezen (0.5.60), feit: werkt het nauwkeurige tekstmodel van macOS niet (28 nov 2026: de modelcache kon
niet worden aangemaakt), dan leest Sleeve snel (CPU). Stand en kant zijn dan minder zeker; dat staat in de uitkomst
(ocr_fast_only + tekst in 'applied'). Gelezen met nauwkeurig model: geen melding."""
import os, sys, tempfile, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Resources', 'Tools'))
import scan_rechtop as R, scan_feitenroute as FR
img = np.full((120, 700, 3), 255, np.uint8); cv2.putText(img, 'Jerusalem 3:55', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
R.FAST_READS = 0; R.accurate_is_slow = lambda: True
lines = R.ocr_lines(img)
assert lines and R.FAST_READS >= 1, (lines, R.FAST_READS)
res = {'applied': 'x'}; FR.note_fast_reading(res)
assert res.get('ocr_fast_only') and 'fast mode only' in res['applied'], res
R.FAST_READS = 0; res2 = {}; FR.note_fast_reading(res2)
assert 'ocr_fast_only' not in res2, res2
print('alleen snel gelezen: gemeld in de uitkomst; anders geen melding')
