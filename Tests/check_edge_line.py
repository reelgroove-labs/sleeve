"""Smalle donkere lijn op de snede van een hoes: hooguit 0,5 mm weg; drukwerk tot de rand en een brede gedrukte rand blijven."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
import scan_feiten as SF

px_mm = 600 / 25.4
cap = round(SF.EDGE_LINE_MAX_MM * px_mm)


def sheet():
    img = np.full((2400, 2000, 3), 236, np.uint8)
    rng = np.random.default_rng(1)
    return np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)


# 1. golvende donkere lijn van 6-10 px langs de hele rechterzijde (HNY rood back): weg, alleen rechts, hooguit 0,5 mm
img = sheet()
for y in range(img.shape[0]):
    wdt = 6 + int(2 + 2 * np.sin(y / 90))
    img[y, -wdt:] = 25
out, cut = SF.trim_dark_edge_lines(img, px_mm)
assert cut[0] == cut[2] == cut[3] == 0, cut
assert 9 <= cut[1] <= cap, cut
assert out[:, -3:].min() > 150, 'lijn niet weg'

# 2. lijn van 15 px (breder dan 0,5 mm): niet meer dan 0,5 mm snijden, wel hoogstens cap
img = sheet(); img[:, -15:] = 25
_, cut = SF.trim_dark_edge_lines(img, px_mm)
assert cut[1] == 0, cut                                  # mediaan breder dan cap: gedrukte rand, blijft

# 3. zwart drukwerk tot de rand over 8% van de zijde (HNY front, balk bij ANOUK): blijft heel
img = sheet(); img[100:300, -400:] = 20
_, cut = SF.trim_dark_edge_lines(img, px_mm)
assert cut == (0, 0, 0, 0), cut

# 4. brede gedrukte zwarte kaderrand van 3 mm rondom: blijft heel
img = sheet(); k = round(3 * px_mm)
img[:k] = img[-k:] = 20; img[:, :k] = img[:, -k:] = 20
_, cut = SF.trim_dark_edge_lines(img, px_mm)
assert cut == (0, 0, 0, 0), cut

# 5. schone hoes: niets
_, cut = SF.trim_dark_edge_lines(sheet(), px_mm)
assert cut == (0, 0, 0, 0), cut

print('randlijn: 5 gevallen goed')
