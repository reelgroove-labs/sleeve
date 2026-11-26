"""Strak gesneden losse labelscan (Queen For A Day 2021, 300 dpi, scan ±108 x 114 mm, label 99 mm): de papierrand moet
bevestigd worden ondanks een lichte papiervezel langs de rand in de ene helft en een schaduwlijn in de andere, en ondanks
meetpunten die buiten de scan vallen (daar mag de 'buitenkleur' niet zwart worden). Een label zonder blijvende rand
(alleen een gedrukte ring) blijft geweigerd."""
import os, sys, tempfile, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
from PIL import Image
import scan_label
pm = 300 / 25.4; W, H = round(114 * pm), round(108 * pm); cx, cy = 55 * pm, 56 * pm; R = 49.6 * pm
yy, xx = np.mgrid[:H, :W].astype(np.float32); r = np.hypot(xx - cx, yy - cy); ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
rng = np.random.default_rng(3)
def make(edge=True):
    img = np.full((H, W, 3), 229, np.float32)                       # wit vinyl
    img[r < R] = (48, 44, 42)                                       # zwart label
    if edge:
        # lichte papiervezel (0,8 mm) langs de rand op vier stukken van 30 graden, tegenover elkaar (zoals Queen 2021):
        # de eerste blijvende verandering ligt daar 0,8 mm binnen de rand, elders op de rand
        m = np.zeros_like(r, bool)
        for a0 in (0, 90, 180, 270):
            m |= (ang >= a0) & (ang < a0 + 30)
        img[(r > R - 0.8 * pm) & (r < R) & m] = 100
    else:
        img[:] = 48; img[(r > R - 2 * pm) & (r < R - 1.6 * pm)] = 200   # alleen een gedrukte ring, geen rand
    img[r < 3.6 * pm] = 245
    img += rng.normal(0, 2.5, img.shape)
    p = os.path.join(tempfile.mkdtemp(), 'label.png'); Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(p, dpi=(300, 300)); return p
lab, facts = scan_label.extract(make(), 300., material_only=True)
assert lab is not None, facts
assert abs(facts['diameter_mm'] - 99.2) < .5, facts
cx_, cy_ = facts['source_label_ellipse'][:2]; assert abs(cx_ - cx) < .3 * pm and abs(cy_ - cy) < .3 * pm, facts
lab2, facts2 = scan_label.extract(make(edge=False), 300., material_only=True)
assert lab2 is None, facts2
print('strak gesneden labelscan: papierrand bevestigd (%.2f mm); alleen een gedrukte ring blijft geweigerd' % facts['diameter_mm'])
