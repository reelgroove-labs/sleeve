"""Label en vinyl bijna dezelfde kleur (Paradise And Back Again 2022 oranje, losse labelscans, 300 dpi): op vier stukken (samen een kwart) van
de omtrek is de papierrand niet in kleur te zien (de kleur verandert daar 2,5 mm te vroeg of te laat), wel in structuur (glad papier, korrelig vinyl). De structuurrand mag die
richtingen bevestigen als hij op de kleurellips ligt; zonder structuurverschil, of met een structuurrand op een andere plek,
blijft het label geweigerd."""
import os, sys, tempfile, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
from PIL import Image
import scan_label
pm = 300 / 25.4; W = H = round(130 * pm); cx = cy = 65 * pm; R = 49.6 * pm
yy, xx = np.mgrid[:H, :W].astype(np.float32); r = np.hypot(xx - cx, yy - cy); ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
def make(vinyl_grain=14., grain_from=R):
    rng = np.random.default_rng(7)
    img = np.full((H, W, 3), (205, 120, 45), np.float32)               # oranje vinyl
    late = (ang % 90 >= 20) & (ang % 90 < 31.25)                       # op vier stukken verandert de kleur 2,5 mm te laat,
    early = (ang % 90 >= 31.25) & (ang % 90 < 42.5)                    # op vier andere 2,5 mm te vroeg (samen een kwart)
    img[r < R] = (225, 150, 80)                                        # iets lichter oranje label
    img[(r >= R) & (r < R + 2.5 * pm) & late] = (225, 150, 80)        # vinyl in labelkleur tot 2,5 mm buiten de rand
    img[(r >= R - 2.5 * pm) & (r < R) & early] = (205, 120, 45)       # papier in vinylkleur vanaf 2,5 mm binnen de rand
    img[(r < R - 12 * pm) & (r > R - 14 * pm)] = (40, 40, 40)             # gedrukte ring verder naar binnen
    img[r < 3.6 * pm] = 245
    img += rng.normal(0, 1.2, img.shape)                               # glad papier
    img[r >= grain_from] += rng.normal(0, vinyl_grain, ((r >= grain_from).sum(), 1))   # korrel van het vinyl
    p = os.path.join(tempfile.mkdtemp(), 'label.png'); Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(p, dpi=(300, 300)); return p
lab, facts = scan_label.extract(make(), 300., material_only=True)
assert lab is not None, facts
assert abs(facts['diameter_mm'] - 99.2) < .5, facts
cx_, cy_ = facts['source_label_ellipse'][:2]; assert abs(cx_ - cx) < .3 * pm and abs(cy_ - cy) < .3 * pm, facts
lab2, facts2 = scan_label.extract(make(vinyl_grain=1.2), 300., material_only=True)
assert lab2 is None, facts2
lab3, facts3 = scan_label.extract(make(grain_from=R + 2.5 * pm), 300., material_only=True)
assert lab3 is None, facts3
print('label in vinylkleur: papierrand bevestigd met structuur (%.2f mm); zonder structuurverschil of met een structuurrand '
      'naast de kleurrand blijft hij geweigerd' % facts['diameter_mm'])
