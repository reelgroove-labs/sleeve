"""Wit papier op de witte klep (0.5.57), feit: de rand van wit papier op een witte achtergrond is niet te zien.

Who's Your Momma boekje, pagina 3 en 4 (songteksten, wit tot aan de rand): papier 252-254, klep ±230; de snede kwam 4-6 mm
binnen de rand en Sleeve meldde GOED. Nu: is het papier vlak binnen de snede bijna even licht als de achtergrond vlak
erbuiten, dan twijfel met de reden (opnieuw scannen met donker papier erachter). Met donker papier erachter: geen melding.
"""
import os, sys, json, tempfile, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Resources', 'Tools'))
import scan_feitenroute as FR
DPI = 150; pm = DPI / 25.4
def raw_with(background):
    W, H = int(330 * pm), int(330 * pm); img = np.full((H, W, 4), 255, np.uint8); img[:, :, :3] = background
    x0, y0 = int(15 * pm), int(15 * pm); img[y0:y0 + int(300 * pm), x0:x0 + int(300 * pm), :3] = 253     # wit papier
    p = os.path.join(tempfile.mkdtemp(), 'insert_raw.png'); cv2.imwrite(p, img)
    json.dump({'dpi': DPI, 'scans': []}, open(p + '.scans.json', 'w'))
    return p
facts = {'verdict': 'GOED', 'precise_mm': [15.0, 15.0, 300.0, 300.0]}
lid = FR.paper_on_lid(raw_with(230), facts)
dark = FR.paper_on_lid(raw_with(40), facts)
assert lid and 'white paper on the white scanner lid' in lid and 'dark paper behind it' in lid, lid
assert dark is None, dark
print('wit papier op de witte klep: twijfel met reden; met donker papier erachter: geen melding')
