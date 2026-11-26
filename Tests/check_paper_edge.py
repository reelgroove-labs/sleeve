"""Papier langs de snede: een heel licht randje (onder 20 niveaus) is niet zichtbaar en geeft geen TWIJFEL meer, alleen een
notitie; een duidelijke strook (20 of meer) blijft TWIJFEL. Feit: boekjes HNY p3 (13), US p2/p3 (9, 14) door de gebruiker
verwaarloosbaar genoemd; een echte strook mat 40."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
import scan_feiten as SF
px_mm = 600 / 25.4
def page(depth):
    img = np.full((1800, 1800, 3), 252, np.uint8)
    rng = np.random.default_rng(1); img = np.clip(img.astype(np.int16) + rng.integers(-2, 3, img.shape), 0, 255).astype(np.uint8)
    band = round(.8 * px_mm) + 1; img[:, -band:] = 252 - depth       # rechterrand 0,8 mm donkerder
    return img
for depth, expect in ((12, 'GOED'), (30, 'TWIJFEL')):
    r = {'verdict': 'GOED', 'why': []}
    SF.report_vinyl_edge_tone(r, page(depth), px_mm, False)
    assert r['verdict'] == expect, (depth, r)
    if expect == 'GOED':
        assert r.get('paper_edge_faint') and 'not visible' in r.get('applied', ''), r
    else:
        assert any(w.startswith('paper tone near the cut') for w in r['why']), r
r = {'verdict': 'GOED', 'why': []}; SF.report_vinyl_edge_tone(r, page(0), px_mm, False)
assert r['verdict'] == 'GOED' and not r.get('paper_edge_faint'), r
print('papierrand: licht randje (12) GOED met notitie, strook (30) TWIJFEL, schoon papier niets')
