"""Eén plaat, één maat (0.5.58), feit: kant A en B van één plaat zijn even groot.

Hotel New York 2022 magenta (transparant): de buitenrand werd per richting niet overal even scherp gezien en A werd 294,6,
B 295,3 mm. Scheelt het meer dan 0,3 mm, dan krijgt de grootste kant de maat van de kleinste (zelfde midden, zelfde zachte
rand, dpi en profiel blijven). Synthetisch: twee schijven op 100 dpi, 294,6 en 295,3 mm; daarna allebei 294,6 mm.
"""
import os, sys, tempfile, numpy as np
from PIL import Image
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Resources')
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, 'Tools'))
import studio_bridge as B, scan_feiten as SF
dpi = 100.0; work = tempfile.mkdtemp()
def side(mm, name):
    n = int(round(mm * dpi / 25.4)); a = np.zeros((n, n, 4), np.uint8); a[..., 0] = 200; a[..., 3] = SF.soft_disc(n, n / 2)
    p = os.path.join(work, name); Image.fromarray(a, 'RGBA').save(p, dpi=(dpi, dpi)); return p, round(n / dpi * 25.4, 1)
pa, ma = side(294.6, 'a.png'); pb, mb = side(295.3, 'b.png')
by_part = {'sidea': [{'part': 'sidea', 'path': pa, 'size_mm': [ma, ma], 'dpi': dpi}], 'sideb': [{'part': 'sideb', 'path': pb, 'size_mm': [mb, mb], 'dpi': dpi}]}
changed = B.same_record_size(by_part)
with Image.open(pb) as im: nb = im.size[0]; db = im.info.get('dpi')
with Image.open(pa) as im: na = im.size[0]
assert [r['part'] for r in changed] == ['sideb'], changed
assert nb == na, (nb, na)
assert abs(db[0] - dpi) < .5, db
assert 'cut to the size of side A' in changed[0]['applied'], changed[0]['applied']
by_part2 = {'sidea': [{'part': 'sidea', 'path': pa, 'size_mm': [ma, ma], 'dpi': dpi}], 'sideb': [{'part': 'sideb', 'path': pb, 'size_mm': [ma + .2, ma + .2], 'dpi': dpi}]}
assert B.same_record_size(by_part2) == []                       # binnen 0,3 mm: niets doen
print(f'één plaat, één maat: B {mb} -> {changed[0]["size_mm"][0]} mm (gelijk aan A {ma} mm)')
