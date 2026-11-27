"""Los label uit de kant (0.5.55), feit: een label is zijn eigen maat.

Toont geen scan het hele label, dan wordt het losse label uit de samengevoegde kant gesneden. Op die kant is het label
al 1 mm rondom vergroot (retouche van de kant), zodat het losse label 101 mm werd (For Bitter Or Worse 2021 goud A/B
101,0-101,1 mm) en een label uit een losse labelscan 99,2-99,7 mm. Het losse label uit de kant krijgt die vergroting terug.
Synthetisch: een kant (wit vinyl, 300 dpi) met een zwart label van 101,2 mm (99,2 mm + 2 mm vergroting); het losse label
moet op 99,2 mm komen. Ook: de bridge herkent de vergroting aan de uitkomst van de kant, ook bij een latere beslissing.
"""
import os, sys, json, subprocess, tempfile, numpy as np
from pathlib import Path
from PIL import Image
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Resources')
sys.path.insert(0, ROOT)
pm = 300 / 25.4; N = round(140 * pm); c = N / 2; R = 101.2 / 2 * pm
yy, xx = np.mgrid[:N, :N].astype(np.float32); r = np.hypot(xx - c, yy - c); ang = np.degrees(np.arctan2(yy - c, xx - c)) % 360
rng = np.random.default_rng(5)
img = np.full((N, N, 3), 229, np.float32)
img[r < R] = (48, 44, 42)
img[(r > R - 0.8 * pm) & (r < R) & (((ang // 30) % 3) == 0)] = 100
img[r < 3.6 * pm] = 245
img += rng.normal(0, 2.5, img.shape)
work = tempfile.mkdtemp(); side = os.path.join(work, 'x_sidea.png')
Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(side, dpi=(300, 300))
cmd = [sys.executable, os.path.join(ROOT, 'Tools', 'scan_feitenroute.py'), '--part', 'labela', '--out', os.path.join(work, 'o'),
       '--work', os.path.join(work, 'w'), '--decision', json.dumps({'rotate': 0}), '--label-enlarged-mm', '1', side]
p = subprocess.run(cmd, capture_output=True, text=True)
assert p.returncode == 0, p.stderr[-600:]
res = json.loads(p.stdout.strip().splitlines()[-1])
assert res.get('path'), res
w_mm = res['size_mm'][0]
assert abs(res['diameter_mm'] - 99.2) < .5, res
assert w_mm < 100.6, f'los label uit de kant niet op zijn eigen maat: {w_mm} mm'

import studio_bridge as B
out = Path(work) / 'o2'; out.mkdir()
(out / 'Studio-verwerking.json').write_text(json.dumps({'results': [
    {'part': 'sidea', 'path': str(out / 'k_sidea.png'), 'applied': 'x; user retouch: paper label and printing enlarged 1 mm radially (2 mm diameter); centre hole preserved'},
    {'part': 'sideb', 'path': str(out / 'k_sideb.png'), 'applied': 'label from one scan'}]}))
assert B.label_grown(B.side_result_for(out, out / 'k_sidea.png')) == 1.0
assert B.label_grown(B.side_result_for(out, out / 'k_sideb.png')) == 0.0
assert B.label_grown(B.side_result_for(out, out / 'onbekend.png')) == 0.0
print(f"los label uit de kant: {res['diameter_mm']} mm (kant 101,2 mm met 1 mm vergroting), bestand {w_mm} mm breed")
