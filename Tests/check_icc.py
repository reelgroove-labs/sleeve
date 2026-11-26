"""sRGB-profiel in geleverde PNG's: profiel aanwezig, pixels en dpi ongewijzigd, niet dubbel."""
import os, sys, io, tempfile, hashlib, numpy as np, cv2
from PIL import Image, ImageCms
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
import scan_icc
d = tempfile.mkdtemp()
rng = np.random.default_rng(3)
for name, img in (('rgb', rng.integers(0, 256, (300, 200, 3), np.uint8)), ('rgba', rng.integers(0, 256, (300, 200, 4), np.uint8))):
    p = os.path.join(d, name + '.png'); cv2.imwrite(p, img)
    before = np.asarray(Image.open(p))
    assert scan_icc.tag_srgb(p) is True
    im = Image.open(p); im.load()
    assert 'sRGB' in ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(im.info['icc_profile'])))
    assert (np.asarray(im) == before).all(), 'pixels veranderd'
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    assert scan_icc.tag_srgb(p) is False and hashlib.sha256(open(p, 'rb').read()).hexdigest() == h, 'dubbel getagd'
# PIL-bestand met dpi: dpi blijft
p = os.path.join(d, 'dpi.png'); Image.new('RGB', (50, 50), (200, 10, 10)).save(p, dpi=(600, 600))
scan_icc.tag_srgb(p); assert round(Image.open(p).info['dpi'][0]) == 600
# geen PNG: niets
q = os.path.join(d, 'x.jpg'); Image.new('RGB', (10, 10)).save(q); assert scan_icc.tag_srgb(q) is False
print('kleurprofiel: sRGB erin, pixels en dpi gelijk, niet dubbel')
# kant A/B kleur gelijktrekken (scan_kleur_kanten) schrijft opnieuw weg: profiel en dpi moeten blijven (ronde 11: ging verloren)
import subprocess
yy, xx = np.ogrid[:800, :800]; r = np.hypot(xx - 400, yy - 400) / 400
paths = []
for name, col in (('a', (40, 160, 60)), ('b', (46, 170, 64))):
    img = np.zeros((800, 800, 4), np.uint8); img[..., 0], img[..., 1], img[..., 2] = col; img[..., 3] = np.where((r < 1) & (r > .03), 255, 0)
    img[r < .33, :3] = 230
    p = os.path.join(d, f'side{name}.png'); Image.fromarray(img).save(p, dpi=(600, 600)); assert scan_icc.tag_srgb(p); paths.append(p)
tool = os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools', 'scan_kleur_kanten.py')
out = subprocess.run([sys.executable, tool, *paths], capture_output=True, text=True); assert out.returncode == 0, out.stderr
assert 'gain' in out.stdout, 'testopzet: kanten niet gelijkgetrokken: ' + out.stdout
for p in paths:
    im = Image.open(p); assert 'icc_profile' in im.info, 'profiel kwijt na kleur gelijktrekken: ' + p; assert round(im.info['dpi'][0]) == 600
print('kleur kant A/B gelijktrekken: profiel en dpi blijven')
