"""Kleurstap: tegenlichtfoto, smalle overgang langs het label, Helderheid en Felheid."""
import os, sys, tempfile, numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
import scan_colour as C

# 1. tegenlicht: label in de foto veel donkerder dan in de scan -> geen versterking, eigen belichting
gain, note = C.label_gain(np.array([.9, .9, .88], np.float32), np.array([.02, .015, .01], np.float32))
assert np.allclose(gain, 1) and 'backlit' in note['label_exposure'], note
# gewone foto: label iets donkerder -> wel gelijktrekken
gain, note = C.label_gain(np.array([.9, .9, .88], np.float32), C.linear(np.array([.8, .8, .78], np.float32)))
assert 1 < gain.max() < C.BACKLIT_GAIN and 'label_exposure' not in note, (gain, note)

# 2. uitgesneden plaat: wit label met zwarte rand, paarsig vinyl, transparant gat en buitenkant
n = 800; yy, xx = np.mgrid[:n, :n]; r = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
img = np.zeros((n, n, 4), np.uint8)
img[r < .98] = (150, 70, 140, 255)                       # vinyl
img[r < .345] = (20, 20, 20, 255)                        # zwarte labelrand
img[r < .30] = (235, 232, 225, 255)                      # papier
img[r < .03] = 0                                         # middengat
d = tempfile.mkdtemp(); src = os.path.join(d, 'plaat.png'); Image.fromarray(img).save(src, dpi=(600, 600))
ref = os.path.join(d, 'foto.png'); Image.new('RGB', (400, 400), (230, 60, 140)).save(ref)
C.photo_target = lambda raw, reference: None             # geen labelmatch in een effen foto: ring rond het midden
outs = {}
for name, kw in (('gewoon', {}), ('lichter', {'lightness': 8}), ('feller', {'vividness': 1.2})):
    dst = os.path.join(d, name + '.png'); C.recolor(src, ref, dst, **kw)
    outs[name] = np.asarray(Image.open(dst).convert('RGBA')).astype(int)
o = outs['gewoon']
assert (o[r < .34] == img[r < .34]).all(), 'label of gat veranderd'
assert (o[:, :, 3] == img[:, :, 3]).all(), 'transparantie veranderd'
# vlak buiten het label (1% van de straal) is het vinyl al volledig de nieuwe kleur: geen bandje
near = (r > .347) & (r < .352); far = (r > .6) & (r < .7)
assert np.abs(o[near][:, :3].mean(0) - o[far][:, :3].mean(0)).max() < 6, (o[near][:, :3].mean(0), o[far][:, :3].mean(0))
import cv2
lab = lambda a: cv2.cvtColor(a[far][:, :3].astype(np.float32)[None] / 255, cv2.COLOR_RGB2LAB)[0].mean(0)
assert lab(outs['lichter'])[0] > lab(o)[0] + 5, 'helderheid werkt niet'
assert np.hypot(*lab(outs['feller'])[1:]) > np.hypot(*lab(o)[1:]) * 1.1, 'felheid werkt niet'
print('kleurpaneel: tegenlicht, overgang, helderheid en felheid goed')
