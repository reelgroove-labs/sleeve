"""Losse labelscan tijdens de koude start van het nauwkeurige lezen (For Bitter Or Worse 2009, Scan 16): de eerste
standen breken af op de tijdgrens en lezen niets. Sleeve warmt het lezen dan op en leest opnieuw; blijft de kant
ongelezen terwijl de scan al bij een kant ligt, dan staat dat in de uitleg van dat vak."""
import os, sys, subprocess, tempfile, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
from PIL import Image
import scan_vakken as V
d = tempfile.mkdtemp(); p = os.path.join(d, 'label.png')
img = np.full((1300, 1300, 3), 30, np.uint8); cv2.circle(img, (650, 650), 580, (90, 90, 90), -1)
Image.fromarray(img).save(p, dpi=(300, 300))
circle = (55.0, 55.0, 49.0)
real_run = subprocess.run
calls = {'boxes': 0, 'warm': 0}
def fake(cmd, *a, **k):
    if cmd and cmd[0] == V.OCR:
        if k.get('timeout', 0) >= 180:
            calls['warm'] += 1; return subprocess.CompletedProcess(cmd, 0, '1.0\tSIDE A\t0,0;1,0;1,1;0,1\n', '')
        calls['boxes'] += 1
        if calls['boxes'] <= 8:
            raise subprocess.TimeoutExpired(cmd, k.get('timeout'))
        return subprocess.CompletedProcess(cmd, 0, '1.0\tSIDE A\t0.1,0.9;0.5,0.9;0.5,0.8;0.1,0.8\n', '')
    return real_run(cmd, *a, **k)
V.subprocess.run = fake
got = V.label_side(p, circle)
assert got == 'sidea', (got, calls)
assert calls['warm'] == 1 and calls['boxes'] == 16, calls
# tweede labelscan in dezelfde run: niet opnieuw opwarmen
calls.update(boxes=8, warm=0); V._WARMED = True
def always_timeout(cmd, *a, **k):
    if cmd and cmd[0] == V.OCR: raise subprocess.TimeoutExpired(cmd, k.get('timeout'))
    return real_run(cmd, *a, **k)
V.subprocess.run = always_timeout
assert V.label_side(p, circle) is None
V.subprocess.run = real_run
# ongelezen kant, al in een kantvak: blijft liggen, met een melding
V.label_side = lambda s, c: None
named = [['sidea', ['/x/Scan 8.jpeg'], 'rim'], ['sideb', ['/x/Scan 12.jpeg', '/x/Scan 16.jpeg'], 'rim']]
out = V.place_label_scans(named, {'/x/Scan 16.jpeg': circle}, {})
b = next(x for x in out if x[0] == 'sideb')
assert '/x/Scan 16.jpeg' in b[1] and 'Scan 16.jpeg is a separate scan of a label whose side could not be read' in b[2], b
print('labelkant: na koude start opgewarmd en opnieuw gelezen (SIDE A); ongelezen kant wordt gemeld')
