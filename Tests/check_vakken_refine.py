"""Vakken indelen (0.5.51), twee feiten met synthetische scans op 2 px/mm:

1. Een vel waarvan één stuk bijna leeg is (Urban Solitude 2025 crystal clear, achterkant 0007) begon een nieuwe stapel;
   de scans daarna passen duidelijk op de eerste stukken. Twee opeenvolgende stapels die elk nog geen hele hoes zijn en
   samen precies één vel bedekken, met echte overlap ertussen, horen bij elkaar. Twee losse vellen zonder gedeeld
   drukwerk blijven apart.
2. Een vorm met afgeronde hoeken die ook op een andere scan van de stapel staat, is drukwerk (een tegel op de foto,
   Together Alone 2016 zwart 0002); een sticker op de krimpfolie staat alleen op zijn eigen scan (0001).
"""
import os, sys, tempfile, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
import scan_vakken as V

P = V.PXMM                                   # 2 px/mm
rng = np.random.default_rng(11)


def textured(h, w, seed):
    r = np.random.default_rng(seed)
    img = np.full((h, w), 235, np.uint8)
    for _ in range(int(h * w / 900)):
        x, y = int(r.integers(0, w)), int(r.integers(0, h))
        cv2.circle(img, (x, y), int(r.integers(2, 9)), int(r.integers(20, 200)), -1)
    return img


# 1. Eén vel van 310 x 440 mm in vier scans van het glas (216 x 297 mm), boven en onder. In de strook waar boven en
# onder overlappen staat links geen drukwerk; daardoor heeft de scan linksonder bijna geen overlap met de scans erboven,
# de scan rechtsonder wel (zoals 0007 en 0008 op de achterkant van Urban Solitude 2025 crystal clear).
H, W = int(440 * P), int(310 * P)
sheet = textured(H, W, 1)
sheet[int(143 * P):int(297 * P), :int(222 * P)] = 235        # lege strook links in de overlap
gh, gw = int(297 * P), int(216 * P)
def glass(y, x):
    g = np.full((gh, gw), 250, np.uint8)
    part = sheet[y:y + gh, x:x + gw]; g[:part.shape[0], :part.shape[1]] = part
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
scans = {'s1': glass(0, 0), 's2': glass(0, W - gw), 's3': glass(H - gh, 0), 's4': glass(H - gh, W - gw)}
order = ['s1', 's2', 's3', 's4']
V.number_key = lambda p: order.index(p)
feats = {k: V.features(v) for k, v in scans.items()}
groups = V.stacks(order, feats, {k: None for k in order})
assert groups == [order], groups

# Twee losse vellen zonder gedeeld drukwerk: blijven twee stapels.
other = textured(H, W, 7)
def glass2(img, y, x):
    g = np.full((gh, gw), 250, np.uint8); part = img[y:y + gh, x:x + gw]; g[:part.shape[0], :part.shape[1]] = part
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
two = {'a1': glass2(sheet, 0, 0), 'a2': glass2(sheet, 0, W - gw), 'b1': glass2(other, 0, 0), 'b2': glass2(other, 0, W - gw)}
order2 = ['a1', 'a2', 'b1', 'b2']
V.number_key = lambda p: order2.index(p)
feats2 = {k: V.features(v) for k, v in two.items()}
groups2 = V.stacks(order2, feats2, {k: None for k in order2})
assert groups2 == [['a1', 'a2'], ['b1', 'b2']], groups2

# 2. Drukwerk of sticker: scans op 300 dpi.
pm = 300 / 25.4
cover = cv2.cvtColor(textured(int(150 * pm / 2), int(150 * pm / 2), 3), cv2.COLOR_GRAY2BGR)
cover = cv2.resize(cover, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
def tile(img, x, y, w, h, colour):
    m = np.zeros(img.shape[:2], np.uint8); r = int(5 * pm)
    cv2.rectangle(m, (x + r, y), (x + w - r, y + h), 255, -1); cv2.rectangle(m, (x, y + r), (x + w, y + h - r), 255, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    img[m > 0] = colour
    return img
work = tempfile.mkdtemp()
from PIL import Image
def save(img, name):
    p = os.path.join(work, name); Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(p, dpi=(300, 300)); return p
printed = tile(cover.copy(), int(40 * pm), int(40 * pm), int(40 * pm), int(35 * pm), (60, 140, 210))   # tegel in het drukwerk
a = save(printed, 'a.png'); b = save(printed[int(10 * pm):, int(5 * pm):].copy(), 'b.png')
stick = tile(cover.copy(), int(60 * pm), int(30 * pm), int(33 * pm), int(80 * pm), (25, 25, 25))         # sticker op de folie
cv2.putText(stick, 'MUSIC ON VINYL', (int(64 * pm), int(70 * pm)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
c = save(stick, 'c.png'); d = save(cover.copy(), 'd.png')
assert V.shape_in_print(a, [b], work), 'tegel in het drukwerk niet herkend'
assert not V.shape_in_print(c, [d], work), 'sticker op de folie als drukwerk gezien'
print('vakken: bijna leeg stuk van een vel blijft bij zijn vel, losse vellen blijven apart; '
      'vorm in het drukwerk is geen sticker, sticker op de folie wel')
