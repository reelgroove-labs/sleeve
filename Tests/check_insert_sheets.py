"""Twee vellen met dezelfde opmaak zijn geen tweede scanronde (0.5.53). Queen For A Day 2021: tekstvel kant A en kant B
hebben dezelfde foto en hetzelfde logo, maar andere teksten; het tweede vel werd 'een tweede scanronde van hetzelfde vel'
en bleef open. Een echte tweede scanronde brengt elk stuk met drukwerk op dezelfde plek terug; twee vellen met dezelfde
opmaak delen hun tekstdeel niet. Synthetisch op 2 px/mm."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
import scan_vakken as V
P = V.PXMM

def textured(h, w, seed):
    r = np.random.default_rng(seed); img = np.full((h, w), 235, np.uint8)
    for _ in range(int(h * w / 900)):
        cv2.circle(img, (int(r.integers(0, w)), int(r.integers(0, h))), int(r.integers(2, 9)), int(r.integers(20, 200)), -1)
    return img

H, W = int(440 * P), int(310 * P); gh, gw = int(297 * P), int(216 * P)      # een vel in vier scans: boven en onder
def sheet(text_seed):
    s = textured(H, W, text_seed)                       # tekst: per vel anders
    s[:, :int(80 * P)] = textured(H, int(80 * P), 1)    # foto en logo links: op beide vellen gelijk
    return s
def glass(s, y, x, noise=0):
    g = np.full((gh, gw), 250, np.uint8); part = s[y:y + gh, x:x + gw]; g[:part.shape[0], :part.shape[1]] = part
    if noise:
        g = np.clip(g.astype(np.int16) + np.random.default_rng(noise).integers(-3, 4, g.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
POS = {'1': (0, 0), '2': (0, W - gw), '3': (H - gh, 0), '4': (H - gh, W - gw)}

def note_for(first_sheet, second_sheet, noise):
    scans = {f'a{k}': glass(first_sheet, *v) for k, v in POS.items()}
    scans.update({f'b{k}': glass(second_sheet, *v, noise) for k, v in POS.items()})
    feats = {k: V.features(v) for k, v in scans.items()}
    V.scan_dpi.size_px = lambda path: (1000, 1000)
    notes = {}
    V.related_paper_groups([['a1', 'a2', 'a3', 'a4'], ['b1', 'b2', 'b3', 'b4']], feats, {}, notes)
    return notes.get('b1', '')

A, B = sheet(10), sheet(20)
assert 'same position' in note_for(A, A, 5), 'echte tweede scanronde niet herkend'
assert 'same position' not in note_for(A, B, 5), 'ander vel met dezelfde opmaak als tweede scanronde gezien'
print('tweede scanronde van hetzelfde vel herkend; ander vel met dezelfde opmaak blijft een eigen vel')
