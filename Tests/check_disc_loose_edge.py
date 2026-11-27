"""Plaatmaat (0.5.54, 0.5.57), feit: kant A en B van één plaat zijn even groot.

US groen (Urban Solitude 2023 moss green) kreeg kant A 295,1 en kant B 293,4 mm, terwijl de plaat op beide kanten
300,5 mm gemeten werd. De snede (binnenkant van de afgeronde rand) is de kleinste afstand over de richtingen; op kant B
lag de buitenrand in drie richtingen (340-350 graden) 1,1-1,3 mm naar binnen (los van de cirkel), en die drie richtingen
bepaalden de snede. Richtingen waarvan de buitenrand niet op de cirkel ligt, tellen voor de snede niet mee.
Ook de binnenkant van de afgeronde rand is geen vaste plek: aan de lichtkant glanst de rand vanaf ±2 mm binnen de
buitenrand, aan de schaduwkant pas vanaf ±1 mm; de oude zoeker vond per richting de buitenrand of die glans, en hoeveel
richtingen glans gaven bepaalde de maat. De snede ligt daarom op een vaste afstand binnen de gemeten buitencirkel.
Synthetisch: dezelfde plaat met en zonder zo'n deuk van 1,3 mm in 25 graden, en met glans op de rand in 10 of in 150
graden van de omtrek; de snede moet steeds gelijk blijven.
"""
import os, sys, json, math, subprocess, tempfile, numpy as np, cv2
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Resources', 'Tools')
DPI = 200; pm = DPI / 25.4; R = 150.2 * pm; N = int(320 * pm)


def disc(notch, shine=0, shadow=False):
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32); c = N / 2
    d = np.hypot(xx - c, yy - c); ang = (np.degrees(np.arctan2(yy - c, xx - c)) + 360) % 360
    edge = np.full_like(d, R)
    if notch:
        edge[(ang >= 330) & (ang < 355)] = R - 1.3 * pm          # buitenrand hier 1,3 mm naar binnen
    img = np.full((N, N, 3), 235, np.float32)                   # witte klep
    inside = d < edge
    img[inside] = 40                                            # vlak vinyl
    rim = inside & (d > edge - 0.6 * pm)                        # smalle glans op de afgeronde rand
    img[rim] = 70
    if shadow:                                                  # schaduw in de afgeronde rand, 1,6 tot 0,6 mm binnen de rand
        img[inside & (d > edge - 1.6 * pm) & (d <= edge - 0.6 * pm)] = 22
    if shine:                                                   # lichtkant: glans van 1,6 tot 0,6 mm binnen de rand
        img[inside & (d > edge - 1.6 * pm) & (d <= edge - 0.6 * pm) & (ang < shine)] = 170
    img = cv2.GaussianBlur(img, (0, 0), 0.6).astype(np.uint8)
    return img


def cut(img, work, name):
    raw = os.path.join(work, name + '_raw.png'); out = os.path.join(work, name + '_crop.png')
    cv2.imwrite(raw, np.dstack([img, np.full(img.shape[:2], 255, np.uint8)]))
    json.dump({'scale': 1.0, 'dpi': DPI, 'scans': [{'scan': 's.tif', 'corners': [[-60, -60], [N + 60, -60], [N + 60, N + 60], [-60, N + 60]]}]},
              open(raw + '.scans.json', 'w'))
    r = subprocess.run([sys.executable, os.path.join(HERE, 'scan_feiten.py'), raw, 'sidea', '--format', 'vinyl', '--out', out],
                       capture_output=True, text=True)
    f = json.load(open(out + '.feiten.json'))
    assert 'cut_diameter_mm' in f, (r.stdout[-800:], r.stderr[-800:])
    return f['cut_diameter_mm']


work = tempfile.mkdtemp()
plain, dented = cut(disc(False), work, 'plain'), cut(disc(True), work, 'dented')
little, much = cut(disc(False, 10), work, 'little'), cut(disc(False, 150), work, 'much')
assert abs(plain - dented) <= 0.3, f'een deuk in 25 graden van de buitenrand veranderde de snede: {plain} -> {dented} mm'
assert abs(little - much) <= 0.3, f'glans op meer van de rand veranderde de snede: {little} (10 graden) -> {much} (150 graden) mm'
shaded = cut(disc(False, 0, True), work, 'shaded')
# 0.5.57: een schaduw in de rand die binnen de 2,75 mm blijft die er toch af gaat, verandert de snede niet
# (Together Alone 2022 geel met lichtcorrectie: A 294,8 / B 295,5 mm doordat alleen op A zo'n rand gemeten werd)
assert abs(plain - shaded) <= 0.3, f'schaduw in de rand (1 mm) veranderde de snede: {plain} -> {shaded} mm'
assert abs(plain - (2 * 150.2 - 1.5)) <= 0.4, f'snede niet 0,75 mm binnen de buitenrand: {plain} mm'
print(f'plaat: snede {plain} mm, met deuk in de buitenrand {dented} mm, glans op 10/150 graden {little}/{much} mm, schaduw in de rand {shaded} mm')
