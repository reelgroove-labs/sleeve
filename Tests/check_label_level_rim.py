"""Label rechtzetten (0.5.58), feiten: de randtekst rond een label ligt nooit recht en een woord daarin kan op een tracktitel
lijken ('ALL RIGHTS' ~ 'Alright'); en de grove stand kan meer dan 7,5 graad naast zitten. Hotel New York 2023 geel kant B
stond ±8 graden scheef. Synthetisch: een label met een tracklist die 22 graden gedraaid ligt en randtekst eromheen; na
rechtzetten moet de tracklist binnen 0,5 graad recht liggen.
"""
import os, sys, math, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Resources', 'Tools'))
import scan_rechtop as R
# 1. Echt label (fixture): Hotel New York 2023 geel kant B zoals hij 7,7 graden scheef uit 0.5.57 kwam; snel lezen (CPU).
R.accurate_is_slow = lambda: True
fix = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'hny_geel_2023_kant_b_label.jpg'))
hny = ['Anouk', 'Hotel New York', 'Girl', 'Heaven Knows', 'More Than You Deserve', 'Falling Sun', 'Lost', 'Alright', 'Help', 'Our Own Love', 'Jerusalem', 'One Word', 'I Spy', 'Fading']
a_, how_ = R.level_label(fix, 0.0, hny)
assert how_.startswith('tracklist') and abs(a_ - 7.7) <= .5, (a_, how_)
# 2. Synthetisch (nauwkeurig lezen nodig): alleen als het nauwkeurige model van macOS werkt (28 nov: macOS kon zijn
# modelcache niet aanmaken, 'mkdir failed ... errno = 1'; dan wordt dit deel overgeslagen en gemeld).
R.accurate_is_slow = lambda: False
_probe = np.full((120, 600, 3), 255, np.uint8); cv2.putText(_probe, 'Jerusalem 3:55', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
_pp = os.path.join(__import__('tempfile').mkdtemp(), 'p.png'); cv2.imwrite(_pp, _probe)
if not R.run_ocr([_pp, '--boxes'], 30).strip():
    print('label rechtgezet (echt label HNY geel B: %.1f graden); synthetisch deel overgeslagen: nauwkeurig lezen van macOS niet beschikbaar' % a_)
    sys.exit(0)
TILT = 22
n = 1240; c = n // 2; img = np.full((n, n, 3), 255, np.uint8)
cv2.circle(img, (c, c), int(n / 2 * 100 / R.LABEL_MM), (235, 235, 235), -1)
words = ['Anouk', 'Hotel New York', 'Help', 'Our Own Love', 'Jerusalem', 'One Word', 'I Spy', 'Fading', 'Alright', 'Girl']
txt = np.full((n, n, 3), 255, np.uint8)
for i, t in enumerate(['1 Help 3:24', '2 Our Own Love 3:34', '3 Jerusalem 3:55', '4 One Word 4:02', '5 I Spy 3:44', '6 Fading 4:47']):
    cv2.putText(txt, t, (c - 170, c + 40 + 48 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
M = cv2.getRotationMatrix2D((c, c), TILT, 1.0); txt = cv2.warpAffine(txt, M, (n, n), borderValue=(255, 255, 255))
img = np.minimum(img, txt)
rim = 'ALL RIGHTS RESERVED UNAUTHORIZED COPYING HIRING LENDING PUBLIC PERFORMANCE ' * 2
rr = int(n / 2 * 47 / (R.LABEL_MM / 2))
for k, ch in enumerate(rim):
    a = math.radians(200 + k * 2.2); x, y = c + rr * math.cos(a), c + rr * math.sin(a)
    tile = np.full((40, 40, 3), 255, np.uint8); cv2.putText(tile, ch, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 2)
    T = cv2.warpAffine(tile, cv2.getRotationMatrix2D((20, 20), -math.degrees(a) - 90, 1.0), (40, 40), borderValue=(255, 255, 255))
    xi, yi = int(x) - 20, int(y) - 20
    if 0 <= xi < n - 40 and 0 <= yi < n - 40: img[yi:yi + 40, xi:xi + 40] = np.minimum(img[yi:yi + 40, xi:xi + 40], T)
angle, how = R.level_label(img, 0.0, words)
err = ((angle - TILT + 180) % 360) - 180
assert how.startswith('tracklist'), how
assert abs(err) <= 0.5, f"label niet recht: hoek {angle:.1f} (moet {TILT} zijn), {how}"
print(f'label rechtgezet: {angle:.1f} graden ({how}); randtekst telt niet mee')
