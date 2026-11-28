#!/usr/bin/env python3
"""Eén onderdeel door de feitenroute: samenvoegen, uitsnijden op feiten, rechtop,
beslissing van de gebruiker toepassen, naam geven.

Uitkomst per onderdeel: GOED, of TWIJFEL met de reden en de keuzes. Bij TWIJFEL gokt
Sleeve niet; de gebruiker kiest, en die keuze wordt per onderdeel onthouden
(.sleeve-beslissingen.json in de map met de scans) en de volgende keer toegepast.

Gebruik (vanuit studio_bridge.py):
  scan_feitenroute.py --part sidea --discogs <link> --out <map> --work <tijdelijke map> scan1 scan2 ...
  [--decision '{"crop":"square","rotate":90}']
Schrijft op stdout één JSON-regel met de uitkomst.
"""
import argparse, json, math, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PY = sys.executable
DECISIONS = '.sleeve-beslissingen.json'
VINYL_INSET_MM = 2.0
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def step(text):
    """Wat er nu gebeurt, voor de voortgang in de app (via de bridge)."""
    print('SLEEVE_STEP ' + text, file=sys.stderr, flush=True)


def key_for(scans):
    """Een onderdeel herken je aan zijn scans (namen, gesorteerd)."""
    return '|'.join(sorted(os.path.basename(s) for s in scans))


def load_decisions(folder):
    try:
        return json.load(open(os.path.join(folder, DECISIONS)))
    except (OSError, ValueError):
        return {}


def fingerprint(scans):
    """Grootte en wijzigtijd per scan: een nieuwe scan met dezelfde naam is een ander onderdeel."""
    out = {}
    for s in scans:
        try:
            st = os.stat(s); out[os.path.basename(s)] = [st.st_size, int(st.st_mtime)]
        except OSError:
            out[os.path.basename(s)] = None
    return out


def remembered(folder, scans):
    """De onthouden beslissing voor precies deze scans. Feit: een beslissing gold voor wat jij zag;
    is een scan sindsdien vervangen (zelfde naam, andere inhoud), dan geldt hij niet meer
    (0.5.29: een nieuwe scan kreeg stil 'zo laten' van de oude)."""
    got = load_decisions(folder).get(key_for(scans), {})
    if got.get('files') != fingerprint(scans):
        return {}
    return got.get('decision', {})


def save_decision(folder, scans, part, decision):
    data = load_decisions(folder)
    data[key_for(scans)] = {'part': part, 'decision': decision, 'files': fingerprint(scans)}
    try:
        json.dump(data, open(os.path.join(folder, DECISIONS), 'w'), indent=1, ensure_ascii=False)
    except OSError:
        pass                                               # alleen-lezen map: dan niet onthouden


def release_data(discogs):
    """De release van Discogs (via scan_net: één keer opgehaald, daarna uit het geheugen), of None
    zonder link. Lukt het ophalen niet, dan komt de fout omhoog; de aanroeper meldt hem."""
    import scan_naam as N
    m = re.search(r'/(release|master)/(\d+)', discogs or '')
    if not m:
        return None
    r = N.get(f'https://api.discogs.com/{"masters" if m[1] == "master" else "releases"}/{m[2]}')
    if m[1] == 'master':
        r = N.get(f'https://api.discogs.com/releases/{r["main_release"]}')
    return r


def release_words(discogs, disc=False):
    """Bekende tekst van de release: artiest, titel, tracks, labels (feit voor rechtop)."""
    r = release_data(discogs)
    if not r:
        return []
    if disc:
        # Keep the two heading fields separate from actual track titles.
        # Publisher/logo text is often curved or independently rotated.
        return [' / '.join(a['name'] for a in r.get('artists', [])), r.get('title', '')] + [
            t['title'] for t in r.get('tracklist', []) if t.get('title')]
    return ([a['name'] for a in r.get('artists', [])] + [r.get('title', '')] +
            [t['title'] for t in r.get('tracklist', []) if t.get('title')] + [l['name'] for l in r.get('labels', [])])


def release_format(discogs):
    """Feit (Discogs): cd of vinyl."""
    r = release_data(discogs)
    if not r:
        return None
    names = {(f.get('name') or '').lower() for f in r.get('formats', [])}
    return 'cd' if names & {'cd', 'cdr'} else ('vinyl' if 'vinyl' in names else None)


def scan_resolution(path, release_dpi=0.0):
    """Feit: de resolutie uit het bestand (EXIF/JFIF, zie scan_dpi); staat die er niet
    betrouwbaar in, dan die van de andere scans van deze release (zelfde scanner)."""
    import scan_dpi
    return scan_dpi.read(path) or (release_dpi if release_dpi and 50 < release_dpi < 5000 else 0.0)


def already_cut(path, spec, release_dpi=0.0, inset_mm=0.0):
    """Is dit beeld al uitgesneden? Het zichtbare deel (niet-transparant) heeft dan al de
    maat die de feiten voor dit onderdeel geven. Geeft (beeld, notitie) of None."""
    dpi = scan_resolution(path, release_dpi)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cut_out = False
    if img.shape[2] == 4:
        ys, xs = np.nonzero(img[:, :, 3] > 0)
        if not len(xs):
            return None
        cut_out = (img[:, :, 3] == 0).any()                # transparant eromheen: al uitgesneden
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    inferred = False
    if not dpi and cut_out:
        # Geen resolutie in het bestand, maar wel al uitgesneden: de bekende maat van het
        # onderdeel (feit) geeft de resolutie, als de verhouding past.
        h_px, w_px = img.shape[:2]
        if spec[0] == 'disc':
            if abs(w_px - h_px) <= .015 * w_px:
                dpi = w_px / (sum(spec[1]) / 2 / 25.4); inferred = True
        else:
            (wl, wh), (hl, hh) = spec[1], spec[2]
            for W_, H_ in ((w_px, h_px), (h_px, w_px)):
                lo_d = max(W_ / (wh / 25.4), H_ / (hh / 25.4)); hi_d = min(W_ / (wl / 25.4), H_ / (hl / 25.4))
                if lo_d <= hi_d:
                    dpi = (lo_d + hi_d) / 2; inferred = True; break
    if not dpi:
        return None
    w_mm, h_mm = img.shape[1] / dpi * 25.4, img.shape[0] / dpi * 25.4
    if spec[0] == 'disc':
        ok = spec[1][0] - 1 - (2 * inset_mm if cut_out else 0) <= w_mm <= spec[1][1] + 1 and abs(w_mm - h_mm) < 1.5
    else:
        (wl, wh), (hl, hh) = spec[1], spec[2]
        ok = (wl <= w_mm <= wh and hl <= h_mm <= hh) or (wl <= h_mm <= wh and hl <= w_mm <= hh)
    return (img, {'mm': [round(w_mm, 1), round(h_mm, 1)], 'dpi': dpi, 'dpi_from_size': inferred, 'transparent': bool(cut_out)}) if ok else None


def inset_vinyl(img, dpi, diameter_mm):
    """Apply a radial cut at native scale to the already-centred disc."""
    import scan_feiten as SF
    n = img.shape[0]
    if img.shape != (n, n, 4) or not diameter_mm or dpi <= 0:
        return None
    # Circle coordinates in the centred output follow soft_disc's n / 2.
    radius = (diameter_mm / 2 - VINYL_INSET_MM) * dpi / 25.4
    if radius <= 0 or abs(n - diameter_mm * dpi / 25.4) > 2:
        return None
    border = int(math.floor(n / 2 - radius))
    if border < 1:
        return None
    trimmed = img[border:n-border, border:n-border].copy()
    np.minimum(trimmed[:, :, 3], SF.soft_disc(trimmed.shape[0], radius),
               out=trimmed[:, :, 3])
    return trimmed


def face_orientation(img):
    """Stand uit gezichten (Apple Vision): een gezicht staat rechtop. Alleen als er een groot gezicht is (het
    onderwerp, minstens 15% van de breedte of hoogte), duidelijk herkend, met een kanteling binnen 20 graden van
    een kwartslag, en alle grote gezichten het eens zijn (GF boekje: alleen een gezicht en een logo, geen tekst).
    Geeft de draaiing met de klok mee (0/90/180/270) of None."""
    im = img
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[:, :, 3:4] / 255.0; im = (im[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    k = 1200 / max(im.shape[:2]); sm = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else im
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'gezicht.png'); cv2.imwrite(path, sm)
        try:
            out = subprocess.run([os.path.join(HERE, 'sleeve_ocr'), path, '--faces'], capture_output=True, text=True, timeout=60).stdout
            faces = json.loads(out or '[]')
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None
    turns = set()
    for f in faces:
        if f.get('confidence', 0) < .6 or max(f.get('w', 0), f.get('h', 0)) < .15 or abs(f.get('roll', 999)) > 360:
            continue
        q = round(f['roll'] / 90.0)
        if abs(f['roll'] - 90 * q) > 20:
            return None
        turns.add(int(90 * q) % 360)
    return turns.pop() if len(turns) == 1 else None


def discogs_orientation(img, discogs, disc=False):
    """Feit: de Discogs-foto's van deze release laten zien hoe het onderdeel staat.
    Overeenkomende punten (SIFT, draaiingsvrij) tussen ons beeld en elke foto; de
    draaiing van de best passende foto is de stand. Hoes: op een kwartslag afgerond.
    Geeft (graden met de klok mee, aantal punten) of None als niets duidelijk past."""
    import scan_discogs as SD
    try:
        _, photos = SD.release_images(discogs)
    except Exception:
        return None
    if not photos:
        return None
    base = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        a_ = img[:, :, 3:4] / 255.0
        base = (img[:, :, :3] * a_ + 255 * (1 - a_)).astype(np.uint8)
    k = 1200 / max(base.shape[:2])
    small = cv2.resize(base, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else base
    found = SD.matches(small, photos, with_coverage=True)
    # Sleeve 0.5.53 - feit: een overeenkomst met schaal 0 is ontaard (alle punten op één punt), geen stand. For Bitter
    # Or Worse 2021 goud, tekstkant van het insert: 150 'punten' op de foto van de plaat, schaal 0,0, dekking 0; het vel
    # bleef een kwartslag gedraaid staan. Zoals bij het indelen (scan_vakken.real) telt alleen een echte schaal.
    found = [f for f in found if float(np.hypot(f[2][0, 0], f[2][1, 0])) > .05]
    if not found:
        return None
    inliers, _, M, cover = found[0]
    # Een foto die maar een klein stukje dekt (bv. een klein logo op een binnenpagina) is geen
    # tegenspraak voor een foto die de hele hoes dekt (GF silver voorkant: binnenpagina met logo
    # 48 punten op 133°, hoesfoto's 58/40/34 op 90°).
    rivals = [f for f in found[1:] if not (cover >= .1 and f[3] < .25 * cover)]
    if inliers < 40 or (rivals and rivals[0][0] > .8 * inliers and abs(
            math.degrees(math.atan2(rivals[0][2][1, 0], rivals[0][2][0, 0])) - math.degrees(math.atan2(M[1, 0], M[0, 0]))) > 10):
        return None                                        # te weinig, of twee foto's die het oneens zijn
    ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))       # draaiing ons -> foto (beeldcoördinaten: met de klok mee)
    ang = ang % 360
    if not disc:
        q = round(ang / 90) * 90 % 360
        if abs(((ang - q + 180) % 360) - 180) > 12:
            return None                                    # hoes ligt niet op een kwartslag: niet vertrouwen
        ang = q
    return ang, inliers


def choices_for(why):
    """Welke keuzes horen bij een twijfel."""
    out = ['keep']
    for w in why:
        if w.startswith('not square'):
            out.append('square')
    return out


def backing_mask(L, lab, pm):
    """Feit: buiten het onderdeel ligt de klep of het vel dat erachter gelegd is: effen, en te zien
    langs de rand van de scan. Geeft de pixels met die kleur, of None als de rand van de scan niet
    effen is (dan loopt het onderdeel zelf tot de rand). L in 0-100, lab zoals OpenCV het geeft."""
    b = max(2, int(1.5 * pm))
    q = np.dstack([L, lab[:, :, 1] - 128, lab[:, :, 2] - 128])
    strip = np.concatenate([q[:b].reshape(-1, 3), q[-b:].reshape(-1, 3), q[:, :b].reshape(-1, 3), q[:, -b:].reshape(-1, 3)])
    med = np.median(strip, 0)
    if (np.linalg.norm(strip - med, axis=1) < 12).mean() < .7:
        return None
    return np.linalg.norm(q - med, axis=2) < 12, med


def print_outside(raw, facts):
    """Ligt er tussen de gevonden snede en de beeldrand (aan minstens één zijde, strook van
    minstens 0,3 mm) duidelijk drukwerk in plaats van effen lichte deksel?"""
    im = cv2.imread(raw, cv2.IMREAD_UNCHANGED)
    if im is None:
        return False
    dpi = facts.get('dpi') or 300.0
    k = min(1.0, 300.0 / dpi); pm = dpi * k / 25.4
    im = cv2.resize(im[:, :, :3], None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else im[:, :, :3]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * 100 / 255; C = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    lid = (L > 85) & (C < 10)
    back = backing_mask(L, lab, pm)                        # ook een donker of gekleurd vel achter het onderdeel is geen drukwerk
    if back is not None:
        lid |= back[0]
    x, y, w, h = (facts.get('precise_mm') or facts.get('box_mm'))[:4]
    H, W = lid.shape; m = int(.3 * pm)
    x0, y0, x1, y1 = int(x * pm), int(y * pm), int((x + w) * pm), int((y + h) * pm)
    bands = [lid[y0:y1, :max(0, x0 - 1)], lid[y0:y1, min(W, x1 + 1):], lid[:max(0, y0 - 1), x0:x1], lid[min(H, y1 + 1):, x0:x1]]
    for b in bands:
        if b.size and min(b.shape) >= m and (~b).mean() > .5:
            return True
    return False


def colour_extent(raw, dpi, cut, fmt, part):
    """Feit: alles wat gekleurd is, hoort bij het kaartje; het kaartje is nooit kleiner dan zijn
    gekleurde vlak. Steekt het gekleurde vlak duidelijk (>= 2 mm) buiten de gevonden snede en past
    het zelf op een maat van deze verpakking, dan is de snede een drukrand: snij op het gekleurde
    vlak. (Places To Go: geel kader tot vlak bij de beeldrand, oranje vlak erbinnen.)
    Wit kaartje op witte deksel: alleen de bedrukking is gekleurd en die ligt binnen de snede."""
    import scan_feiten as SF
    full = cv2.imread(raw, cv2.IMREAD_UNCHANGED)
    if full is None:
        return None
    if full.ndim == 2:
        full = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
    bgr = full[:, :, :3]
    k = min(1.0, 300.0 / dpi)
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else bgr
    pm = dpi * k / 25.4
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * 100 / 255; C = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    mask = (C > 18) | (L < 60)                                      # duidelijk gekleurd of donker
    back = backing_mask(L, lab, pm)
    if back is not None and back[1][0] < 45 and np.hypot(back[1][1], back[1][2]) < 15:
        mask &= ~back[0]                                            # feit: een donker vel achter het onderdeel is geen kaartje
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((max(3, int(pm)),) * 2, np.uint8))
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = cv2.findNonZero((lab_ == big).astype(np.uint8))
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    if ang > 45:
        ang -= 90; rw, rh = rh, rw
    wm, hm = rw / pm, rh / pm
    cw, ch = cut.shape[1] / dpi * 25.4, cut.shape[0] / dpi * 25.4
    if not (min(wm, hm) >= min(cw, ch) + 2 and max(wm, hm) >= max(cw, ch) + 2):
        return None
    for f in [fmt] + [x for x in ('cd', 'cd-slimjewel', 'cd-cardsleeve', 'cd-digipak') if x != fmt]:
        spec = SF.SHAPES[f].get(part)
        if spec and spec[0] == 'rect':
            (wl, wh), (hl, hh) = spec[1], spec[2]
            if (wl - 1 <= wm <= wh + 1 and hl - 1 <= hm <= hh + 1) or (wl - 1 <= hm <= wh + 1 and hl - 1 <= wm <= hh + 1):
                break
    else:
        return None
    s = 1 / k                                                        # terug naar volle resolutie
    R = cv2.getRotationMatrix2D((cx * s, cy * s), ang, 1.0)
    W, H = int(round(rw * s)), int(round(rh * s))
    R[0, 2] += W / 2 - cx * s; R[1, 2] += H / 2 - cy * s
    out = cv2.warpAffine(full, R, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, (wm, hm)


def square_up(img, sides):
    """Beslissing 'vierkant': de zijde met het zwakste randbewijs schuift in."""
    h, w = img.shape[:2]; k = abs(h - w)
    long_sides = ('top', 'bottom') if h > w else ('left', 'right')
    weak = min(long_sides, key=lambda s: sides[s]['cover'])
    sl = {'top': (slice(k, None), slice(None)), 'bottom': (slice(0, h - k), slice(None)),
          'left': (slice(None), slice(k, None)), 'right': (slice(None), slice(0, w - k))}[weak]
    return img[sl], weak


def side_on_label(img, dpi):
    """Kantletter zoals op het label gedrukt (SIDE A / SIDE ONE ...), of None."""
    h, w = img.shape[:2]; r = int(52 / 25.4 * dpi)
    lab = img[max(0, h // 2 - r):h // 2 + r, max(0, w // 2 - r):w // 2 + r, :3]
    s = 1600 / max(lab.shape[:2])
    if s < 1:
        lab = cv2.resize(lab, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # zelfde veilige tekstherkenning als bij rechtop zetten (nauwkeurig hooguit 8 s, anders snel;
    # vroeger 4 x 120 s wachten, en bij hangen viel de hele kant om: 'maar 1 LP gedaan')
    import scan_rechtop as R
    try:
        out = '\n'.join(l['text'] for l in R.ocr_lines(lab))
    except Exception:
        out = ''
    return side_letter(out)


SIDE_WORDS = {'one': 'a', 'two': 'b', 'three': 'c', 'four': 'd', '1': 'a', '2': 'b', '3': 'c', '4': 'd'}
SIDE_PATTERNS = (r'\b(?:side|kant|seite)\s*([abcd]|one|two|three|four|[1-4])\b',   # SIDE A, Side One, Side 1, Kant A, Seite 2
                 r'\b([abcd])-side\b')                                               # A-Side


def side_letter(text):
    """De kant zoals hij op een label gedrukt staat (a-d), of None. Alleen schrijfwijzen die echt
    een kant aanwijzen; een losse letter of een woord uit een titel telt niet."""
    for line in text.splitlines():
        for pattern in SIDE_PATTERNS:
            m = re.search(pattern, line, re.I)
            if m:
                v = m[1].lower(); return SIDE_WORDS.get(v, v)
    return None


STITCH_FILES = ('scan_stitch.py', 'scan_lichtkolom.py', 'scan_blend.py', 'scan_fields.py', 'scan_memory.py', 'scan_progress.py', 'scan_dpi.py', 'scan_edges.py', 'scan_cutout.py', 'scan_feiten.py', 'scan_line_source.py', 'scan_seam_probe.py', 'scan_paper_light.py')


def stitch_key(scans, halved, disc):
    """Sleutel van een samengevoegd beeld: de scans (naam, grootte, tijd), de opties en de samenvoeg-code."""
    import hashlib
    h = hashlib.sha1()
    for f in STITCH_FILES:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            h.update(open(p, 'rb').read())
    for s in sorted(scans):
        st = os.stat(s); h.update(f'{os.path.abspath(s)}|{st.st_size}|{int(st.st_mtime)}'.encode())
    env = '|'.join(os.environ.get(v, '') for v in ('SLEEVE_SEAM_LEVEL', 'SLEEVE_LOCAL_ALIGN', 'SLEEVE_TONE_KEY'))
    if os.environ.get('SLEEVE_LICHTKOLOM'):
        env += '|lichtkolom:' + hashlib.sha1(open(os.environ['SLEEVE_LICHTKOLOM'], 'rb').read()).hexdigest() + '|geen-kleur-v2'
    h.update(f'{halved}|{disc}|{env}'.encode())
    return h.hexdigest()


def stitch_cache_get(scans, halved, disc, raw):
    """Vaste toets: een eerder samengevoegd beeld van precies deze scans met precies deze code hergebruiken."""
    d = os.environ.get('SLEEVE_STITCH_CACHE')
    if not d:
        return None
    k = os.path.join(d, stitch_key(scans, halved, disc))
    if not all(os.path.exists(k + x) for x in ('.png', '.png.scans.json', '.log')):
        return None
    shutil.copyfile(k + '.png', raw); shutil.copyfile(k + '.png.scans.json', raw + '.scans.json')
    os.utime(k + '.png')                                   # laatst gebruikt (voor het opruimen)
    return subprocess.CompletedProcess([], 0, open(k + '.log').read(), '')


def stitch_cache_put(scans, halved, disc, raw, log):
    d = os.environ.get('SLEEVE_STITCH_CACHE')
    if not d or not os.path.exists(raw + '.scans.json'):
        return
    try:
        os.makedirs(d, exist_ok=True)
        # nooit de schijf vol: alleen bewaren als er daarna nog ruim plek is voor het samenvoegen zelf
        if shutil.disk_usage(d).free < 8e9 + os.path.getsize(raw):
            return
        k = os.path.join(d, stitch_key(scans, halved, disc))
        shutil.copyfile(raw, k + '.png'); shutil.copyfile(raw + '.scans.json', k + '.png.scans.json'); open(k + '.log', 'w').write(log)
        # binnen het budget blijven: de langst niet gebruikte eerst weg
        budget = float(os.environ.get('SLEEVE_STITCH_CACHE_GB', '3')) * 1e9
        items = sorted((os.path.getmtime(p), p) for p in (os.path.join(d, f) for f in os.listdir(d)) if p.endswith('.png') and not p.endswith('.scans.json'))
        total = sum(os.path.getsize(p) for _, p in items)
        for _, p in items:
            if total <= budget:
                break
            total -= os.path.getsize(p)
            for x in ('', '.scans.json'):
                try: os.remove(p + x)
                except OSError: pass
            try: os.remove(p[:-4] + '.log')
            except OSError: pass
    except OSError:
        pass                                               # schijf vol: zonder cache verder


def dark_sticker_upright(crop, words_path, work):
    """Sleeve Transparant: stand van een donkere sticker uit een zwart-witversie. Glad (ruis van het goud weg),
    plaatselijke drempel (tekst donkerder dan de grond eromheen). Geeft het scan_rechtop-resultaat als dat
    GOED of LEESBAAR is, anders None. Alleen de stand telt; de pixels van de sticker blijven die van de scan."""
    im = cv2.imread(crop, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    bgr = im[:, :, :3].astype(np.float32)
    if im.ndim == 3 and im.shape[2] == 4:
        a_ = im[:, :, 3:4] / 255.0
        ground = np.median(bgr[im[:, :, 3] > 200], axis=0) if (im[:, :, 3] > 200).any() else np.array([128, 128, 128])
        bgr = bgr * a_ + ground * (1 - a_)                 # buiten de ronde hoeken: de kleur van de sticker zelf
    grey = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    k = max(grey.shape) / 1880.0                           # gemeten op TA clear / GF pink (1880 px lang)
    grey = cv2.GaussianBlur(grey, (0, 0), max(1.0, 3 * k))
    block = int(75 * k) | 1
    bw = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, max(15, block), 4)
    bw = cv2.medianBlur(bw, 5)
    bw = cv2.resize(bw, None, fx=.6, fy=.6, interpolation=cv2.INTER_AREA)
    path = os.path.join(work, 'sticker_zwartwit.png')
    cv2.imwrite(path, cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))   # scan_rechtop leest drie kanalen
    ro = subprocess.run([PY, os.path.join(HERE, 'scan_rechtop.py'), path, '--words', words_path], capture_output=True, text=True)
    try:
        got = json.loads(ro.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    return got if got.get('verdict') in ('GOED', 'LEESBAAR') else None


def sticker_route(a, decision):
    """Sticker (eigen scan, de sticker op zijn plastic hoes): uitsnijden op feiten (scan_sticker.py: kader en rand,
    of cirkel), rechtzetten op de tekst (artiest en titel van de release, of andere duidelijk leesbare tekst; een
    ronde sticker op de graad, zoals een label), met transparante hoeken bewaren."""
    os.makedirs(a.work, exist_ok=True)
    crop = os.path.join(a.work, 'sticker_crop.png'); up = os.path.join(a.work, 'sticker_up.png')
    result = {'part': 'sticker', 'scans': a.scans, 'verdict': 'GOED', 'why': [], 'choices': [], 'decision': decision}
    if len(a.scans) > 1:
        result['why'].append(f'a sticker is one scan; only {os.path.basename(a.scans[0])} was used')
    dpi = scan_resolution(a.scans[0], a.release_dpi) or 0.0
    step('finding the edge of the sticker')
    r = subprocess.run([PY, os.path.join(HERE, 'scan_sticker.py'), a.scans[0], '--out', crop] + (['--dpi', str(dpi)] if dpi else []), capture_output=True, text=True)
    try:
        st = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        st = {'verdict': 'FOUT', 'why': [(r.stderr or 'sticker not found').strip().splitlines()[-1]]}
    if st.get('verdict') == 'FOUT' or not os.path.exists(crop):
        # Sleeve Transparant: een kleine scan (hooguit 140 mm) die de scanner bijsneed tot de sticker zelf: er is geen rand
        # rondom te vinden, de scan is de sticker (Hotel New York 2023, 00 18: 55 x 86 mm)
        import scan_dpi as _sd
        sz_ = _sd.size_px(a.scans[0])
        if dpi and sz_ and max(sz_) / dpi * 25.4 <= 140:
            from PIL import Image as _Im
            _Im.MAX_IMAGE_PIXELS = None
            with _Im.open(a.scans[0]) as im_:
                cv2.imwrite(crop, cv2.cvtColor(np.asarray(im_.convert('RGB')), cv2.COLOR_RGB2BGR))
            st = {'verdict': 'GOED', 'why': [], 'shape': 'rectangle'}
            result['applied'] = (f'the scanner cut this scan to the sticker itself ({sz_[0] / dpi * 25.4:.0f} x {sz_[1] / dpi * 25.4:.0f} mm, '
                                 'no edge found inside); not cropped again')
        else:
            result.update(verdict='FOUT', why=st.get('why', ['sticker not found'])); print(json.dumps(result, ensure_ascii=False)); return
    result['why'] += st.get('why', []); result['shape'] = st.get('shape')
    img = cv2.imread(crop, cv2.IMREAD_UNCHANGED)
    words_note = None
    try:
        words = release_words(a.discogs) if a.discogs else []
    except Exception as e:
        words, words_note = [], f'which way is up: the text of the release could not be fetched from Discogs ({e})'
    circle = st.get('shape') == 'circle'
    if decision.get('rotate') in ROTATE or decision.get('rotate') == 0:
        if decision['rotate']:
            img = cv2.rotate(img, ROTATE[decision['rotate']])
        result['applied'] = f'rotated {decision["rotate"]}'
    else:
        step('reading text to set it upright (macOS text recognition)')
        json.dump(words, open(os.path.join(a.work, 'words.json'), 'w'))
        args = [PY, os.path.join(HERE, 'scan_rechtop.py'), crop, '--words', os.path.join(a.work, 'words.json'), '--out', up]
        if circle:
            args += ['--disc', '--dpi', str(dpi or 300.0)]
        def read_upright(env=None):
            ro = subprocess.run(args, capture_output=True, text=True, env=env)
            try:
                return json.loads(ro.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                return {'verdict': 'TWIJFEL'}
        upright = read_upright()
        import scan_rechtop as RT
        if upright.get('verdict') not in ('GOED', 'LEESBAAR') and RT.accurate_is_slow():
            # Feit (TA zwart sticker, zwart reliëf op zwart): alleen het nauwkeurige lezen leest die tekst, het snelle niets.
            # Het nauwkeurige lezen stond uit omdat een eerdere poging de 8 s niet haalde (koude start: 98 s, daarna 0,4 s).
            # Eén keer opnieuw, met ruimte om op te starten.
            try:
                os.remove(RT.SLOW_FLAG)
            except OSError:
                pass
            step('reading text again with accurate recognition (it was starting up)')
            second = read_upright(dict(os.environ, SLEEVE_OCR_ACCURATE_WAIT='120'))
            if second.get('verdict') in ('GOED', 'LEESBAAR'):
                upright = second
        result['upright'] = {k: upright.get(k) for k in ('rotate_clockwise', 'verdict', 'read_upright')}
        dark = None
        if upright.get('verdict') not in ('GOED', 'LEESBAAR') and not circle:
            # Sleeve Transparant: donkere sticker (zwarte tekst op donker goud, TA clear / GF pink): Vision leest
            # de kleuren niet. Nog eens lezen in een zwart-witversie (glad, plaatselijke drempel); de stand
            # die daar uitkomt geldt voor de echte sticker.
            dark = dark_sticker_upright(crop, os.path.join(a.work, 'words.json'), a.work)
            if dark:
                result['upright'] = {k: dark.get(k) for k in ('rotate_clockwise', 'verdict', 'read_upright')}
        if dark:
            if dark['rotate_clockwise'] in ROTATE:
                img = cv2.rotate(img, ROTATE[dark['rotate_clockwise']])
            result['applied'] = 'upright from ' + ('known text of the release' if dark['verdict'] == 'GOED' else 'readable text') + ' (read in black and white)'
        elif upright.get('verdict') in ('GOED', 'LEESBAAR') and os.path.exists(up):
            img = cv2.imread(up, cv2.IMREAD_UNCHANGED)
            result['applied'] = 'upright from ' + ('known text of the release' if upright.get('verdict') == 'GOED' else 'readable text')
        else:
            result['why'].append(words_note or 'which way is up: no text could be read clearly')
    if result['why'] and decision.get('crop') == 'keep':
        result['applied'] = (result.get('applied', '') + ' kept as it is').strip(); result['why'] = []
    if result['why']:
        result['verdict'] = 'TWIJFEL'; result['choices'] = ['keep'] + (['rotate90', 'rotate180', 'rotate270'] if any(w.startswith('which way is up') for w in result['why']) else [])
    name = None
    if a.discogs:
        try:
            import scan_naam as N
            name = N.naam(a.discogs, 'sticker', a.variant, a.year)
        except Exception as e:
            result['name_note'] = str(e)
    name = name or f'{os.path.splitext(os.path.basename(a.scans[0]))[0]}_sticker'
    os.makedirs(a.out, exist_ok=True)
    path = a.target or os.path.join(a.out, name + '.png'); n = 2
    while not a.target and os.path.exists(path) and not a.decision:
        path = os.path.join(a.out, f'{name}_{n}.png'); n += 1
    cv2.imwrite(path, img)
    if dpi:
        try:
            from PIL import Image
            with Image.open(path) as im_:
                im_.load(); im_.save(path, dpi=(dpi, dpi))
        except Exception:
            pass
    result['path'] = path; result['dpi'] = dpi
    result['size_mm'] = [round(img.shape[1] / dpi * 25.4, 1), round(img.shape[0] / dpi * 25.4, 1)] if dpi else None
    for f in (crop, up):
        if os.path.exists(f):
            os.remove(f)
    note_fast_reading(result)
    print(json.dumps(result, ensure_ascii=False))


def scan_background(path):
    """Achtergrond van een plaatscan langs de scanrand: 'lid' (de witte klep: licht en kleurloos) of 'sheet' (een
    gekleurd of donker vel op de plaat). Optie A voor doorschijnend vinyl gebruikt van elk een stel."""
    im = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    if im is None:
        im = cv2.imread(path)
        if im is None:
            return 'unknown'
        im = cv2.resize(im, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]; b = max(3, int(.03 * min(h, w)))
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    # de plaat zelf kan tot aan de scanrand liggen: tel alleen het deel van de rand dat wit en kleurloos is
    white = (ring[:, 0] * 100 / 255 > 80) & (np.hypot(ring[:, 1] - 128, ring[:, 2] - 128) < 10)
    return 'lid' if white.mean() > .25 else 'sheet'


def map_circle(src_raw, dst_raw, circle):
    """Een cirkel (X, Y, R in px) van het ene samengevoegde beeld op het andere leggen: zelfde plaat, dus een
    gelijkvormige verplaatsing (schuiven, draaien, schaal 1). Op overeenkomende punten (label, groeven), op 1/4."""
    a_ = cv2.imread(src_raw, cv2.IMREAD_REDUCED_GRAYSCALE_4); b_ = cv2.imread(dst_raw, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if a_ is None or b_ is None:
        return None
    clahe = cv2.createCLAHE(2.0, (8, 8)); sift = cv2.SIFT_create(8000)
    ka, da = sift.detectAndCompute(clahe.apply(a_), None); kb, db = sift.detectAndCompute(clahe.apply(b_), None)
    if da is None or db is None:
        return None
    m = [x[0] for x in cv2.BFMatcher().knnMatch(da, db, k=2) if len(x) == 2 and x[0].distance < .75 * x[1].distance]
    if len(m) < 30:
        return None
    M, inl = cv2.estimateAffinePartial2D(np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m]),
                                         method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=5000, confidence=.999)
    if M is None or int(inl.sum()) < 30 or abs(float(np.hypot(M[0, 0], M[1, 0])) - 1) > .01:
        return None
    X, Y, R = circle
    x2, y2 = M @ np.array([X / 4, Y / 4, 1.0])
    return [float(x2 * 4), float(y2 * 4), float(R * np.hypot(M[0, 0], M[1, 0]))]


def sharpest_label(scans, release_dpi):
    """Sleeve Transparant (alle vinyl): welke scan van deze kant het scherpste volledige label heeft. Per scan
    snijdt scan_label.extract het label uit (eigen rand en gat gemeten); een zwakke rand aan één kant (twijfel) mag.
    Scherpte = Laplace-variantie binnen 40 mm van het labelmidden, na het rekken van de toon (zo telt alleen scherpte).
    Geeft (pad, label RGBA, feiten, dpi, scores) of None."""
    import scan_label
    cands = []
    for f in scans:
        dpi = scan_resolution(f, release_dpi)
        if not dpi:
            continue
        try:
            lab, facts = scan_label.extract(f, dpi)
        except Exception:
            continue
        if lab is None or not facts.get('diameter_mm'):
            continue
        al = lab[..., 3] == 255
        g = cv2.cvtColor(lab[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = np.percentile(g[al], [5, 99.5])
        # vast gebied (straal 40 mm rond het labelmidden): de gemeten maat verschilt per scan (92 of 100 mm: binnenring of
        # buitenrand), en meer rand-tekst zou anders als 'scherper' tellen
        hh, ww = al.shape
        yy, xx = np.ogrid[:hh, :ww]
        core = (np.hypot(xx - ww / 2, yy - hh / 2) < 40 * dpi / 25.4) & (cv2.erode(al.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
        sc = float(cv2.Laplacian(((g - lo) / max(hi - lo, 1) * 100).astype(np.float32), cv2.CV_32F)[core].var())
        cands.append((sc, f, lab, facts, dpi))
    if not cands:
        return None
    scores = [(os.path.basename(c[1]), round(c[0], 1), c[3]['diameter_mm']) for c in cands]
    sc, f, lab, facts, dpi = max(cands, key=lambda c: c[0])
    return f, lab, facts, dpi, scores, [c[1] for c in cands if c[1] != f]   # de andere volledige labels: voor compose_label


_COMPOSED = {}


def compose_label(piece, centre, others, dpi):
    """Sleeve Transparant: het label per plek uit de scherpste scan samenstellen (GF magenta A, 20 nov: in de labelscan 0010
    was de randtekst rechtsboven zijwaarts uitgesmeerd, in 0002 daar 1,5-3x scherper; elders juist andersom).
    - Basis en maatvoering: de labelscan (piece). Elke andere scan met een volledig label: eerst gelijkvormig uitgelijnd
      (SIFT binnen het label), dan plaatselijk: het label lag niet bij elke scan even plat, aan de rand tot 6 px verschil.
      Fasecorrelatie op een raster (elke 2,5 mm, venster 8 mm), uitschieters weg, glad veld (sigma 3 mm). Controle op een
      verschoven raster; ligt de scan daarna niet binnen 0,6 px (mediaan), dan doet hij niet mee.
    - Toon plaatselijk (per kanaal, 3 mm) op die van de labelscan; per plek de scherpste scan (Laplace na toonrek, sigma 3 mm; de labelscan
      houdt het tenzij een andere 1,5x scherper is), overgang 2 mm, aan de rand van een andere scan 4 mm. Alleen tussen 9 mm van het midden (gat en spil blijven uit de labelscan) en 1 mm binnen de labelrand:
      de rand zelf en het vinyl blijven precies zoals de labelscan ze toont, dus de randlogica van paste_label verandert niet.
    - Geen winst (randtekst per 30 graden nergens 1,2x scherper, of ergens onder 0,93x) of geen bruikbare scan: piece ongewijzigd.
    Geeft (piece, tekst)."""
    import scan_label
    from scan_stitch import open_srgb
    pm = dpi / 25.4; q = 4
    cx, cy, sr = centre
    H, W = piece.shape[:2]
    yy, xx = np.ogrid[:H, :W]
    dist = np.hypot(xx - cx, yy - cy)
    # buitenrand van het label in de labelscan (label_boundary vindt soms een gedrukte binnenring: GF magenta A 92 mm)
    r_edge = sr
    r_out = float(scan_label.outer_material_boundary(piece, cx, cy, sr, pm))
    if r_out > sr + .5 * pm:
        col = lambda m: np.median(piece[m].astype(np.float64), 0)
        band = col((dist > sr + .3 * pm) & (dist < r_out - .3 * pm))
        if np.linalg.norm(band - col((dist > .6 * sr) & (dist < .85 * sr))) < np.linalg.norm(band - col((dist > r_out + 1 * pm) & (dist < r_out + 3 * pm))):
            r_edge = r_out
    sift = cv2.SIFT_create(8000); clahe = cv2.createCLAHE(2.0, (8, 8))
    grey = lambda x: clahe.apply(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
    pmask = ((dist < .95 * r_edge) * 255).astype(np.uint8)
    kb, db = sift.detectAndCompute(grey(piece), pmask)
    soft = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    base_s = soft(piece)
    P = int(round(4 * pm))
    hw = cv2.createHanningWindow((2 * P, 2 * P), cv2.CV_32F)
    def shifts(img, cov, step, off):
        g = soft(img); pts = []; sh = []
        for y in range(P + off, H - P, step):
            for x in range(P + off, W - P, step):
                r = math.hypot(x - cx, y - cy)
                if r < 9 * pm or r > r_edge - .5 * pm or not cov[y - P:y + P, x - P:x + P].all():
                    continue
                pb = base_s[y - P:y + P, x - P:x + P]
                if pb.std() < 6:
                    continue
                # kopieën: phaseCorrelate (OpenCV 5.0) beschadigde het venster na een eerste ronde, daarna 0 treffers
                (dx, dy), resp = cv2.phaseCorrelate(pb.copy(), g[y - P:y + P, x - P:x + P].copy(), hw.copy())
                if resp > .15 and abs(dx) < 10 and abs(dy) < 10:
                    pts.append((x, y)); sh.append((dx, dy))
        return np.array(pts, np.float32).reshape(-1, 2), np.array(sh, np.float32).reshape(-1, 2)
    step = int(round(2.5 * pm))
    gy, gx = np.mgrid[0:H:16, 0:W:16].astype(np.float32)
    fy, fx = np.mgrid[:H, :W].astype(np.float32)
    used = [('labelscan', piece.astype(np.float32), np.ones((H, W), bool))]
    notes = []
    for path in others:
        name = os.path.basename(path)
        try:
            with open_srgb(path) as s:
                src = cv2.cvtColor(np.asarray(s), cv2.COLOR_RGB2BGR)
            f = scan_label.label_boundary(cv2.resize(src, None, fx=1/q, fy=1/q, interpolation=cv2.INTER_AREA), pm / q)
            if f is None:
                notes.append(f'{name}: no label edge'); continue
            if os.environ.get('SLEEVE_LICHTKOLOM'):
                import scan_lichtkolom
                scan_lichtkolom.apply(src, scan_lichtkolom.load_profile(os.environ['SLEEVE_LICHTKOLOM']))
            ox, oy, orr = f[1] * q, f[2] * q, f[3] * q; m = int(orr + 8 * pm)
            a0, b0 = max(0, int(ox) - m), max(0, int(oy) - m)
            O = src[b0:int(oy) + m, a0:int(ox) + m].copy(); src = None
            om = np.zeros(O.shape[:2], np.uint8); cv2.circle(om, (int(ox - a0), int(oy - b0)), int(.95 * orr), 255, -1)
            ko, do = sift.detectAndCompute(grey(O), om)
            if do is None or db is None:
                notes.append(f'{name}: no match'); continue
            mm = [x[0] for x in cv2.BFMatcher().knnMatch(do, db, k=2) if len(x) == 2 and x[0].distance < .75 * x[1].distance]
            if len(mm) < 20:
                notes.append(f'{name}: no match'); continue
            T, inl = cv2.estimateAffinePartial2D(np.float32([ko[x.queryIdx].pt for x in mm]), np.float32([kb[x.trainIdx].pt for x in mm]),
                                                 method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=5000, confidence=.999)
            if T is None or int(inl.sum()) < 50 or abs(float(np.hypot(T[0, 0], T[1, 0])) - 1) > .01:
                notes.append(f'{name}: not aligned'); continue
            V = cv2.warpAffine(O, T, (W, H), flags=cv2.INTER_LANCZOS4)
            cov = cv2.warpAffine(np.full(O.shape[:2], 255, np.uint8), T, (W, H), flags=cv2.INTER_NEAREST) == 255; O = None
            pts, sh = shifts(V, cov, step, 0)
            if len(pts) < 40:
                notes.append(f'{name}: too little structure to align'); continue
            keep = np.array([np.hypot(*(sh[i] - np.median(sh[np.hypot(*(pts - pts[i]).T) < 4 * pm], 0))) <= 1.0 for i in range(len(pts))])
            pts, sh = pts[keep], sh[keep]
            s2 = 2 * (3 * pm) ** 2
            F = np.zeros(gx.shape + (2,), np.float32); Wt = np.zeros(gx.shape, np.float32)
            for (x, y), (dx, dy) in zip(pts, sh):
                w_ = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / s2); F[..., 0] += w_ * dx; F[..., 1] += w_ * dy; Wt += w_
            F /= np.maximum(Wt, 1e-6)[..., None]
            F = cv2.resize(F, (W, H), interpolation=cv2.INTER_LINEAR)
            V = cv2.remap(V, fx + F[..., 0], fy + F[..., 1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
            cov = cv2.remap(cov.astype(np.uint8) * 255, fx + F[..., 0], fy + F[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) == 255
            cov = cv2.erode(cov.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            p2, s2_ = shifts(V, cov, step, step // 2)
            rest = float(np.median(np.hypot(*s2_.T))) if len(p2) >= 20 else 99.
            if rest > .6:
                notes.append(f'{name}: lies {rest:.2f} px off after local alignment, not used'); continue
            # toon plaatselijk (per kanaal, sigma 3 mm) op die van de labelscan: één factor voor het hele label liet een naad
            # zien waar de scans verschillend belicht waren (Together Alone helder A: donkerder vlak naast de randtekst)
            v = V.astype(np.float32); V = None
            cw = (cov & (dist < r_edge - .5 * pm)).astype(np.float32)
            blur = lambda x: cv2.GaussianBlur(x, (0, 0), 3 * pm)
            den = np.maximum(blur(cw), 1e-3)
            for ch in range(3):
                ratio = (blur(piece[..., ch].astype(np.float32) * cw) / den + 1) / (blur(v[..., ch] * cw) / den + 1)
                v[..., ch] *= np.clip(ratio, .7, 1.4)
            used.append((name, v, cov))
            notes.append(f'{name}: local shift up to {float(np.abs(F[cov]).max()):.1f} px, {rest:.2f} px after')
        except Exception as e:                           # nooit het label laten mislukken om een extra scan
            notes.append(f'{name}: skipped ({e})')
    if len(used) == 1:
        return piece, 'label from one scan (' + '; '.join(notes) + ')' if notes else 'label from one scan'
    lab = dist < r_edge - .5 * pm
    rim = (dist > r_edge - 6 * pm) & (dist < r_edge - 1 * pm)
    sharp = []
    for _, v, cov in used:
        g = cv2.cvtColor(np.clip(v, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = np.percentile(g[lab & cov], [5, 99.5])
        L = cv2.Laplacian(((g - lo) / max(hi - lo, 1) * 100).astype(np.float32), cv2.CV_32F) ** 2
        sharp.append(cv2.GaussianBlur(L, (0, 0), 3 * pm) * cov)
    # per plek één scan (geen gemiddelde: mengen van twee even scherpe scans middelt ruis en een restverschil van 0,3 px weg
    # en kostte scherpte); de labelscan houdt het tenzij een andere daar 1,5x scherper is. Overgang zacht over 0,5 mm.
    S = np.stack(sharp); S[0] *= 1.5
    pick = S.argmax(0)
    # overgang 2 mm, en aan de rand van een andere scan (waar die scan ophoudt) over 4 mm teruglopen naar de labelscan:
    # een scherpere scan toont ook de papierstructuur scherper; met 0,5 mm stond er een rechte naad (Together Alone helder A)
    # de overgang ligt helemaal binnen het stuk van de andere scan (0 op de grens, vol 2 mm verder): een overgang over de
    # grens heen mengde de vagere scan in waar de labelscan juist scherper was (randtekst daar 0,83x)
    Wt = np.zeros((len(used), H, W), np.float32)
    for i in range(1, len(used)):
        Wt[i] = np.clip(2 * cv2.GaussianBlur((pick == i).astype(np.float32), (0, 0), 2 * pm) - 1, 0, 1)
    for i in range(1, len(used)):
        edge_d = cv2.distanceTransform(used[i][2].astype(np.uint8), cv2.DIST_L2, 5)
        Wt[i] *= np.clip(edge_d / (4 * pm), 0, 1)
    Wt[0] = np.clip(1 - Wt[1:].sum(0), 0, 1)
    region = (np.clip((dist - 9 * pm) / pm, 0, 1) * np.clip((r_edge - 1.0 * pm - dist) / (.5 * pm), 0, 1)).astype(np.float32)
    mix = sum(Wt[i][..., None] * u[1] for i, u in enumerate(used))
    out = np.clip(mix * region[..., None] + piece.astype(np.float32) * (1 - region[..., None]), 0, 255).astype(np.uint8)
    # winst per 30 graden randtekst (één getal over de hele rand middelde winst op de ene plek weg tegen ruis elders)
    ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360) % 360
    def rim_sharp(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = np.percentile(g[lab], [5, 99.5])
        L = cv2.Laplacian(((g - lo) / max(hi - lo, 1) * 100).astype(np.float32), cv2.CV_32F)
        return np.array([L[rim & (ang >= a) & (ang < a + 30)].var() for a in range(0, 360, 30)])
    ratio = rim_sharp(out) / np.maximum(rim_sharp(piece), 1e-6)
    share = ', '.join(f'{u[0]} {100 * float(Wt[i][lab & (region > 0)].mean()):.0f}%' for i, u in enumerate(used))
    if ratio.max() < 1.2 or ratio.min() < .93:
        return piece, f'label from one scan (combining: rim text {ratio.min():.2f}-{ratio.max():.2f}x per 30 degrees, not enough; ' + '; '.join(notes) + ')'
    return out, f'label combined from the sharpest scan per spot ({share}; rim text up to {ratio.max():.2f}x sharper, nowhere below {ratio.min():.2f}x; ' + '; '.join(notes) + ')'


def paste_label(raw, scan_path, dpi, others=(), expand_mm=None, label_data=None):
    """Sleeve Transparant: het label uit de scan met het scherpste label op het samengevoegde beeld zetten.
    - Stuk: de scan rond het label (labelrand + 3 mm), uitgelijnd met SIFT, gelijkvormig (schuiven, draaien,
      schaal 1), eerst grof op 1/4, dan scherp op ware grootte, alleen op punten binnen het label.
    - Grens: de labelrand zoals het samengevoegde beeld hem toont (de andere scans: schone rand, geen schaduw),
      per halve graad gemeten en begrensd op -0,2..+0,45 mm rond de cirkel (dieper staat tekst langs de rand); tot 0,1 mm daarbinnen, zacht over 1 px.
      De eigen rand van de labelscan was soms zwak of had een schaduwstrook (GF magenta A, 0010).
    - Toon per kanaal op het licht van het samengevoegde label.
    Geeft (tekst, None) of (None, reden)."""
    if expand_mm is None:
        expand_mm = 1.0 if os.environ.get('SLEEVE_LABEL_EXPAND_MM') == '1' else 0.0
    if expand_mm not in (0, 1):
        return None, 'unsupported label retouch amount'
    import scan_label
    from scan_stitch import open_srgb
    im = cv2.imread(raw, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None, 'the merged image could not be read'
    pm = dpi / 25.4; q = 4
    if expand_mm:
        # Reuse the native material cut. A Hough print ring is not a paper mask.
        # Confirm a lasting material change independently of the print-circle finder.
        label_data = scan_label.extract(scan_path, dpi, material_only=True)
        label_pixels, label_facts = label_data
        if (label_pixels is None or not label_facts.get('source_bounds')
                or not label_facts.get('source_label_ellipse')
                or label_facts.get('edge_coverage', 0) < .8):
            return None, 'retouch not applied: native paper boundary was not measured'
    # labelcirkel in de labelscan en in het samengevoegde beeld (op 1/4)
    with open_srgb(scan_path) as src_img:
        src = cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR)
    # labelrand meten op de onbewerkte scan (de randzoeker is gevoelig: met de lichtcorrectie vond hij het label in Together
    # Alone helder 0013 niet meer), daarna dezelfde lichtcorrectie als bij het samenvoegen (anders keren de strepen terug)
    fs = scan_label.label_boundary(cv2.resize(src, None, fx=1/q, fy=1/q, interpolation=cv2.INTER_AREA), pm / q)
    for q2 in ((5, 3) if fs is None else ()):
        # De cirkelzoeker kiest op één schaal soms een verkeerde kandidaat (US white B, labelscan 0015: op 1/4 niets,
        # op 1/5 en 1/3 dezelfde rand). Dan een andere schaal, terug omgerekend naar 1/4.
        f2 = scan_label.label_boundary(cv2.resize(src, None, fx=1/q2, fy=1/q2, interpolation=cv2.INTER_AREA), pm / q2)
        if f2 is not None:
            fs = (f2[0], f2[1] * q2 / q, f2[2] * q2 / q, f2[3] * q2 / q, f2[4])
            break
    if fs is None:
        fs = scan_label.closeup_label(cv2.resize(src, None, fx=1/q, fy=1/q, interpolation=cv2.INTER_AREA), pm / q)   # losse labelscan
    if fs is not None:
        # Sleeve Transparant: het label ligt rond het middengat. Geen gat in het midden van de gevonden cirkel of geen
        # lp-labelmaat, en wel een gat elders met een labelrand eromheen: dat is het label (Trails of Fails 2022 kant B: de
        # gevonden cirkel was een rond stuk foto tussen de takken)
        from scan_feiten import find_hole as _fh
        X_, Y_, R_ = fs[1] * q, fs[2] * q, fs[3] * q; k_ = int(6 * pm)
        sub_ = src[int(max(0, Y_ - k_)):int(Y_ + k_), int(max(0, X_ - k_)):int(X_ + k_)]
        st_ = _fh(sub_, 3.6 * pm, 1.7 * pm)[2] if sub_.size else 0
        if st_ < .3 or 2 * R_ / pm < 97:
            h_ = scan_label.hole_in_scan(src, pm)
            if h_ is not None and np.hypot(h_[0] - X_, h_[1] - Y_) > 3 * pm:
                e_ = scan_label.median_edge(src, h_[0], h_[1], pm)
                if e_ is not None and 97 <= 2 * e_ / pm <= 103:
                    fs = (fs[0], h_[0] / q, h_[1] / q, e_ / q, fs[4])
    if os.environ.get('SLEEVE_LICHTKOLOM'):
        import scan_lichtkolom
        scan_lichtkolom.apply(src, scan_lichtkolom.load_profile(os.environ['SLEEVE_LICHTKOLOM']))
    small = cv2.resize(im[..., :3], None, fx=1/q, fy=1/q, interpolation=cv2.INTER_AREA)
    small[cv2.resize(im[..., 3], (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) == 0] = 255
    fr = scan_label.label_boundary(small, pm / q)
    for q2 in ((5, 3) if fr is None else ()):
        # zelfde als bij de labelscan: op een andere schaal vindt de cirkelzoeker de rand soms wel (FBOW 2023 rood)
        sm2 = cv2.resize(im[..., :3], None, fx=1/q2, fy=1/q2, interpolation=cv2.INTER_AREA)
        sm2[cv2.resize(im[..., 3], (sm2.shape[1], sm2.shape[0]), interpolation=cv2.INTER_NEAREST) == 0] = 255
        f2 = scan_label.label_boundary(sm2, pm / q2)
        if f2 is not None:
            fr = (f2[0], f2[1] * q2 / q, f2[2] * q2 / q, f2[3] * q2 / q, f2[4]); break
    if fr is None and fs is not None:
        # Op het hele samengevoegde beeld (veel wit, scanranden) vond de cirkelzoeker het label soms niet (Together Alone
        # helder A, 20 nov). Dan eerst grof opzoeken met de labelscan (SIFT op 1/4), en de rand meten op een uitsnede daar.
        r4 = int(fs[3] + 6 * pm / q)
        lab4 = cv2.resize(src, None, fx=1/q, fy=1/q, interpolation=cv2.INTER_AREA)
        lab4 = lab4[max(0, int(fs[2]) - r4):int(fs[2]) + r4, max(0, int(fs[1]) - r4):int(fs[1]) + r4]
        sf = cv2.SIFT_create(4000); cl = cv2.createCLAHE(2.0, (8, 8))
        ka, da = sf.detectAndCompute(cl.apply(cv2.cvtColor(lab4, cv2.COLOR_BGR2GRAY)), None)
        kb, db = sf.detectAndCompute(cl.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)), None)
        if da is not None and db is not None:
            mm = [x[0] for x in cv2.BFMatcher().knnMatch(da, db, k=2) if len(x) == 2 and x[0].distance < .75 * x[1].distance]
            if len(mm) >= 20:
                T, inl = cv2.estimateAffinePartial2D(np.float32([ka[x.queryIdx].pt for x in mm]), np.float32([kb[x.trainIdx].pt for x in mm]),
                                                     method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if T is not None and int(inl.sum()) >= 20:
                    gx, gy = T @ np.array([lab4.shape[1] / 2, lab4.shape[0] / 2, 1.0])
                    c0, c1 = max(0, int(gx) - r4), max(0, int(gy) - r4)
                    sub = small[c1:int(gy) + r4, c0:int(gx) + r4]
                    f2 = scan_label.label_boundary(sub, pm / q)
                    if f2 is not None:
                        fr = (f2[0], f2[1] + c0, f2[2] + c1, f2[3], f2[4])
                    else:
                        # Feit: de uitlijning legt het label van de labelscan op het samengevoegde beeld; dat is de plek
                        # (grof; de precieze uitlijning en de rand volgen hieronder op ware grootte)
                        fr = (0.0, float(gx), float(gy), float(fs[3]), 0.0)
    if fs is None or fr is None:
        return None, 'the label edge could not be found'
    sx, sy, sr = fs[1] * q, fs[2] * q, fs[3] * q
    if expand_mm:
        sx, sy, da_, db_, angle_ = label_facts['source_label_ellipse']
        sr = (da_ + db_) / 4
    rx, ry, rr = fr[1] * q, fr[2] * q, fr[3] * q
    m = int(sr + 8 * pm)                                 # ruim: de gevonden cirkel kan een binnenring zijn (92 mm bij een 100 mm-label)
    a0, b0 = max(0, int(sx) - m), max(0, int(sy) - m)
    piece = src[b0:min(src.shape[0], int(sy) + m), a0:min(src.shape[1], int(sx) + m)].copy(); src = None
    yy, xx = np.ogrid[:piece.shape[0], :piece.shape[1]]
    pmask = ((np.hypot(xx - (sx - a0), yy - (sy - b0)) < .95 * sr) * 255).astype(np.uint8)
    compose_note = ''
    match_piece = piece                                  # uitlijnen op het eigen label van de labelscan: dezelfde plaatsing als zonder samenstellen
    if others:                                           # Sleeve Transparant: per plek de scherpste scan (compose_label)
        key = (scan_path, tuple(others), piece.shape)
        if key not in _COMPOSED:                         # rechtgezette variant: hetzelfde stuk, niet opnieuw samenstellen
            _COMPOSED[key] = compose_label(piece, (sx - a0, sy - b0, sr), others, dpi)
        piece, compose_note = _COMPOSED[key][0].copy(), _COMPOSED[key][1]
    M = int(rr + 9 * pm)
    x0, y0 = max(0, int(rx) - M), max(0, int(ry) - M); x1, y1 = min(im.shape[1], int(rx) + M), min(im.shape[0], int(ry) + M)
    crop = im[y0:y1, x0:x1]
    sift = cv2.SIFT_create(8000); clahe = cv2.createCLAHE(2.0, (8, 8))
    def grey(x):
        return clahe.apply(cv2.cvtColor(x[..., :3], cv2.COLOR_BGR2GRAY))
    def match(a_img, a_mask, b_img, thr):
        ka, da = sift.detectAndCompute(grey(a_img), a_mask); kb, db = sift.detectAndCompute(grey(b_img), None)
        if da is None or db is None:
            return None, 0
        mm = [x[0] for x in cv2.BFMatcher().knnMatch(da, db, k=2) if len(x) == 2 and x[0].distance < .75 * x[1].distance]
        if len(mm) < 20:
            return None, 0
        T, inl = cv2.estimateAffinePartial2D(np.float32([ka[x.queryIdx].pt for x in mm]), np.float32([kb[x.trainIdx].pt for x in mm]),
                                             method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=5000, confidence=.999)
        return T, (0 if T is None else int(inl.sum()))
    A, n = match(match_piece, pmask, crop, 1.5); match_piece = None
    if A is None or n < 50 or abs(float(np.hypot(A[0, 0], A[1, 0])) - 1) > .01:
        return None, 'the sharpest label could not be laid precisely on the record'
    original_label_piece = piece if expand_mm else None
    w = cv2.warpAffine(piece, A, (x1 - x0, y1 - y0), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cover = cv2.warpAffine(np.full(piece.shape[:2], 255, np.uint8), A, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST) > 0
    piece = None
    # labelrand per halve graad in het samengevoegde beeld. Eerst ruim zoeken en een cirkel door de randpunten passen
    # (de grove cirkel lag soms ±0,5 mm uit het midden: dan sneed de begrenzing door tekst langs de rand), dan per graad.
    cx, cy = rx - x0, ry - y0
    N = 720
    th = np.arange(N, dtype=np.float32)[:, None] * (2 * np.pi / N)
    stp = max(.25, .05 * pm)
    j = max(1, int(round(.15 * pm / stp)))
    src_f = crop[..., :3].astype(np.float32)
    def edges(cx, cy, r0, lo, hi, img=None, first=False):
        rad = (r0 + np.arange(lo * pm, hi * pm, stp, dtype=np.float32))[None, :]
        prof = cv2.remap(src_f if img is None else img, (cx + np.cos(th) * rad).astype(np.float32), (cy + np.sin(th) * rad).astype(np.float32), cv2.INTER_LINEAR)
        prof = cv2.GaussianBlur(prof, (0, 0), sigmaX=max(1.0, .1 * pm / stp), sigmaY=1.0)
        grad = np.linalg.norm(prof[:, 2 * j:] - prof[:, :-2 * j], axis=2)
        at = grad.argmax(1)
        if first:
            # de buitenste plek die nog de kleur van het label heeft (gemeten binnen in het label): tekst en drukwerk liggen
            # daarbinnen, het lichte randje of vinyl van een labelscan uit een andere sessie daarbuiten (Together Alone helder
            # 0013: licht randje; met 'eerste overgang' telde de binnenkant van witte randtekst als rand, GF magenta B, 20 nov)
            src_ = src_f if img is None else img
            ring_ = cv2.remap(src_, (cx + np.cos(th) * (r0 * np.linspace(.86, .94, 8))[None, :]).astype(np.float32),
                              (cy + np.sin(th) * (r0 * np.linspace(.86, .94, 8))[None, :]).astype(np.float32), cv2.INTER_LINEAR)
            label_col = np.median(ring_.reshape(-1, 3), 0)
            like = np.linalg.norm(prof - label_col, axis=2) < max(30.0, .25 * np.linalg.norm(label_col - np.median(prof[:, -max(2, prof.shape[1] // 6):].reshape(-1, 3), 0)))
            has = like.any(1)
            last = prof.shape[1] - 1 - np.argmax(like[:, ::-1], 1)
            at = np.clip(np.where(has, last - j, at), 0, grad.shape[1] - 1)
            return rad[0, np.minimum(at + j, rad.shape[1] - 1)].copy(), has & (last < prof.shape[1] - 2)
        return rad[0, at + j].copy(), grad[np.arange(N), at] >= 20
    # de buitenrand van het label (papier naar vinyl): label_boundary vindt soms een gedrukte binnenring (GF magenta A:
    # 92 mm, de echte rand 99,6 mm); scan_label.outer_material_boundary zoekt de overgang van materiaal daarbuiten
    # De materiaalovergang daarbuiten alleen nemen als de strook tussen de twee cirkels label is (kleur als het label binnenin),
    # niet vinyl. Was de eerste cirkel al de echte rand (magenta B zonder kleurstap: 99,6 mm), dan vond outer_material_boundary
    # een overgang in het vinyl verderop, sneed het label daar af en kwam een reepje vinyl uit de labelscan mee (20 nov).
    r_out = float(scan_label.outer_material_boundary(crop[..., :3], cx, cy, rr, pm))
    if r_out > rr + .5 * pm:
        yy_, xx_ = np.ogrid[:crop.shape[0], :crop.shape[1]]
        d_ = np.hypot(xx_ - cx, yy_ - cy)
        ok_px = crop[..., 3] == 255
        col = lambda m: np.median(crop[..., :3][m & ok_px].astype(np.float64), 0)
        band = col((d_ > rr + .3 * pm) & (d_ < r_out - .3 * pm))
        inside = col((d_ > .6 * rr) & (d_ < .85 * rr))
        outside = col((d_ > r_out + 1.0 * pm) & (d_ < r_out + 3.0 * pm))
        if np.linalg.norm(band - inside) < np.linalg.norm(band - outside):
            rr = r_out
    e0, ok = edges(cx, cy, rr, -.25, 2.0)
    if ok.sum() >= N // 2:
        px_ = cx + np.cos(th[:, 0]) * e0; py_ = cy + np.sin(th[:, 0]) * e0; keep = ok.copy()
        for _ in range(4):
            xs, ys = px_[keep], py_[keep]
            d_, e_, f_ = np.linalg.lstsq(np.c_[xs, ys, np.ones(len(xs))], -(xs * xs + ys * ys), rcond=None)[0]
            fx, fy = -d_ / 2, -e_ / 2; fr_ = np.sqrt(max(0, (d_ * d_ + e_ * e_) / 4 - f_))
            new_keep = ok & (np.abs(np.hypot(px_ - fx, py_ - fy) - fr_) < .3 * pm)
            if new_keep.sum() < N // 3 or np.array_equal(new_keep, keep):
                break
            keep = new_keep
        if abs(fr_ - rr) < 2 * pm and np.hypot(fx - cx, fy - cy) < 2 * pm:   # alleen een kleine correctie
            cx, cy, rr = fx, fy, fr_
    # Meten rond het label zoals de labelscan het toont (zijn eigen cirkel, via de uitlijning naar dit beeld): in de
    # samengevoegde scans maakt een schaduw naast het label de rand soms 0,3-0,5 mm groter, en dan kwam er een licht
    # randje vinyl uit de labelscan mee (Together Alone helder A, 20 nov). Alleen als die cirkel past bij de gevonden rand.
    lx_, ly_ = A @ np.array([sx - a0, sy - b0, 1.0])
    if abs(sr - rr) < 1.5 * pm and np.hypot(lx_ - cx, ly_ - cy) < 1.5 * pm:
        cx, cy, rr = float(lx_), float(ly_), float(sr)
    edge, ok = edges(cx, cy, rr, -.25, 1.2)                # niet dieper dan -0,25 mm: daar staat tekst langs de rand
    # dezelfde meting in de uitgelijnde labelscan; per graad de kleinste rand van de twee: waar een van beide een schaduw
    # naast het label heeft, blijft het label binnen de echte rand (geen roze reepje in het label, geen veeg erbuiten)
    edge2, ok2 = edges(cx, cy, rr, -.9, 1.2, w.astype(np.float32), first=True)   # ruimer naar binnen: de schaduw in de samengevoegde scans maakt rr soms 0,5 mm te groot
    if ok2.sum() >= N // 3:
        # waar de labelscan geen duidelijke rand toont: zijn rand overnemen van de buren, niet de (door schaduw grotere) rand
        # van het samengevoegde beeld; anders kwam daar een wit vlakje uit de labelscan mee (Together Alone helder A)
        i2 = np.flatnonzero(ok2)
        edge2 = np.where(ok2, edge2, np.interp(np.arange(N), np.r_[i2 - N, i2, i2 + N], np.r_[edge2[i2], edge2[i2], edge2[i2]]))
        edge = np.where(ok, np.minimum(edge, edge2), edge2); ok = np.ones(N, bool)
    else:
        edge = np.where(ok & ok2, np.minimum(edge, edge2), np.where(ok2, edge2, edge)); ok = ok | ok2
    weak = ~ok
    if ok.sum() >= 8:                                    # geen duidelijke rand: overnemen van de buren met een duidelijke rand (rondom)
        ii = np.flatnonzero(ok)
        edge[weak] = np.interp(np.flatnonzero(weak), np.r_[ii - N, ii, ii + N], np.r_[edge[ii], edge[ii], edge[ii]])
    else:
        edge[weak] = rr
    edge = np.concatenate([edge[-7:], edge, edge[:7]])
    edge = np.clip(np.array([np.median(edge[i:i + 15]) for i in range(N)], np.float32), rr - .8 * pm, rr + .45 * pm)
    yy, xx = np.ogrid[:y1 - y0, :x1 - x0]
    dist = np.hypot(xx - cx, yy - cy)
    e = edge[(((np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)) * (N / (2 * np.pi))).astype(np.int32) % N]
    # Explicit user retouch: enlarge the measured PAPER, never an inner print ring.
    if expand_mm:
        cx, cy = A @ np.array([sx-a0, sy-b0, 1.])
        rr = sr * float(np.hypot(A[0, 0], A[1, 0]))
        yy, xx = np.ogrid[:y1-y0, :x1-x0]
        dist = np.hypot(xx-cx, yy-cy)
        scale = (rr + float(expand_mm) * pm) / rr
        transform = np.array([[scale, 0, cx*(1-scale)], [0, scale, cy*(1-scale)]], np.float64)
        enlarged_A = transform @ np.vstack([A, [0., 0., 1.]])
        w = cv2.warpAffine(original_label_piece, enlarged_A, (w.shape[1], w.shape[0]), flags=cv2.INTER_LANCZOS4)
        cover = cv2.warpAffine(np.ones(original_label_piece.shape[:2], np.uint8), enlarged_A, (w.shape[1], w.shape[0]), flags=cv2.INTER_NEAREST) > 0
        original_label_piece = None
        # The alpha belongs to extract()'s refined native material ellipse,
        # in its own crop coordinates. Transform it with the SAME source map.
        left_, top_ = label_facts['source_bounds'][:2]
        mask_A = enlarged_A.copy()
        mask_A[:, 2] += enlarged_A[:, :2] @ np.array([left_-a0, top_-b0])
        paper_alpha = cv2.warpAffine(label_pixels[..., 3].astype(np.float32)/255., mask_A,
                                    (w.shape[1], w.shape[0]), flags=cv2.INTER_LINEAR)
        e = e * scale
    al = (np.clip(e - .1 * pm - dist + .5, 0, 1) * cover * (crop[..., 3] > 0)).astype(np.float32)[..., None]
    if expand_mm:
        al = (paper_alpha * cover * (crop[..., 3] > 0) * (dist > 6*pm))[..., None]

    inner = (dist < .9 * rr) & cover & (crop[..., 3] == 255)
    gain = crop[..., :3][inner].astype(np.float64).mean(0) / np.maximum(w[inner].astype(np.float64).mean(0), 1)
    gain = np.clip(gain, .8, 1.25)
    crop[..., :3] = np.clip(w * gain * al + crop[..., :3] * (1 - al), 0, 255).astype(np.uint8)
    shadow_note = ''
    if not expand_mm and os.environ.get('SLEEVE_LICHTKOLOM') and not os.environ.get('SLEEVE_LABEL_SHADOW_OFF'):   # uit: vergelijken
        # Transparant vinyl: schaduw van de labelrand op het vinyl net buiten het label (Together Alone helder B, 20 nov:
        # alle vier de kantscans hadden aan dezelfde kant een donkere, wazige band tot 43 grijswaarden; de labelscan daar
        # niet). In de strook tot 3 mm buiten de rand alleen lichter maken, naar wat de labelscan daar toont (zijn toon
        # verderop, 3,5-6 mm, gelijkgetrokken). De structuur van het vinyl blijft uit de samengevoegde scans.
        ok_ = cover & (crop[..., 3] == 255)
        vin = (ok_ & (dist > e + .03 * pm) & (dist < e + 6 * pm)).astype(np.float32)   # vanaf de snede (was 0,15 mm)
        far = (ok_ & (dist > e + 3.5 * pm) & (dist < e + 6 * pm)).astype(np.float32)
        def nblur(x, m, s):
            return cv2.GaussianBlur(x * m, (0, 0), s) / np.maximum(cv2.GaussianBlur(m, (0, 0), s), 1e-3)
        # randpixels van het label niet: die werden een licht bruin randje (eerste proef). Wat label is, zegt de labelscan
        # (schoon, uitgelijnd); op de helderheid van het samengevoegde beeld zelf afgaan liet de donkerste schaduw staan
        # (die is even donker als het label; tweede proef)
        grey_w = (w.astype(np.float32) * gain).mean(2)
        vin_tone = np.median(grey_w[far > 0]) if far.any() else 200.
        vinyl_w = np.clip((cv2.GaussianBlur(grey_w, (0, 0), 1.0) - .5 * vin_tone) / (.3 * vin_tone), 0, 1)
        # pas vanaf 0,3 mm buiten de rand (vol op 0,55 mm): het dunne donkere randje van het label zelf staat rondom in alle
        # kantscans en hoort erbij; alleen de wazige schaduw daarbuiten weg (vierde proef haalde ook dat randje weg)
        # Sleeve Transparant (21 nov, André: yellow B en clear B sluiten niet netjes aan): het begin op 0,3 mm liet een grijze
        # band van 0,3-0,5 mm staan (clear B 135-146 tegen vinyl 195). Nu vanaf 0,05 mm buiten de snede (vol op 0,2 mm); de
        # snede zelf ligt al 0,1 mm binnen de rand, dus het randje van het label blijft.
        fade = (np.clip((e + 3 * pm - dist) / pm, 0, 1) * np.clip((dist - e - .05 * pm) / (.15 * pm), 0, 1) * vinyl_w * ok_).astype(np.float32)
        # het verschil optellen, niet vermenigvuldigen: in de kantscans loopt het donker van het label 0,3 mm verder door dan
        # in de labelscan (49,8 tegen 50,1 mm); een factor (max 1,8) maakte daar een lichtbruin randje (derde proef).
        # Alleen lichter (verschil >= 0); de fijne structuur van het vinyl blijft uit de samengevoegde scans.
        add = np.zeros(crop.shape[:2] + (3,), np.float32)
        for ch in range(3):
            mc = crop[..., ch].astype(np.float32); wc = w[..., ch].astype(np.float32) * gain[ch]
            tone = (nblur(mc, far, 3 * pm) + 1) / (nblur(wc, far, 3 * pm) + 1)          # toon labelscan -> samengevoegd, plaatselijk
            add[..., ch] = np.maximum(nblur(wc, vin, .2 * pm) * tone - nblur(mc, vin, .2 * pm), 0)
        add = add.min(2, keepdims=True) * fade[..., None]                                # gelijk over de kanalen: geen kleurzweem
        band = fade > 0
        if band.any() and float(np.percentile(add[band], 99)) > 5:
            crop[..., :3] = np.clip(crop[..., :3] + add, 0, 255).astype(np.uint8)
            shadow_note = f'; label shadow on the vinyl lifted (up to +{float(np.percentile(add[band], 99.9)):.0f} within 3 mm of the edge)'
        # Sleeve Transparant (21 nov, André: clear B en yellow B sluiten niet netjes aan). Direct langs het label staat een
        # verhoogd randje dat per scan aan de kant van het licht een grijze band van 0,3-0,5 mm geeft (ook in de labelscan,
        # TA clear 0018: bij 90 graden 156-160 over 0,4 mm, bij 180 graden strak). Die strook (tot 0,8 mm buiten de snede)
        # alleen lichter maken naar het vinyl net daarbuiten (0,9-1,6 mm), plaatselijk gemeten; structuur blijft.
        near = (ok_ & (dist > e + .03 * pm) & (dist < e + .8 * pm)).astype(np.float32)
        ring = (ok_ & (dist > e + .9 * pm) & (dist < e + 1.6 * pm)).astype(np.float32)
        if near.any() and ring.any() and not os.environ.get('SLEEVE_LABEL_EDGEBAND_OFF'):   # uit: vergelijken
            lift = np.stack([np.maximum(nblur(crop[..., ch].astype(np.float32), ring, 1.5 * pm)
                                        - nblur(crop[..., ch].astype(np.float32), near, .15 * pm), 0) for ch in range(3)], 2).min(2)
            edge_fade = (np.clip((e + .9 * pm - dist) / (.3 * pm), 0, 1) * np.clip((dist - e - .03 * pm) / (.1 * pm), 0, 1)
                         * ok_).astype(np.float32)
            lift = lift * edge_fade
            if float(np.percentile(lift[near > 0], 99)) > 5:
                crop[..., :3] = np.clip(crop[..., :3].astype(np.float32) + lift[..., None], 0, 255).astype(np.uint8)
                shadow_note += f'; edge band beside the label lifted to the nearby vinyl (up to +{float(np.percentile(lift[near > 0], 99.9)):.0f})'
    im[y0:y1, x0:x1] = crop
    if not cv2.imwrite(raw, im, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        return None, 'the merged image could not be saved'
    dev = (edge - rr) / pm
    cut_note = (f'cut at the label edge of the merged image ({dev.min():+.2f} to {dev.max():+.2f} mm from the circle'
                + (f', {int(weak.sum())} of {N} directions without a clear edge' if weak.any() else '') + ')')
    if expand_mm:
        cut_note = 'cut from the native source paper mask; inner printed rings not used'
        shadow_note += '; user retouch: paper label and printing enlarged 1 mm radially (2 mm diameter); centre hole preserved'
    return (f'{n} matching points, tone x{gain[2]:.3f}/{gain[1]:.3f}/{gain[0]:.3f} (R/G/B), ' + cut_note
            + shadow_note + (f'; {compose_note}' if compose_note else '')), None


def note_fast_reading(result):
    """Sleeve 0.5.60: werd tekst (kant, stand, label) alleen snel gelezen omdat het nauwkeurige model van macOS niet werkte
    (28 nov: 'ANECompilerService ... mkdir failed'), dan staat dat in de uitkomst; de stand en kant zijn dan minder zeker."""
    import scan_rechtop as R_
    if getattr(R_, 'FAST_READS', 0):
        result['ocr_fast_only'] = True
        result['applied'] = (result.get('applied', '') + ' text read in fast mode only (accurate macOS text recognition unavailable)').strip()


def trim_disc(path, target_mm, dpi):
    """Sleeve 0.5.58: een uitgesneden plaatkant (RGBA, midden van het beeld) rondom kleiner maken tot target_mm: midden
    uitsnijden en dezelfde zachte cirkelrand als bij het snijden. Profiel en dpi blijven. Geeft de nieuwe maat in mm."""
    import scan_feiten as SF
    from PIL import Image
    with Image.open(path) as im:
        info = {k: im.info[k] for k in ('icc_profile', 'dpi') if k in im.info}
        arr = np.array(im.convert('RGBA'))
    n = arr.shape[0]; m = int(round(target_mm * dpi / 25.4))
    if arr.shape[0] != arr.shape[1] or m >= n or m < n - int(3 * dpi / 25.4):
        raise ValueError(f'unexpected sizes ({n} px -> {m} px)')
    b = (n - m) // 2; out = arr[b:b + m, b:b + m].copy()
    np.minimum(out[:, :, 3], SF.soft_disc(m, m / 2), out=out[:, :, 3])
    tmp = path + '.trim.png'; Image.fromarray(out, 'RGBA').save(tmp, **info); os.replace(tmp, path)
    return round(m / dpi * 25.4, 1)


def paper_on_lid(raw, facts):
    """Sleeve 0.5.57 - feit: wit papier op de witte klep van de scanner heeft geen zichtbare rand. Who's Your Momma boekje,
    pagina 3 en 4 (songteksten, wit tot aan de rand): het papier 252-254, de klep ±230, en de snede kwam 4-6 mm binnen de
    rand (293,9 en 296,2 mm tegen ±300) terwijl het als GOED gemeld werd. Is het papier vlak binnen de snede bijna even licht
    als de achtergrond vlak erbuiten (minder dan 12 L-punten verschil op een zijde), dan is de rand niet te zien: twijfel,
    met de reden. Met donker papier erachter is het verschil groot en blijft het stil. Geeft de reden of None."""
    box = (facts or {}).get('precise_mm') or (facts or {}).get('box_mm')
    try:
        dpi = float(json.load(open(raw + '.scans.json')).get('dpi') or 0)
    except (OSError, ValueError):
        dpi = 0
    if not box or len(box) < 4 or not dpi:
        return None
    im = cv2.imread(raw, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    k = 4; im = im[::k, ::k]; pm = dpi / 25.4 / k
    a_ = im[:, :, 3] > 0 if im.ndim == 3 and im.shape[2] == 4 else np.ones(im.shape[:2], bool)
    L = cv2.cvtColor(np.ascontiguousarray(im[:, :, :3]), cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) * 100 / 255
    x0, y0, w0, h0 = (v * pm for v in box[:4]); x1, y1 = x0 + w0, y0 + h0; H_, W_ = L.shape
    def band(xa, xb, ya, yb):
        xa, xb, ya, yb = (int(round(v)) for v in (max(0, xa), min(W_, xb), max(0, ya), min(H_, yb)))
        if xb - xa < 3 or yb - ya < 3:
            return None
        sel = a_[ya:yb, xa:xb]
        return float(np.median(L[ya:yb, xa:xb][sel])) if sel.sum() > 50 else None
    near, far = 3 * pm, 8 * pm; mx, my = .25 * w0, .25 * h0
    sides = {'left': (band(x0 + near, x0 + far, y0 + my, y1 - my), band(x0 - far, x0 - near, y0 + my, y1 - my)),
             'right': (band(x1 - far, x1 - near, y0 + my, y1 - my), band(x1 + near, x1 + far, y0 + my, y1 - my)),
             'top': (band(x0 + mx, x1 - mx, y0 + near, y0 + far), band(x0 + mx, x1 - mx, y0 - far, y0 - near)),
             'bottom': (band(x0 + mx, x1 - mx, y1 - far, y1 - near), band(x0 + mx, x1 - mx, y1 + near, y1 + far))}
    flat = [n for n, (pin, pout) in sides.items() if pin is not None and pout is not None and pin >= 85 and abs(pin - pout) < 12]
    if not flat:
        return None
    return ('white paper on the white scanner lid (' + ', '.join(flat) + ' edge): the paper edge cannot be seen clearly, so the '
            'size may be too small; scan it again with dark paper behind it')


def facts_levelled(crop):
    try:
        return bool(json.load(open(crop + '.feiten.json')).get('seam_levelled'))
    except (OSError, ValueError):
        return False


def label_route(a, decision):
    import scan_label
    import scan_rechtop

    result = {'part': a.part, 'scans': a.scans, 'verdict': 'TWIJFEL',
              'why': [], 'choices': [], 'decision': decision}
    if len(a.scans) > 1:
        # Sleeve Transparant: hetzelfde label meer keer gescand (Fake It 2021: Scan 9 en 10, beide label A): de scan met
        # het scherpste volledige label (zoals bij een kant: sharpest_label)
        pick = sharpest_label(a.scans, a.release_dpi)
        if pick:
            result['applied'] = f'{len(a.scans)} scans of this label; the sharpest is used ({os.path.basename(pick[0])})'
            a.scans = [pick[0]]
    if len(a.scans) != 1:
        result['why'] = ['a loose label needs exactly one complete label scan']
        print(json.dumps(result))
        return
    source = a.scans[0]
    dpi = scan_resolution(source, a.release_dpi)
    if not dpi:
        result['why'] = ['the label scan has no reliable resolution']
        print(json.dumps(result))
        return
    step('measuring the printed label edge and centre hole')
    image, facts = scan_label.extract(source, dpi)
    result.update(facts)
    if image is None:
        print(json.dumps(result))
        return
    grow = getattr(a, 'label_enlarged_mm', 0.0) or 0.0
    if grow > 0 and facts.get('diameter_mm'):
        # Feit: een label is zijn eigen maat (99-100 mm); de vergroting van 1 mm rondom is een retouche van de kant
        # (paste_label: het papier geschaald om het midden van het label). Een los label uit die kant (geen scan toont
        # het hele label: For Bitter Or Worse 2021 goud, 101,0 mm) krijgt die schaal terug, zodat het even groot is
        # als een label uit een losse labelscan.
        d = float(facts['diameter_mm']); k = (d - 2 * grow) / d
        image = cv2.resize(image, (max(1, round(image.shape[1] * k)), max(1, round(image.shape[0] * k))), interpolation=cv2.INTER_AREA)
        result['diameter_mm'] = round(d - 2 * grow, 2)
        result['applied'] = (result.get('applied', '') + f' label enlargement of the side undone ({d:.2f} -> {d - 2 * grow:.2f} mm): the label at its own size').strip()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.work, 'label.png')
    if not cv2.imwrite(raw, image):
        raise OSError('Could not save the label for text recognition')
    angle = decision.get('rotate')
    if angle not in (0, 90, 180, 270):
        try:
            words = release_words(a.discogs, disc=True) if a.discogs else []
        except Exception:
            words = []
        step('reading the tracklist on the loose label')
        angle, scores, hits = scan_rechtop.disc_angle(raw, words, dpi / 25.4)
        other = [value for degree, value in scores.items()
                 if abs(((degree - angle + 180) % 360) - 180) >= 30]
        enough = 1 if words else scan_rechtop.READABLE_MIN
        score = max(scores.values(), default=0)
        if any(h_.startswith('direction from the title') for h_ in hits):
            # Sleeve Transparant: geen tracklist, wel de leesrichting van artiest/titel (To Get Her Together 2022): die stand
            # toepassen, met twijfel (een titel kan schuin ontworpen zijn)
            result['why'].append('which way is up: set upright on the title (no tracklist on the label); a title can be printed at an angle')
        elif not score >= max(enough, scan_rechtop.CLEAR * max(other, default=1)) or 'title only' in hits:
            result['why'].append('which way is up: no label tracklist could be confirmed')
            angle = 0
        result['upright'] = {'rotate_clockwise': round(angle, 2), 'read_upright': hits}
    if angle:
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1)
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                                borderValue=(255, 255, 255, 0))
    if decision.get('crop') == 'keep':
        result['why'] = []
    result['verdict'] = 'TWIJFEL' if result['why'] else 'GOED'
    result['choices'] = ['keep', 'rotate90', 'rotate180', 'rotate270'] if result['why'] else []
    try:
        import scan_naam
        side_name = scan_naam.naam(a.discogs, 'side' + a.part[-1], a.variant, a.year) if a.discogs else ''
    except Exception:
        side_name = ''
    filename = (side_name + '_label') if side_name else Path(source).stem + '_uitgesneden'
    path = a.target or os.path.join(a.out, filename + '.png')
    if not a.target:
        counter = 2
        while os.path.exists(path):
            path = os.path.join(a.out, filename + f'_{counter}.png')
            counter += 1
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)).save(path, dpi=(dpi, dpi))
    result.update(path=path, dpi=dpi, size_mm=[round(image.shape[1] / dpi * 25.4, 1),
                                               round(image.shape[0] / dpi * 25.4, 1)])
    note_fast_reading(result)
    print(json.dumps(result, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--window-detail', help='complete detail scan of a framed cover opening, at the same dpi')
    p.add_argument('scans', nargs='+'); p.add_argument('--part', required=True)
    p.add_argument('--discogs', default=''); p.add_argument('--out', required=True); p.add_argument('--work', required=True)
    p.add_argument('--decision', default=''); p.add_argument('--variant', default=''); p.add_argument('--format', default=''); p.add_argument('--release-dpi', type=float, default=0.0); p.add_argument('--year', default='')
    p.add_argument('--seam-level', action='store_true')       # optie: toon langs de naden gelijktrekken (randen uit de gewone run)
    p.add_argument('--level-vinyl', action='store_true')     # optie: helderheid van effen licht vinyl gelijktrekken (scan_vlak)
    p.add_argument('--occurrence', type=int, default=1)    # Sleeve 0.5.53: het hoeveelste deel met deze naam in scanvolgorde (insert, insert_2, ...)
    p.add_argument('--label-enlarged-mm', type=float, default=0.0)   # los label uit een kant waarop het label al zoveel mm rondom vergroot is
    p.add_argument('--target', default='')                 # beslissing: precies dit bestand vervangen (niet een ander onderdeel met dezelfde naam)
    a = p.parse_args()
    folder = os.path.dirname(os.path.abspath(a.scans[0]))
    decision = json.loads(a.decision) if a.decision else remembered(folder, a.scans)
    if a.decision:
        save_decision(folder, a.scans, a.part, decision)
    if a.part == 'sticker':
        sticker_route(a, decision); return
    if a.part in ('labela', 'labelb', 'labelc', 'labeld'):
        label_route(a, decision); return
    import scan_feiten as SF
    fmt_note = None
    fmt = a.format
    if not fmt and a.discogs:
        try:
            fmt = release_format(a.discogs)
        except Exception as e:                             # Discogs niet bereikbaar: het vak bepaalt het formaat, met een melding
            fmt_note = f'the format (cd or vinyl) could not be read from Discogs ({e}); taken from the slot'
    fmt = fmt or ('cd' if a.part.startswith('cd') else 'vinyl')
    feit_part = a.part if a.part in SF.SHAPES[fmt] else ('insert' if a.part.startswith('insert') else 'front')
    result_fmt = fmt
    os.makedirs(a.work, exist_ok=True)
    raw = os.path.join(a.work, f'{a.part}_raw.png'); crop = os.path.join(a.work, f'{a.part}_crop.png')
    edge_raw = os.path.join(a.work, f'{a.part}_raw_edge.png'); edge_crop = os.path.join(a.work, f'{a.part}_crop_edge.png')
    edge_circle = cut_circle = None
    level_raw = os.path.join(a.work, f'{a.part}_raw_level.png'); levelled = False
    up = os.path.join(a.work, f'{a.part}_up.png')
    for f in (raw, raw + '.scans.json', level_raw, level_raw + '.scans.json', edge_raw, edge_raw + '.scans.json', edge_crop, edge_crop + '.feiten.json', crop, crop + '.feiten.json', up):   # nooit een oud tussenbestand gebruiken
        if os.path.exists(f):
            os.remove(f)
    result = {'part': a.part, 'scans': a.scans, 'verdict': 'GOED', 'why': [fmt_note] if fmt_note else [], 'choices': [], 'decision': decision}
    already = already_cut(a.scans[0], SF.SHAPES[fmt][feit_part], a.release_dpi,
                          VINYL_INSET_MM if fmt == 'vinyl' and feit_part.startswith('side') else 0.0) if len(a.scans) == 1 else None
    # Feit: alleen transparantie rondom bewijst dat een beeld al uitgesneden is. Een gewone scan
    # met de juiste buitenmaat kan nog deksel rondom hebben (wit hoesje op witte deksel, scheef
    # gelegd): eerst echt randen zoeken; vindt dat niets, dan pas 'al uitgesneden'.
    opaque_fallback = None
    label_pick = None                                       # Sleeve Transparant: scan met het scherpste label (één keer gezocht)
    if already and not already[1].get('transparent'):
        opaque_fallback, already = already, None
    if already:
        # Feit: dit beeld heeft al precies de maat van dit onderdeel: het is al uitgesneden.
        img0, note = already
        cv2.imwrite(crop, img0)
        json.dump({'verdict': 'GOED', 'why': [], 'dpi': note['dpi'], 'already_cut': note}, open(crop + '.feiten.json', 'w'))
        result['applied'] = f'already cut ({note["mm"][0]} x {note["mm"][1]} mm matches the facts); not cropped again' + (' (no resolution in the file: taken from the known size)' if note.get('dpi_from_size') else '')
        st = subprocess.CompletedProcess([], 0, '', '')
    elif len(a.scans) == 1:
        # één scan: niets samen te voegen; de scan zelf is het hele beeld
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        dpi0 = scan_resolution(a.scans[0], a.release_dpi)
        with Image.open(a.scans[0]) as im:
            w0, h0 = im.size
            im.convert('RGBA').save(raw)
        json.dump({'scale': 1, 'dpi': dpi0 or 300.0, 'scans': [{'scan': os.path.basename(a.scans[0]),
                   'corners': [[0, 0], [w0, 0], [w0, h0], [0, h0]]}]}, open(raw + '.scans.json', 'w'))
        st = subprocess.CompletedProcess([], 0, '', '')
        if not dpi0:
            result['why'].append('the scan has no resolution (dpi) stored; 300 dpi assumed for the size facts')
    else:
        scans_in = a.scans
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        total_px = 0
        for f in scans_in:
            with Image.open(f) as im:
                total_px += im.width * im.height
        # Vinyl sources are decoded one at a time. Admission is governed by
        # scan_stitch's actual decoder, preparation and scratch reservations,
        # rather than the sum of all original pixels. Keep the CD path intact.
        if total_px > 700e6 and fmt != 'vinyl':
            result.update(verdict='FOUT', why=[
                f'{len(scans_in)} scans ({total_px / 1e6:.0f} Mpx) exceed the full-resolution merge limit; '
                'no lower-resolution export was made'
            ])
            print(json.dumps(result, ensure_ascii=False))
            return
        step(f'merging {len(scans_in)} scans')
        is_disc = SF.SHAPES.get(fmt, SF.SHAPES['vinyl']).get(feit_part, ('rect',))[0] == 'disc'
        tone_on = os.environ.get('SLEEVE_TONE', '1') != '0'                 # 0 = uit (voor vergelijken)
        os.environ['SLEEVE_TONE_KEY'] = f'{feit_part}|{fmt}' if tone_on else 'off'   # het onderdeel hoort bij de sleutel van de tussenopslag
        tone_args = ['--toon-onderdeel', feit_part, '--toon-format', fmt] if tone_on else []    # feit: de grove toon alleen op het onderdeel zelf (scan_stitch.tone_support)
        if is_disc:
            # Optie A (doorschijnend vinyl): per positie een scan met de witte klep (de echte kleur) en een met een vel op
            # de plaat (daar is de rand te zien). Feit per scan: de achtergrond langs de scanrand, wit of gekleurd.
            kinds = {f: scan_background(f) for f in scans_in}
            lids = [f for f in scans_in if kinds[f] == 'lid']; sheets = [f for f in scans_in if kinds[f] == 'sheet']
            # Sleeve Transparant: een losse labelscan (hooguit 140 mm, alleen het label met donker vinyl eromheen) is geen
            # scan met een vel op de plaat (FBOW 2009 kant A: drie labelscans maakten van zwart vinyl 'optie A')
            import scan_dpi as _sd
            def _small(f):
                d_, s_ = _sd.read(f), _sd.size_px(f)
                return bool(d_ and s_ and max(s_) / d_ * 25.4 <= 140)
            sheets = [f for f in sheets if not _small(f)]
            if len(lids) >= 2 and len(sheets) >= 2:
                step(f'option A: finding the edge on the {len(sheets)} scans with a sheet on top')
                se = subprocess.run([PY, os.path.join(HERE, 'scan_stitch.py'), edge_raw] + sheets + ['--vorm', 'ruw', '--plaat-onderdeel'], capture_output=True, text=True)
                if se.returncode == 0 and os.path.exists(edge_raw):
                    subprocess.run([PY, os.path.join(HERE, 'scan_feiten.py'), edge_raw, feit_part, '--format', fmt, '--out', edge_crop], capture_output=True, text=True)
                    try:
                        ef = json.load(open(edge_crop + '.feiten.json'))
                    except (OSError, ValueError):
                        ef = {}
                    if ef.get('cut_circle_px') and ef.get('verdict') != 'FOUT':
                        edge_circle = ef['cut_circle_px']; scans_in = lids
                        result['applied'] = (result.get('applied', '') + f' option A: colour from {len(lids)} scans with the white lid, edge from {len(sheets)} scans with a sheet on top').strip()
                if edge_circle is None:
                    result['why'].append('option A: the edge could not be measured on the scans with a sheet on top; all scans were merged together')
        cached = stitch_cache_get(a.scans, scans_in != a.scans, is_disc, raw) if edge_circle is None else None
        if cached is not None:
            st = cached                                     # alleen in de vaste toets (SLEEVE_STITCH_CACHE)
        else:
            light_args = ['--lichtkolom', os.environ['SLEEVE_LICHTKOLOM']] if is_disc and os.environ.get('SLEEVE_LICHTKOLOM') else []   # Sleeve Transparant: lichtcorrectie per scannerkolom (alleen plaat)
            # Sleeve Transparant (knop aan): de belichting tussen de deelscans niet gelijktrekken (--geen-kleur). Bij doorzichtig
            # vinyl schijnt de ondergrond door en maakte het gelijktrekken een hele deelscan donkerder (GF groen 0001: -5 L,
            # 20 nov). Alleen de scan met het scherpste label (vaak een labelscan uit een andere sessie) wordt nog gelijkgetrokken:
            # die levert het smalle stuk vinyl naast de labelrand (anders een licht reepje, GF magenta B).
            if light_args:
                light_args = light_args + ['--geen-kleur']
                if fmt == 'vinyl' and feit_part.startswith('side') and len(a.scans) > 1:
                    try:
                        label_pick = sharpest_label(a.scans, a.release_dpi)
                    except Exception:
                        label_pick = None
                    if label_pick:
                        light_args += ['--belichting-alleen', os.path.basename(label_pick[0]),
                                       '--niet-op-vinyl', os.path.basename(label_pick[0])]   # de labelscan levert het label, niet het vinyl
            st = subprocess.run([PY, os.path.join(HERE, 'scan_stitch.py'), raw] + scans_in + ['--vorm', 'ruw'] + (['--plaat-onderdeel'] if is_disc else []) + tone_args + light_args, capture_output=True, text=True)
            if st.returncode == 0 and edge_circle is None:
                stitch_cache_put(a.scans, scans_in != a.scans, is_disc, raw, st.stdout)
        if edge_circle is not None and st.returncode == 0:
            # de snede van de vel-versie op de klep-versie leggen: beide tonen dezelfde plaat en hetzelfde label
            mapped = map_circle(edge_raw, raw, edge_circle)
            if mapped is None:
                result['why'].append('option A: the two merges could not be laid on each other; cut on the scans with the white lid')
            else:
                cut_circle = mapped
        if a.seam_level and st.returncode == 0:
            # Optie: nog een keer samenvoegen met de toon langs de naden gelijkgetrokken. De randen worden op de
            # gewone versie gezocht (gelijktrekken verschoof eerder de gevonden rand: Queen-insert 1,2 mm); alleen
            # de pixels komen uit deze versie, als die precies op hetzelfde canvas ligt.
            step('merging again with the tone levelled across the seams')
            lv = subprocess.run([PY, os.path.join(HERE, 'scan_stitch.py'), level_raw] + scans_in + ['--vorm', 'ruw'] + (['--plaat-onderdeel'] if is_disc else []) + tone_args + light_args,
                                capture_output=True, text=True, env=dict(os.environ, SLEEVE_SEAM_LEVEL='1'))
            try:
                same = lv.returncode == 0 and json.load(open(raw + '.scans.json'))['scans'] == json.load(open(level_raw + '.scans.json'))['scans']
            except (OSError, ValueError, KeyError):
                same = False
            if same:
                levelled = True
            else:
                result['why'].append('the tone could not be levelled across the seams (the second merge did not line up exactly); made without it')
    # Wat het samenvoegen over losse deelscans meldt, hoort de gebruiker te weten (nooit stil weglaten).
    for line in (st.stdout or '').splitlines():
        if line.startswith('SEAM_DETAILS: '):
            path = line.removeprefix('SEAM_DETAILS: ')
            if os.path.isdir(path):
                result['seamDetails'] = path
        elif line.startswith('PAPER_WHITE_CHECK: '):
            result['paper_white_check'] = json.loads(line.removeprefix('PAPER_WHITE_CHECK: '))
        elif line.startswith('PAPER_WHITE: '):
            result['paper_white'] = json.loads(line.removeprefix('PAPER_WHITE: '))
            result['applied'] = (result.get('applied', '') + ' preserved source-observed white paper').strip()
        elif line.startswith('SEAM_CHECK: '):
            result.setdefault('source_notes', []).append(line.removeprefix('SEAM_CHECK: '))
            print(line, file=sys.stderr, flush=True)
        elif line.startswith('NOTE: left out'):
            name = line.rsplit(': ', 1)[-1]
            result['why'].append(f'{name} was left out: it shares no print with the other scans and its paper edges are not visible. '
                                 'Scan that corner with a contrasting sheet behind it.')
        elif line.startswith('NOTE: cover line changes at a seam:'):
            result['why'].append(line[6:])
        elif line.startswith(('NOTE: vinyl spots replaced from overlapping scans:',
                              'NOTE: cover lines restored from overlapping scans:')):
            result['applied'] = (result.get('applied', '') + ' ' + line[6:]).strip()
        elif line.startswith('NOTE: placed on its paper edges') or line.startswith('NOTE: aligned on faint print'):
            result['applied'] = (result.get('applied', '') + ' ' + line[6:]).strip()
        elif line.startswith(('label scan not used on the vinyl', 'label scan lies like')):
            result['applied'] = (result.get('applied', '') + ' ' + line).strip()
        elif line.startswith('lichtkolom:'):                # Sleeve Transparant: lichtcorrectie altijd in het logboek
            result.setdefault('light_correction', []).append(line)
            print(line, file=sys.stderr, flush=True)
    if not already and st.returncode == 0 and os.path.exists(raw + '.scans.json') and a.release_dpi:
        meta = json.load(open(raw + '.scans.json'))
        if not meta.get('dpi'):                            # samengevoegd zonder betrouwbare resolutie: die van de release
            meta['dpi'] = a.release_dpi; json.dump(meta, open(raw + '.scans.json', 'w'))
    if not already and (st.returncode != 0 or not os.path.exists(raw)):
        result.update(verdict='FOUT', why=[(st.stderr or st.stdout).strip().splitlines()[-1] if (st.stderr or st.stdout).strip() else 'merging failed'])
        print(json.dumps(result)); return
    if not already and fmt == 'vinyl' and feit_part.startswith('side') and len(a.scans) > 1:
        # Sleeve Transparant (alle vinyl): het hele label uit de scan met het scherpste volledige label (paste_label)
        step('placing the sharpest label')
        try:
            pick = label_pick or sharpest_label(a.scans, a.release_dpi)
        except Exception as e:                          # nooit de hele kant laten mislukken om het label
            pick = None; print(f'labelplaatsing: label search failed ({e})', file=sys.stderr, flush=True)
        if pick is None:
            result['applied'] = (result.get('applied', '') + ' no complete label found in a single scan; label as merged').strip()
        else:
            lpath, lab, lfacts, ldpi, lscores = pick[:5]
            lothers = (pick[5] if len(pick) > 5 else []) if not os.environ.get('SLEEVE_LABEL_ONE_SCAN') else []   # uit: alleen de labelscan (vergelijken)
            if lothers:
                step('combining the sharpest parts of the label')
            for target in [raw] + ([level_raw] if levelled else []):
                try:
                    note, err = paste_label(target, lpath, ldpi, lothers, label_data=(lab, lfacts))
                except Exception as e:                  # nooit de hele kant laten mislukken om het label
                    note, err = None, f'error while placing ({e})'
                if err:
                    break
            lab = None
            msg = (f'whole label from {os.path.basename(lpath)}: ' + (note if not err else f'not placed ({err}); label as merged')
                   + ' (label sharpness and size: ' + ', '.join(f'{n_} {s_} {d_} mm' for n_, s_, d_ in lscores) + ')')
            result['applied'] = (result.get('applied', '') + ' ' + msg).strip()
            if err:
                result['why'].append(f'the sharpest label ({os.path.basename(lpath)}) could not be placed: {err}')
            print('labelplaatsing: ' + msg, file=sys.stderr, flush=True)
    if (fmt == 'vinyl' and feit_part.startswith('side')
            and os.environ.get('SLEEVE_LABEL_EXPAND_MM') == '1'
            and 'user retouch: paper label and printing enlarged' not in result.get('applied', '')):
        result['why'].append('label enlargement requested but not applied: no independently confirmed paper mask was placed')
    if result.get('light_correction'):
        skipped = [x for x in result['light_correction'] if 'niet gecorrigeerd' in x]
        result['applied'] = (result.get('applied', '') + f' light correction per scanner column ({len(a.scans) - len(skipped)} of {len(a.scans)} scans'
                             + (f'; not on {", ".join(x.split()[1] for x in skipped)}: other glass' if skipped else '') + ')'
                             + ' exposure of the partial scans kept as scanned (transparent vinyl)').strip()
    if not already:
        step('finding the edges and cutting out')
    fe = subprocess.run([PY, os.path.join(HERE, 'scan_feiten.py'), raw, feit_part, '--format', fmt, '--out', crop] + (['--pixels', level_raw] if levelled else [])
                        + (['--cut-circle', ','.join(f'{v:.2f}' for v in cut_circle)] if cut_circle else []), capture_output=True, text=True) if not already else subprocess.CompletedProcess([], 0, '', '')
    if levelled and facts_levelled(crop):
        result['applied'] = (result.get('applied', '') + ' tone levelled across the seams').strip()
    try:
        facts = json.load(open(crop + '.feiten.json'))
    except (OSError, ValueError):
        facts = None
    white_on_white = paper_on_lid(raw, facts) if (fmt == 'vinyl' and feit_part == 'insert' and result.get('paper_white')
                                                  and facts and facts.get('verdict') != 'FOUT') else None
    if white_on_white:
        result['why'].append(white_on_white)
    # Feit: een cd-onderdeel heeft de maat van zijn verpakking. Past het niet bij de jewel case
    # (bv. de 'back' is een vierkant kaartje zonder zijflappen, zoals bij een promo), dan bepaalt
    # de gemeten vorm welke verpakking het is: slim jewel (kaartje ±121), kartonnen hoesje (±127),
    # digipak. En: de echte rand is de buitenste. Een drukrand binnen een groter kaartje (Places
    # To Go: geel kader om een oranje vlak) kan toevallig op de jewel-maat passen; vindt een andere
    # cd-maat een rand die er duidelijk omheen ligt, dan is dat de echte rand.
    def box_of(f):
        b = (f or {}).get('precise_mm') or (f or {}).get('box_mm')
        return (b[2], b[3]) if b and len(b) >= 4 else None
    if not already and fmt == 'cd' and SF.SHAPES['cd'][feit_part][0] == 'rect':
        first = box_of(facts) if facts and facts.get('verdict') != 'FOUT' else None
        # vierkante delen zoeken al zelf tot 131 mm (buitenste rand); alleen als niets gevonden is
        # de andere maten proberen. Niet-vierkante delen (achterkaart 150 mm) wel altijd vergelijken.
        if first and SF.SHAPES['cd'][feit_part][3]:
            first = 'keep'
        if first and first != 'keep':
            os.replace(crop, crop + '.jewel.png'); os.replace(crop + '.feiten.json', crop + '.jewel.json')
        chosen = None
        for alt in ([] if first == 'keep' else ('cd-slimjewel', 'cd-cardsleeve', 'cd-digipak')):
            if feit_part not in SF.SHAPES[alt] or SF.SHAPES[alt][feit_part] == SF.SHAPES['cd'].get(feit_part):
                continue
            step('measuring against other CD packagings')
            subprocess.run([PY, os.path.join(HERE, 'scan_feiten.py'), raw, feit_part, '--format', alt, '--out', crop], capture_output=True, text=True)
            try:
                f2 = json.load(open(crop + '.feiten.json'))
            except (OSError, ValueError):
                continue
            b2 = box_of(f2)
            outside = first is None or (b2 is not None and min(b2) >= min(first) + 2 and max(b2) >= max(first) + 2)
            if f2.get('verdict') != 'FOUT' and outside:
                chosen = alt; facts, fmt = f2, alt
                label = {'cd-slimjewel': 'a square card of a slim jewel case (no side flaps)', 'cd-cardsleeve': 'a cardboard sleeve or larger card',
                         'cd-digipak': 'a digipak panel'}[alt]
                result['applied'] = (result.get('applied', '') + f' measured as {label}, not a jewel case part').strip()
                break
        if first and first != 'keep':
            if chosen is None:                             # de jewel-maat bleef de buitenste echte rand
                os.replace(crop + '.jewel.png', crop); os.replace(crop + '.jewel.json', crop + '.feiten.json')
                facts = json.load(open(crop + '.feiten.json'))
            else:
                os.remove(crop + '.jewel.png'); os.remove(crop + '.jewel.json')
    def fits_cd(wh):
        # past deze maat op een cd-verpakking voor dit onderdeel (±1 mm)?
        for f in ('cd', 'cd-slimjewel', 'cd-cardsleeve', 'cd-digipak'):
            sp = SF.SHAPES[f].get(feit_part)
            if sp and sp[0] == 'rect':
                (wl, wh_), (hl, hh) = sp[1], sp[2]
                W, H = wh
                if (wl - 1 <= W <= wh_ + 1 and hl - 1 <= H <= hh + 1) or (wl - 1 <= H <= wh_ + 1 and hl - 1 <= W <= hh + 1):
                    return True
        return False
    if opaque_fallback is None and len(a.scans) == 1 and fmt.startswith('cd') and SF.SHAPES['cd'].get(feit_part, ('',))[0] == 'rect':
        # het hele beeld tegen alle cd-maten (niet alleen de gekozen verpakking)
        for f in ('cd', 'cd-slimjewel', 'cd-cardsleeve', 'cd-digipak'):
            if feit_part in SF.SHAPES[f]:
                got = already_cut(a.scans[0], SF.SHAPES[f][feit_part], a.release_dpi)
                if got and not got[1].get('transparent'):
                    opaque_fallback = got; break
    if opaque_fallback is None and len(a.scans) == 1 and fmt == 'vinyl' and feit_part == 'insert':
        # Sleeve Transparant: één scan die in één richting duidelijk kleiner is dan het glas (216 x 297 mm; een volle
        # glasscan is precies dat): de scanner heeft hem bijgesneden tot het vel zelf (A4-kaartje bij Set This Thing On
        # Fire: 208 x 295 mm). Vindt het snijden geen rand, dan is de scan het vel: niets meer afsnijden.
        got = already_cut(a.scans[0], ('rect', (20, 1000), (20, 1000), False), a.release_dpi)
        if got and not got[1].get('transparent'):
            mm_ = sorted(got[1]['mm'])
            if mm_[1] <= 299 and (mm_[0] <= 211 or mm_[1] <= 292):
                opaque_fallback = got
    # Feit: past de gevonden snede op geen enkele cd-maat, maar het hele beeld wel, dan is het beeld
    # al uitgesneden (wit kaartje tot de beeldrand: Dominique-achterkant 121 mm, snede op 115 mm).
    if (opaque_fallback and fmt.startswith('cd') and facts and facts.get('verdict') != 'FOUT'
            and box_of(facts) and not fits_cd(box_of(facts))):
        facts = {'verdict': 'FOUT'}
    # Feit: buiten het kaartje ligt de deksel (effen en licht). Ligt er tussen de gevonden snede en
    # de beeldrand nog drukwerk, dan loopt het kaartje door tot de beeldrand: het beeld was al
    # uitgesneden (Deena-boekje 119,7 mm: de snede viel in de foto).
    if (opaque_fallback and fmt.startswith('cd') and facts and facts.get('verdict') != 'FOUT'
            and (facts.get('precise_mm') or facts.get('box_mm')) and print_outside(raw, facts)):
        facts = {'verdict': 'FOUT'}
    if opaque_fallback and (facts is None or facts.get('verdict') == 'FOUT'):
        # geen rand gevonden en de hele scan heeft al de maat van dit onderdeel: al uitgesneden
        img0, note = opaque_fallback
        cv2.imwrite(crop, img0)
        facts = {'verdict': 'GOED', 'why': [], 'dpi': note['dpi'], 'already_cut': note}
        json.dump(facts, open(crop + '.feiten.json', 'w'))
        result['applied'] = (f'the scanner cut this scan to the sheet itself ({note["mm"][0]} x {note["mm"][1]} mm, no edge found inside); not cropped again'
                             if fmt == 'vinyl' and feit_part == 'insert' else
                             f'already cut ({note["mm"][0]} x {note["mm"][1]} mm matches the facts, no edge found inside); not cropped again')
    if facts is None:
        result.update(verdict='FOUT', why=[fe.stderr.strip().splitlines()[-1] if fe.stderr.strip() else 'cropping failed'])
        print(json.dumps(result)); return
    if facts.get('verdict') == 'FOUT' and a.part == 'inside' and fmt == 'vinyl' and not already:
        # Feit: een opengeklapte hoes is twee panelen (±62 x 31 cm). Past dat niet in deze scans, maar een
        # los vel wel, dan is het een los vel: de gemeten vorm bepaalt het (zoals bij cd-verpakkingen).
        step('no opened gatefold in these scans; measuring as a single sheet')
        subprocess.run([PY, os.path.join(HERE, 'scan_feiten.py'), raw, 'insert', '--format', fmt, '--out', crop], capture_output=True, text=True)
        try:
            sheet = json.load(open(crop + '.feiten.json'))
        except (OSError, ValueError):
            sheet = None
        if sheet and sheet.get('verdict') != 'FOUT':
            facts = sheet; feit_part = 'insert'
            result['applied'] = (result.get('applied', '') + ' measured as a single sheet, not an opened gatefold (for a loose sheet or booklet page use the Insert slot)').strip()
        else:
            facts = {'verdict': 'FOUT', 'why': ['this is not an opened gatefold sleeve (two panels, about 62 x 31 cm), and no single sheet was found either. '
                                               'Inside is only for a gatefold; put a loose sheet or a booklet page in an Insert slot, with all scans of that one side.']}
    if facts.get('verdict') == 'FOUT':
        result.update(verdict='FOUT', why=facts.get('why', ['not found']))
        print(json.dumps(result)); return
    dpi = facts.get('dpi') or 300.0
    why = result['why'] + list(facts.get('why', []))
    if facts.get('message'):
        result['message'] = facts['message']
    img = cv2.imread(crop, cv2.IMREAD_UNCHANGED)
    render_crop = crop
    if (not already and fmt == 'vinyl' and img is not None and cut_circle is None
            and facts.get('cut_circle_px') and decision.get('crop') != 'keep'):
        import scan_rim
        try:
            refinements = scan_rim.refine(img, facts, raw + '.scans.json', scans_in)
        except (OSError, ValueError, cv2.error):
            refinements = []
        if refinements:
            result['rim_sources'] = refinements
            result['applied'] = (result.get('applied', '') + ' record rim improved from overlapping scans').strip()
            why = [note for note in why if not note.startswith('record tone near the cut:')]
            audit = {'verdict': 'GOED', 'why': []}
            SF.report_vinyl_edge_tone(audit, img, dpi / 25.4, True)
            why.extend(audit['why'])
            render_crop = os.path.join(a.work, f'{a.part}_rim.png')
            if not cv2.imwrite(render_crop, img):
                raise OSError('Could not save the refined record rim')
    if not already and fmt.startswith('cd') and SF.SHAPES[fmt][feit_part][0] == 'rect' and img is not None:
        better = colour_extent(raw, dpi, img, fmt, feit_part)
        if better is not None:
            img, mm = better
            cv2.imwrite(crop, img)
            why = [w for w in why if not w.startswith('not square')]
            result['applied'] = (result.get('applied', '') + f' cut on the outer edge of the coloured card ({mm[0]:.1f} x {mm[1]:.1f} mm): the found edge lay inside the print').strip()
    if why and decision.get('crop') == 'square' and 'sides' in facts:
        img, weak = square_up(img, facts['sides'])
        why = [w for w in why if not w.startswith('not square')]
        result['applied'] = f'square ({weak} side moved in)'
        cv2.imwrite(crop, img)
    # Feit: de gemeten maat moet bij de verpakking passen (de randen kunnen na het grove zoeken
    # nog naar binnen schuiven). Past hij niet, dan klopt de verpakkingskeuze of de rand niet.
    spec = SF.SHAPES[fmt][feit_part]
    if spec[0] == 'rect' and img is not None:
        wm, hm = img.shape[1] / dpi * 25.4, img.shape[0] / dpi * 25.4
        (wl, wh), (hl, hh) = spec[1], spec[2]
        fits = lambda W, H: wl - 1 <= W <= wh + 1 and hl - 1 <= H <= hh + 1
        if not (fits(wm, hm) or fits(hm, wm)) and not (isinstance(facts, dict) and (facts.get('loose_sheet_mm') or (fmt == 'vinyl' and feit_part == 'insert' and facts.get('already_cut')))):
            # (een los vel van zijn eigen maat, gemeten in scan_feiten, hoeft niet de standaardmaat te hebben)
            why.append(f'measured {wm:.1f} x {hm:.1f} mm, but this packaging is {wl}-{wh} x {hl}-{hh} mm: check the packaging choice')
    # rechtop
    check_shape = img.shape
    check_rotation = 0
    words_note = None
    disc = SF.SHAPES[fmt][feit_part][0] == 'disc'
    try:
        words = release_words(a.discogs, disc=disc) if a.discogs else []
    except Exception as e:                                 # het uitsnijden is klaar: dat gooien we niet weg om een haperende verbinding
        words, words_note = [], f'which way is up: the text of the release could not be fetched from Discogs ({e})'
    if True:                                               # ook zonder Discogs-tekst: dan bepaalt de tekst die Sleeve zelf leest de stand
        args = [PY, os.path.join(HERE, 'scan_rechtop.py'), crop, '--words', os.path.join(a.work, 'words.json'), '--out', up]
        json.dump(words, open(os.path.join(a.work, 'words.json'), 'w'))
        if disc:
            args += ['--disc', '--dpi', str(dpi)] + (['--lanczos'] if fmt == 'vinyl' else [])   # Sleeve Transparant: scherper draaien (vinyl)
        if render_crop != crop:
            args += ['--pixels', render_crop]
        import scan_rechtop as _R
        step('macOS text recognition is not answering: setting it upright with the Discogs photo' if _R.ocr_unavailable()
             else 'reading text to set it upright (macOS text recognition)' + (' and straightening the label' if disc else ''))
        ro = subprocess.run(args, capture_output=True, text=True)
        try:
            upright = json.loads(ro.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            upright = {'verdict': 'TWIJFEL'}
        result['upright'] = {k: upright.get(k) for k in ('rotate_clockwise', 'verdict', 'read_upright')}
        turned = cv2.imread(up, cv2.IMREAD_UNCHANGED) if os.path.exists(up) else None
        if upright.get('verdict') == 'GOED':
            if turned is not None:
                img = turned
                check_rotation = upright.get('rotate_clockwise')
        else:
            # Geen duidelijke stand: Sleeve gokt niet. Het beeld blijft zoals het uitgesneden is, behalve
            # een label dat op de titel recht gezet is (dat is gemeten, alleen niet zeker): dat zie jij
            # recht gezet. Jouw draaiing geldt voor precies het beeld dat jij zag.
            base = cv2.imread(render_crop, cv2.IMREAD_UNCHANGED)
            title_only = 'title only' in (upright.get('read_upright') or [])
            shown = turned if (title_only and turned is not None) else base
            img = shown
            check_rotation = upright.get('rotate_clockwise') if title_only and turned is not None else 0
            if not title_only:
                result['upright']['rotate_clockwise'] = 0   # er is niets gedraaid
            if decision.get('rotate') in ROTATE or decision.get('rotate') == 0:
                img = cv2.rotate(shown, ROTATE[decision['rotate']]) if decision['rotate'] else shown
                check_rotation = (check_rotation or 0) + decision['rotate']
                result['applied'] = (result.get('applied', '') + f' rotated {decision["rotate"]}').strip()
            elif title_only:
                why.append('which way is up: the label was set straight on the title only (no tracklist read); a title can be printed at an angle')
            else:
                # geen bekende tekst: de Discogs-foto van deze release bepaalt de stand
                orientation_source = cv2.imread(crop, cv2.IMREAD_UNCHANGED) if render_crop != crop else base
                got = discogs_orientation(orientation_source, a.discogs, disc) if a.discogs else None
                if got:
                    ang_d, nk = got
                    check_rotation = ang_d
                    if disc:
                        hh, ww = base.shape[:2]
                        Rm = cv2.getRotationMatrix2D((ww / 2, hh / 2), -ang_d, 1.0)
                        img = cv2.warpAffine(base, Rm, (ww, hh), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
                    else:
                        img = cv2.rotate(base, ROTATE[int(ang_d)]) if int(ang_d) else base
                    result['upright'] = {'rotate_clockwise': round(ang_d, 1), 'verdict': 'GOED', 'read_upright': [f'Discogs photo ({nk} matching points)']}
                    result['applied'] = (result.get('applied', '') + ' upright from the Discogs photo').strip()
                elif upright.get('verdict') == 'LEESBAAR':
                    # feit: andere tekst op dit onderdeel leest duidelijk in één stand
                    if turned is not None:
                        img = turned
                        check_rotation = upright.get('rotate_clockwise')
                    result['upright'] = {'rotate_clockwise': upright.get('rotate_clockwise'), 'verdict': 'GOED', 'read_upright': upright.get('read_upright')}
                    result['applied'] = (result.get('applied', '') + ' upright from readable text').strip()
                elif not disc and (turn := face_orientation(base)) is not None:
                    # feit: een gezicht staat rechtop (GF boekje: een paginagroot gezicht, geen tekst, geen Discogs-foto)
                    img = cv2.rotate(base, ROTATE[turn]) if turn else base
                    check_rotation = turn
                    result['upright'] = {'rotate_clockwise': turn, 'verdict': 'GOED', 'read_upright': ['face (upright)']}
                    result['applied'] = (result.get('applied', '') + ' upright from a face').strip()
                else:
                    if disc and 'level: none' in (upright.get('read_upright') or []):
                        why.append('which way is up: label text was read, but no track rows could be levelled')
                    else:
                        why.append('which way is up: no text could be read clearly' + (', and no Discogs photo matches clearly' if a.discogs else ''))
    elif decision.get('rotate') in ROTATE or decision.get('rotate') == 0:
        # geen tekst om te lezen, maar jij hebt gezegd hoe het moet: dat geldt (0.5.29 negeerde dit)
        if decision['rotate']:
            img = cv2.rotate(img, ROTATE[decision['rotate']])
            check_rotation = decision['rotate']
        result['applied'] = (result.get('applied', '') + f' rotated {decision["rotate"]}').strip()
    else:
        why.append(words_note or 'which way is up: no Discogs release, so no known text to read')
    # kant op het label (feit) tegen het vak
    if disc and feit_part.startswith('side'):
        letter = side_on_label(img, dpi)
        result['side_on_label'] = letter
        if letter and a.part.startswith('side') and a.part != 'side' + letter:
            why.append(f'the label says side {letter.upper()}, the slot says {a.part[4:].upper()}')
    if (not already and not facts.get('already_cut') and fmt == 'vinyl'
            and feit_part.startswith('side') and disc and cut_circle is None
            and decision.get('crop') != 'keep' and img is not None):
        trimmed = inset_vinyl(img, dpi, facts.get('cut_diameter_mm'))
        if trimmed is not None:
            img = trimmed
            result['vinyl_inset_mm'] = VINYL_INSET_MM
            result['applied'] = (result.get('applied', '') + ' vinyl rim inset 2 mm').strip()
            measured_radius = facts.get('precise_diameter_mm', 0) / 2
            cut_radius = facts.get('cut_diameter_mm', 0) / 2 - VINYL_INSET_MM
            deviation = facts.get('precise_inward_mm', facts.get('precise_dev_mm'))   # Sleeve Transparant: alleen afwijking naar binnen telt
            if (deviation is not None and measured_radius > 0 and cut_radius > 0
                    and measured_radius - float(deviation) >= cut_radius + .2):
                removed = [note for note in why if re.match(r'^\d+ directions more than .* off the circle:', note)]
                why = [note for note in why if note not in removed]
                if removed:
                    result['source_edge_notes'] = removed
                    result['applied'] += ' measured rim deviations fall outside the exported cut'
            # Inspect the exported edge; the old reported edge has been removed.
            why = [note for note in why if not note.startswith('record tone near the cut:')]
            audit = {'verdict': 'GOED', 'why': []}
            SF.report_vinyl_edge_tone(audit, img, dpi / 25.4, True)
            why.extend(audit['why'])
    check_trim_left = check_trim_top = 0
    if (fmt == 'vinyl' and feit_part == 'insert' and img is not None
            and not already and not facts.get('already_cut') and decision.get('crop') != 'keep'):
        _, proposed_cut = SF.trim_white_insert_sides(img, dpi / 25.4)
        print_boxes = []
        if any(proposed_cut):
            import scan_rechtop as _margin_text
            lines = _margin_text.faint_known_lines(img, words)
            read_scale = min(1., 1600 / max(img.shape[:2]))
            print_boxes = [tuple(float(v) / read_scale for v in line['box']) for line in lines]
        img, side_cut = SF.trim_white_insert_sides(img, dpi / 25.4, print_boxes)
        if side_cut != proposed_cut:
            result['margin_print_protected_px'] = list(proposed_cut)
            result['margin_print_evidence'] = [{'text': line['text'], 'box_native': list(box)}
                                               for line, box in zip(lines, print_boxes)]
        check_trim_left += side_cut[0]
        if any(side_cut):
            result['white_margin_trimmed_px'] = list(side_cut)
            result['applied'] = (result.get('applied', '') + ' trimmed shadow along white paper').strip()
        if SF.level_white_insert_sides(img, dpi / 25.4):
            result['applied'] = (result.get('applied', '') + ' levelled light against nearby blank paper').strip()
        # The old warning describes the raw orientation and untreated margins.
        # Inspect the actual upright, trimmed and locally levelled paper again.
        why = [note for note in why if not note.startswith('paper tone near the cut:')]
        paper_audit = {'verdict': 'GOED', 'why': []}
        SF.report_vinyl_edge_tone(paper_audit, img, dpi / 25.4, False)
        why.extend(paper_audit['why'])
        if paper_audit.get('applied'):
            result['applied'] = (result.get('applied', '') + ' ' + paper_audit['applied']).strip()
    if (fmt == 'vinyl' and a.part == 'front' and img is not None
            and not already and not facts.get('already_cut') and decision.get('crop') != 'keep'
            and os.environ.get('SLEEVE_COVER_SIDE_SHADOW', '1') == '1'):
        img, side_cut = SF.trim_cover_sides(img, dpi / 25.4)
        check_trim_left += side_cut[0]
        if any(side_cut):
            result['cover_side_trimmed_px'] = list(side_cut)
            result['applied'] = (result.get('applied', '') + ' trimmed measured strips along the cover sides').strip()
    if (fmt == 'vinyl' and a.part in ('front', 'back') and img is not None and img.ndim == 3
            and not already and not facts.get('already_cut') and decision.get('crop') != 'keep'
            and os.environ.get('SLEEVE_EDGE_LINE', '1') == '1'):
        img, line_cut = SF.trim_dark_edge_lines(img, dpi / 25.4)
        check_trim_top += line_cut[0]
        check_trim_left += line_cut[3]
        if any(line_cut):
            result['edge_line_trimmed_px'] = list(line_cut)
            mm_ = ', '.join(f'{n} {v / (dpi / 25.4):.2f} mm' for n, v in zip(('top', 'right', 'bottom', 'left'), line_cut) if v)
            result['applied'] = (result.get('applied', '') + f' thin dark line along the cut trimmed ({mm_})').strip()
    if a.window_detail and a.part == 'front' and fmt == 'vinyl':
        import scan_window_detail
        try:
            img, detail = scan_window_detail.merge_detail(img, a.window_detail, dpi)
            result['window_detail'] = detail
            result['applied'] = (result.get('applied', '') + ' opening restored from the complete detail scan').strip()
        except ValueError as error:
            why.append('detail scan: ' + str(error))
    if (fmt == 'vinyl' and not disc and result.get('seamDetails')
            and facts.get('raw_to_crop_px') and img is not None):
        import scan_final_check
        try:
            geometry = scan_final_check.quarter_turn_matrix(check_shape, check_rotation)
            geometry = geometry @ np.vstack([facts['raw_to_crop_px'], [0, 0, 1]])
            geometry[0, 2] -= check_trim_left
            geometry[1, 2] -= check_trim_top
            why, check = scan_final_check.recheck(img, result['seamDetails'], geometry,
                                                why, result.get('window_detail'))
            if check is not None:
                result['final_source_check'] = check
                if check.get('details'):
                    result['seamDetails'] = check['details']
        except (OSError, ValueError, KeyError, IndexError, TypeError, cv2.error) as error:
            result['final_source_check'] = {'unavailable': str(error)}
    if why and decision.get('crop') == 'keep':
        result['applied'] = (result.get('applied', '') + ' kept as it is').strip()
        why = []                                           # jouw beslissing 'zo laten' = klaar
    if why:
        result['verdict'] = 'TWIJFEL'; result['why'] = why; result['choices'] = choices_for(why)
        if any(w.startswith('not scanned:') for w in why):
            result['verdict'] = 'FOUT'                     # Sleeve Transparant: een stuk ontbreekt op de scans: opnieuw scannen
        if any(w.startswith('which way is up') for w in why):
            result['choices'] += ['rotate90', 'rotate180', 'rotate270']
    # uitklap: in twee panelen splitsen (feit: twee gelijke helften naast elkaar)
    panels = [(a.part, img)]
    if a.part == 'inside' and fmt == 'vinyl' and img.shape[1] > 1.7 * img.shape[0]:
        half = img.shape[1] // 2
        panels = [('insert_left', img[:, :half]), ('insert_right', img[:, half:])]
    elif a.part == 'inside' and fmt == 'vinyl' and img.shape[0] > 1.7 * img.shape[1]:
        half = img.shape[0] // 2
        panels = [('insert_left', img[:half]), ('insert_right', img[half:])]
    if len(panels) > 1:
        paths = []
        for kant, pimg in panels:
            nm = None
            if a.discogs:
                try:
                    import scan_naam as N
                    nm = N.naam(a.discogs, kant, a.variant, a.year)
                except Exception:
                    pass
            nm = nm or f'{os.path.splitext(os.path.basename(a.scans[0]))[0]}_{kant}'
            os.makedirs(a.out, exist_ok=True)
            pth = os.path.join(a.out, nm + '.png'); n = 2
            while os.path.exists(pth) and not a.decision:
                pth = os.path.join(a.out, f'{nm}_{n}.png'); n += 1
            if disc and pimg.ndim == 3 and pimg.shape[2] == 4:
                pimg = pimg.copy(); pimg[pimg[:, :, 3] == 0, :3] = 255     # doorzichtig = geen scan: geen kleur van het vel of deksel meedragen
            cv2.imwrite(pth, pimg)
            try:
                from PIL import Image
                with Image.open(pth) as im:
                    im.load(); im.save(pth, dpi=(dpi, dpi))
            except Exception:
                pass
            paths.append({'path': pth, 'part': kant, 'size_mm': [round(pimg.shape[1] / dpi * 25.4, 1), round(pimg.shape[0] / dpi * 25.4, 1)]})
        result['path'] = paths[0]['path']; result['panels'] = paths; result['dpi'] = dpi
        if result.get('why_note'):
            result['applied'] = (result.get('applied', '') + ' ' + result['why_note']).strip()
        result['size_mm'] = [round(img.shape[1] / dpi * 25.4, 1), round(img.shape[0] / dpi * 25.4, 1)]
        for f in (raw, raw + '.scans.json', level_raw, level_raw + '.scans.json', edge_raw, edge_raw + '.scans.json', edge_crop, edge_crop + '.feiten.json', crop, crop + '.feiten.json', up):
            if os.path.exists(f):
                os.remove(f)
        print(json.dumps(result, ensure_ascii=False)); return
    # naam
    name = None
    if a.discogs:
        try:
            import scan_naam as N
            name = N.naam(a.discogs, a.part, a.variant, a.year)
        except Exception as e:                             # geen verbinding e.d.: neutrale naam
            result['name_note'] = str(e)
    name = name or f'{os.path.splitext(os.path.basename(a.scans[0]))[0]}_{a.part}'
    os.makedirs(a.out, exist_ok=True)
    # Sleeve 0.5.53 - feit: het n-de deel met dezelfde naam in scanvolgorde krijgt _n. Twee delen lopen tegelijk; wie het
    # eerst klaar was kreeg de naam zonder nummer (Queen For A Day 2021: _insert was het tekstvel van kant B, scans 35-38).
    occ = max(1, a.occurrence)
    path = os.path.join(a.out, name + ('.png' if occ == 1 else f'_{occ}.png')); n = occ + 1
    while os.path.exists(path) and not a.decision:
        path = os.path.join(a.out, f'{name}_{n}.png'); n += 1
    if a.target:
        # Feit: een beslissing geldt voor het bestand dat jij zag. Twee inserts hebben dezelfde naam
        # (_insert en _insert_2); zonder dit overschreef de tweede de eerste (0.5.29).
        path = a.target
    if a.level_vinyl and SF.SHAPES.get(fmt, SF.SHAPES['vinyl']).get(feit_part, ('rect',))[0] == 'disc':
        # Optie (André 14 nov): lichte vlekken in effen vinyl ophelderen tot de helderheid rondom; gemeten, niets verzonnen
        import scan_vlak
        img, lv_ = scan_vlak.level(img, dpi)
        if lv_.get('levelled'):
            result['applied'] = (result.get('applied', '') + f" vinyl tone levelled (up to {lv_['max_L']} L lighter)").strip()
        else:
            result['applied'] = (result.get('applied', '') + ' vinyl tone not levelled: ' + lv_.get('why', '')).strip()
    if disc and img.ndim == 3 and img.shape[2] == 4:
        img = img.copy(); img[img[:, :, 3] == 0, :3] = 255                 # idem (na het rechtzetten kan de rand zwart zijn)
    cv2.imwrite(path, img)
    try:                                                   # dpi in het bestand zetten
        from PIL import Image
        with Image.open(path) as im:
            im.load(); im.save(path, dpi=(dpi, dpi))
    except Exception:
        pass
    result['path'] = path; result['dpi'] = dpi
    if result.get('why_note'):
        result['applied'] = (result.get('applied', '') + ' ' + result['why_note']).strip()
    result['size_mm'] = [round(img.shape[1] / dpi * 25.4, 1), round(img.shape[0] / dpi * 25.4, 1)]
    for f in (raw, raw + '.scans.json', level_raw, level_raw + '.scans.json', edge_raw, edge_raw + '.scans.json', edge_crop, edge_crop + '.feiten.json', crop, crop + '.feiten.json', up, render_crop):   # tussenbestanden opruimen (1200 dpi: ±1,5 GB)
        if os.path.exists(f):
            os.remove(f)
    note_fast_reading(result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
