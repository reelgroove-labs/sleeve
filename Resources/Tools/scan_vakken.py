#!/usr/bin/env python3
"""Vakken vooraf invullen. Eerst stapels: welke deelscans horen bij één onderdeel. Dan per stapel de
naam, alleen op feiten. Een stapel zonder vaststaande naam komt als naamloos vak ('unknown'): jij kiest
het onderdeel met één klik in plaats van elke scan apart te slepen.

Stapels (feiten, gemeten op de 17 releases van de vaste toets, 14 nov):
  - deelscans van één onderdeel delen drukwerk (overlap, op schaal 1: zelfde scanner, zelfde dpi);
    zonder overlap met de lopende stapel begint een nieuw onderdeel
  - de scans worden in bestandsvolgorde gelopen (jij scant onderdeel na onderdeel)
  - een scan die op dezelfde plek ligt als een scan uit de lopende stapel (< 12 mm, < 20 graden),
    is een volgend onderdeel met hetzelfde drukwerk (kant B na kant A, twee gelijke inserts)
  - een scan die veel zwakker overeenkomt (< 0,35 x wat binnen de stapel gewoon is) ook: hetzelfde
    drukwerk in een ander onderdeel (de hoes herhaald op een insert)
  - een scan met een cd van 120 mm hoort nooit bij een scan zonder cd

Namen:
  back   de scan bevat de barcode van deze release (Discogs-barcode, cijfer voor cijfer)
  front  de scan komt aantoonbaar overeen met de hoesfoto van Discogs (type 'primary'),
         of, zonder primary, de eerste foto die Discogs als hoes toont; en met geen
         andere Discogs-foto beter (genoeg overeenkomende punten)
  cd     (cd-uitgave, één schijf) er ligt een cirkel van 120 mm in de scan, gemeten
         met de resolutie uit het bestand
  kant   (vinyl) de stapel toont de doorlopende rand van een plaat van ±30 cm; de kanten krijgen
         A, B, C, D in scanvolgorde. Dat is een voorstel: na het verwerken leest Sleeve de letter op
         het label en twijfelt als die niet klopt.
  front/back komen maar één keer voor: twee stapels met hetzelfde feit zijn één onderdeel (opnieuw gescand).

Gebruik: scan_vakken.py <discogs-url> <cd|vinyl> <vakken,komma> <scan> ...
Uitvoer: één JSON-regel {"groups": [{"part": ..., "files": [...]}], "why": {scan: reden}}
"""
import json, os, re, subprocess, sys, tempfile, math, statistics
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import scan_dpi
import scan_discogs as D

OCR = os.path.join(HERE, 'sleeve_ocr')
THUMB = 1600
MIN_FRONT = 30          # overeenkomende punten met de hoesfoto
FRONT_MIN_MM = {'vinyl': 250, 'cd': 96}   # zo breed staat de hoesfoto op de voorkant zelf (lp-hoes 305-318 mm, cd 120-125 mm: 80%)
PXMM = 2.0              # stapels: kleine beelden op 2 px/mm (genoeg drukwerk, snel)
MIN_LINK = 15           # zoveel overeenkomende punten = overlap
SAME_PLACE_MM, SAME_PLACE_DEG = 12.0, 20.0
WEAK = 0.35
RECORD_RIM = 0.21       # deel van de cirkel (r 147-151 mm) dat als doorlopende rand te zien is (mediaan van de stapel)
SIDES = ['sidea', 'sideb', 'sidec', 'sided']


def thumb(path, work):
    """Kleine versie via ImageIO (snel, ook bij grote png's); geeft beeld en schaal t.o.v. het origineel."""
    # de hele map telt mee: twee scans met dezelfde naam uit verschillende mappen mogen elkaars miniatuur niet lezen
    import hashlib
    out = os.path.join(work, hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:10] + '_' + os.path.basename(path) + '.jpg')
    subprocess.run(['/usr/bin/sips', '-Z', str(THUMB), '-s', 'format', 'jpeg', path, '--out', out], capture_output=True)
    im = cv2.imread(out)
    if im is None:
        return None, 0
    w = int(subprocess.run(['/usr/bin/sips', '-g', 'pixelWidth', path], capture_output=True, text=True).stdout.split()[-1])
    return im, im.shape[1] / max(1, w)


def barcodes(path):
    try:
        # Sleeve Transparant: een volle tif van de NAS lezen duurt 14-20 s (TA clear 0008: 14,4 s); 20 s was te krap
        r = subprocess.run([OCR, path, '--identify'], capture_output=True, text=True, timeout=120)
        return [re.sub(r'\D', '', c) for c in json.loads(r.stdout).get('barcodes', [])]
    except Exception:
        return []


def printed_record_side(path, work):
    """Read side lettering on the dark label at multiple physical rotations."""
    import scan_rechtop as upright
    from pathlib import Path
    target = str(Path(work) / 'source.jpg')
    try:
        subprocess.run(['/usr/bin/sips', '-Z', '2200', '-s', 'format', 'jpeg',
                        path, '--out', target], capture_output=True, timeout=25, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    image = cv2.imread(target)
    if image is None:
        return None
    # Full record scans show only part of the outer rim. The dark printed
    # label still provides the readable side designation.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.morphologyEx(np.uint8(gray < 80), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for idx in range(1, count):
        x, y, width, height, area = stats[idx]
        if area < .006*gray.size or not .55 < width/max(1,height) < 1.8:
            continue
        if max(width,height) > .65*max(gray.shape):
            continue
        candidates.append((area, x, y, width, height))
    if not candidates:
        # Sleeve Transparant: een licht label (US white: lichtgrijs op wit vinyl) is geen donkere vlek.
        # Dan het label als cirkel zoeken (scan_label.label_boundary, het hele label moet op de scan staan).
        try:
            import scan_label
            from PIL import Image
            dpi = scan_dpi.read(path)
            with Image.open(path) as original:
                full_width = original.width
            found = scan_label.label_boundary(image, dpi / 25.4 * image.shape[1] / full_width) if dpi else None
        except Exception:
            found = None
        if found is None:
            return None
        _, cx, cy, radius, _ = found
        x, y, width, height = int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
        candidates.append((width * height, max(0, x), max(0, y), width, height))
    _, x, y, width, height = max(candidates)
    pad = max(8, round(.12*max(width,height)))
    crop = image[max(0,y-pad):min(len(image),y+height+pad),
                 max(0,x-pad):min(image.shape[1],x+width+pad)]
    crop = cv2.resize(crop, None, fx=1000/max(crop.shape[:2]), fy=1000/max(crop.shape[:2]),
                      interpolation=cv2.INTER_CUBIC)
    values = {'ONE':'a','TWO':'b','THREE':'c','FOUR':'d','1':'a','2':'b','3':'c','4':'d'}
    choices = set()
    for angle in range(0,90,15):
        h,w = crop.shape[:2]
        matrix = cv2.getRotationMatrix2D((w/2,h/2),-angle,1)
        view = cv2.warpAffine(crop,matrix,(w,h),borderValue=(255,255,255))
        readings = upright.read_image(view)
        for lines in readings.values():
            for line in lines:
                match = re.fullmatch(r'\s*SIDE[ :.-]*(ONE|TWO|THREE|FOUR|[ABCD1-4])\s*',
                                     line.strip(' -._:').upper())
                if match:
                    choices.add('side'+values.get(match[1],match[1].lower()))
    return choices.pop() if len(choices)==1 else None


def assign_printed_sides(named):
    """Regroup only when the label supplies the side of every involved scan."""
    record_groups = [item for item in named if item[0] == 'side']
    if not record_groups:
        return named
    paths = [path for item in record_groups for path in item[1]]
    from concurrent.futures import ThreadPoolExecutor
    with tempfile.TemporaryDirectory() as work, ThreadPoolExecutor(2) as pool:
        # Distinct folders also support duplicate basenames from different inputs.
        def read(item):
            idx, path = item
            folder = os.path.join(work, str(idx))
            os.mkdir(folder)
            return printed_record_side(path, folder)
        sides = list(pool.map(read, enumerate(paths)))
    if any(side is None for side in sides) or len(set(sides)) != len(record_groups):
        return partial_printed_sides(named, record_groups, dict(zip(paths, sides)))
    record = [[side, sorted((p for p, label in zip(paths, sides) if label == side), key=number_key),
               f'the printed label reads SIDE {side[-1].upper()} on each scan'
               + ('; check the scan DPI before export' if any(
                    'check the scan DPI' in item[2] and any(path in item[1] and label == side
                                                           for path,label in zip(paths,sides))
                    for item in record_groups) else '')]
              for side in SIDES if side in sides]
    it = iter(record)
    return [next(it) if item[0] == 'side' else item for item in named]


def partial_printed_sides(named, record_groups, read):
    """Sleeve Transparant: het label is niet op elke scan te lezen (randscans, glans). Feit: wat wel gelezen is, klopt.
    Per stapel de gelezen kant(en); een stapel met één gelezen kant is die kant; een stapel met twee leest de kant die
    nog niet aan een andere stapel toekomt (GF green: stapel 0010-0015 las A op 0014 en B op 0015; stapel 0016-0019
    las B). Een scan waarvan het label een andere kant leest dan zijn stapel, gaat naar de stapel van die kant.
    Alleen als elke stapel zo precies één eigen kant krijgt; anders niets veranderen."""
    seen = [{read[p] for p in item[1] if read.get(p)} for item in record_groups]
    side_of = [None] * len(record_groups)
    for _ in range(len(record_groups)):
        taken = {x for x in side_of if x}
        for i, got in enumerate(seen):
            if side_of[i] is None:
                left = got - taken
                if len(left) == 1:
                    side_of[i] = left.pop(); taken.add(side_of[i])
    # Sleeve Transparant: twee stapels, één leest twee kanten en de ander niets (US white: 0010 leest A en
    # de losse labelscan 0015 leest B, 0016-0019 tonen maar een stuk label). De lezende stapel is de kant
    # van zijn eerste gelezen scan; de andere stapel is de andere gelezen kant.
    if None in side_of and len(record_groups) == 2 and sorted(len(s) for s in seen) == [0, 2]:
        i = 0 if len(seen[0]) == 2 else 1
        first = min((p for p in record_groups[i][1] if read.get(p)), key=number_key)
        side_of[i] = read[first]
        side_of[1 - i] = (seen[i] - {side_of[i]}).pop()
    if None in side_of or len(set(side_of)) != len(side_of):
        return named
    where = {side: item for side, item in zip(side_of, record_groups)}
    moved = []
    for side, item in zip(side_of, record_groups):
        for p in list(item[1]):
            r = read.get(p)
            if r and r != side and r in where:
                item[1].remove(p); where[r][1].append(p); moved.append((os.path.basename(p), r))
    for side, item, got in zip(side_of, record_groups, seen):
        item[1].sort(key=number_key)
        item[0] = side
        item[2] = (f'the printed label reads SIDE {side[-1].upper()} on part of these scans' if got else
                   f'no label read on these scans; the other record scans read the other side, so this is SIDE {side[-1].upper()}')
    for name, r in moved:
        where[r][2] += f'; {name} moved here: its label reads SIDE {r[-1].upper()}'
    return named


def other_version_covers(rel, fmt, limit=4):
    """Hoesfoto's (primary) van andere uitgaven van hetzelfde album (Discogs-master), zelfde soort drager."""
    want = 'cd' if str(fmt).startswith('cd') else 'vinyl'
    out = []
    try:
        versions = json.loads(D.get(f"https://api.discogs.com/masters/{rel['master_id']}/versions?per_page=100")).get('versions', [])
    except Exception:
        return out
    for v in versions:
        if len(out) >= limit:
            break
        if v.get('id') == rel.get('id') or want not in (v.get('format') or '').lower() and not (want == 'vinyl' and 'lp' in (v.get('format') or '').lower()):
            continue
        try:
            r = json.loads(D.get(f"https://api.discogs.com/releases/{v['id']}"))
            prim = next((i for i in r.get('images', []) if i.get('type') == 'primary'), None)
            if prim:
                arr = cv2.imdecode(np.frombuffer(D.get(prim['uri']), np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    out.append(arr)
        except Exception:
            continue
    return out


def same_barcode(a, b):
    a, b = a.lstrip('0'), b.lstrip('0')
    return len(a) >= 8 and a == b


def has_cd(im, px_mm):
    """Cirkel van 120 mm (±2 mm) met een duidelijke rand rondom."""
    if px_mm <= 0:
        return False
    g = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 5)
    r0 = 60 * px_mm
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=r0, param1=80, param2=40,
                         minRadius=int(58 * px_mm), maxRadius=int(62 * px_mm) + 1)
    if c is None:
        return False
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1); mag = np.hypot(gx, gy)
    h, w = g.shape
    for x, y, r in c[0][:3]:
        hits = n = 0
        for k in range(144):
            t = 2 * math.pi * k / 144
            xs = int(round(x + r * math.cos(t))); ys = int(round(y + r * math.sin(t)))
            if 1 <= xs < w - 1 and 1 <= ys < h - 1:
                n += 1
                hits += mag[ys - 1:ys + 2, xs - 1:xs + 2].max() > 40
        if n >= 120 and hits / n >= 0.85:
            return True
    return False


NAME_ORDER = False      # main(): True als de getallen in de namen geen doorlopende teller zijn


def natural(path):
    """Natuurlijke naamvolgorde zoals de Finder: 'Scan' voor 'Scan 1' voor 'Scan 2' voor 'Scan 10'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return [(0, int(x)) if x.isdigit() else (1, x.lower()) for x in re.split(r'(\d+)', stem) if x]


def numbers(path):
    """Alle getallen in de naam, van links naar rechts: datum, tijd en volgnummer van de scanner
    (IMG_20261112_0012 -> 20261112, 12; 'Scan 2026-11-14 at 10.32.05' -> 2026, 11, 14, 10, 32, 5; 'Scan' -> geen)."""
    return tuple(int(x) for x in re.findall(r'\d+', os.path.splitext(os.path.basename(path))[0]))


def set_order(scans):
    """Feit: een doorlopende teller (de scanner nummert elke scan) loopt door, ook als jij tussendoor een andere naam
    geeft (queen_..._0017, queen_sidea_label_..._0018, queen_..._0019). Tellen twee reeksen met een eigen naam elk
    opnieuw over dezelfde getallen (us_white_sidea_20261114_0001 .. 0004, us_white_sideb_20261114_0001 .. 0004),
    dan is er per naam een teller en is de naam de volgorde. Voorheen telde alleen het laatste getal: alle 0001's
    eerst, de onderdelen door elkaar (us_new_white_back3, 15 nov)."""
    global NAME_ORDER
    series = {}
    for s in scans:
        series.setdefault(re.sub(r'\d+', '#', os.path.splitext(os.path.basename(s))[0].lower()), []).append(numbers(s))
    spans = sorted((min(v), max(v)) for v in series.values() if len(v) >= 2)
    NAME_ORDER = any(b[0] <= a[1] for a, b in zip(spans, spans[1:]))


def number_key(path):
    """Scanvolgorde. Doorlopende teller: alle getallen in de naam, van links naar rechts (zo ook over twee dagen
    heen, en een eigen woord in de naam verandert niets: queen_sidea_label_20261114_0018 hoort tussen 0017 en 0019).
    Een teller per naam (set_order): de naam, in natuurlijke volgorde."""
    return (natural(path),) if NAME_ORDER else (numbers(path), natural(path))


def small(path, work):
    """Kleine grijze versie op PXMM pixels per mm (via de resolutie uit het bestand)."""
    dpi = scan_dpi.read(path) or 300.0
    try:
        w = int(subprocess.run(['/usr/bin/sips', '-g', 'pixelWidth', path], capture_output=True, text=True).stdout.split()[-1])
    except (ValueError, IndexError):
        return None
    out = os.path.join(work, f'{abs(hash(path))}_s.png')
    subprocess.run(['/usr/bin/sips', '--resampleWidth', str(max(64, int(round(w / dpi * 25.4 * PXMM)))), '-s', 'format', 'png', path, '--out', out], capture_output=True)
    return cv2.imread(out)


SIFT = cv2.SIFT_create(nfeatures=4000)
BF = cv2.BFMatcher(cv2.NORM_L2)


def features(im):
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    k, d = SIFT.detectAndCompute(g, None)
    return (np.float32([p.pt for p in k]) if k else np.zeros((0, 2), np.float32), d, g.shape)


def link(a, b):
    """Overlap tussen twee scans: (aantal overeenkomende punten, verschuiving in mm, draaiing in graden).
    Alleen op schaal 1 (zelfde scanner, zelfde dpi); anders geen overlap."""
    (pa, da, sa), (pb, db, sb) = a, b
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return 0, 0.0, 0.0
    m = [x[0] for x in BF.knnMatch(da, db, k=2) if len(x) == 2 and x[0].distance < .75 * x[1].distance]
    if len(m) < 8:
        return 0, 0.0, 0.0
    M, inl = cv2.estimateAffinePartial2D(pa[[x.queryIdx for x in m]], pb[[x.trainIdx for x in m]],
                                         method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=3000)
    if M is None or abs(float(np.hypot(M[0, 0], M[1, 0])) - 1) > .04:
        return 0, 0.0, 0.0
    moved = M @ np.array([sa[1] / 2, sa[0] / 2, 1.0]) - np.array([sb[1] / 2, sb[0] / 2])
    ang = abs((np.degrees(np.arctan2(M[1, 0], M[0, 0])) + 180) % 360 - 180)
    LINKS[id(a), id(b)] = M                                  # voor de maat van de hele stapel (extent)
    return int(inl.sum()), float(np.hypot(*moved) / PXMM), float(ang)


LINKS = {}


def extent_mm(g, feats):
    """Breedte en hoogte (mm) van wat de deelscans van een stapel samen bedekken: elke scan op zijn plek via
    de overlap met een eerdere scan uit de stapel. Een opengeklapte hoes is ±62 cm breed, een vel ±30 cm."""
    T = {g[0]: np.eye(3)}
    for s in g[1:]:
        best = None
        for x in T:
            M = LINKS.get((id(feats[s]), id(feats[x])))
            if M is not None and (best is None or True):
                best = (x, M); break
        if best is None:
            return None
        x, M = best
        T[s] = T[x] @ np.vstack([M, [0, 0, 1]])
    pts = []
    for s, t in T.items():
        h, w = feats[s][2]
        c = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64).T
        pts.append((t @ c)[:2].T)
    P = np.vstack(pts)
    return float((P[:, 0].max() - P[:, 0].min()) / PXMM), float((P[:, 1].max() - P[:, 1].min()) / PXMM)


def sheet_extent(g, feats):
    """Zoals extent_mm, maar elke scan op zijn plek via zijn STERKSTE overlap met een al geplaatste scan (een zwakke
    overlap van een bijna leeg stuk papier mag de maat niet bepalen). None als niet alles te plaatsen is."""
    T = {g[0]: np.eye(3)}; todo = list(g[1:])
    while todo:
        best = None
        for s in todo:
            for x in T:
                n = link(feats[s], feats[x])[0]
                M = LINKS.get((id(feats[s]), id(feats[x])))
                if n >= MIN_LINK and M is not None and (best is None or n > best[0]):
                    best = (n, s, x, M)
        if best is None:
            return None
        _, s, x, M = best
        T[s] = T[x] @ np.vstack([M, [0, 0, 1]]); todo.remove(s)
    pts = []
    for s, t in T.items():
        h, w = feats[s][2]
        c = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64).T
        pts.append((t @ c)[:2].T)
    P = np.vstack(pts)
    return float((P[:, 0].max() - P[:, 0].min()) / PXMM), float((P[:, 1].max() - P[:, 1].min()) / PXMM)


def sticker_on(path):
    with tempfile.TemporaryDirectory() as work:
        return bool(is_sticker(path, work))


def record_rim(im, center=None):
    """Deel van een cirkel van een lp (straal 147-151 mm) dat als doorlopende rand in de scan staat, ook als
    het middelpunt buiten de scan ligt. Stemmen langs de gradiënt; dan de rand natellen: loodrecht op de
    straal en in stukken van ten minste 10 graden (een plaatrand loopt door, drukwerk niet)."""
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 1.0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1); mag = np.hypot(gx, gy)
    h, w = g.shape
    ys, xs = np.nonzero(mag > max(20.0, np.percentile(mag, 90)))
    if len(xs) > 60000:
        sel = np.random.default_rng(0).choice(len(xs), 60000, replace=False); ys, xs = ys[sel], xs[sel]
    if not len(xs):
        return 0.0
    nx = gx[ys, xs] / mag[ys, xs]; ny = gy[ys, xs] / mag[ys, xs]
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False); best = 0.0
    for rmm in (147, 149, 151):
        r = rmm * PXMM; pad = int(r) + 4
        acc = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        for sgn in (1, -1):
            np.add.at(acc, (np.round(ys + sgn * ny * r).astype(int) + pad, np.round(xs + sgn * nx * r).astype(int) + pad), 1)
        cy, cx = np.unravel_index(np.argmax(cv2.GaussianBlur(acc, (0, 0), 1.5)), acc.shape); cy -= pad; cx -= pad
        px = np.round(cx + r * np.cos(t)).astype(int); py = np.round(cy + r * np.sin(t)).astype(int)
        inside = (px >= 2) & (px < w - 2) & (py >= 2) & (py < h - 2)
        if inside.sum() < 60:
            continue
        hits = np.zeros(720, bool)
        for k in np.nonzero(inside)[0]:
            y1, x1 = max(0, py[k] - 3), max(0, px[k] - 3)
            win = mag[y1:py[k] + 4, x1:px[k] + 4]
            j = np.unravel_index(np.argmax(win), win.shape); yy, xx = y1 + j[0], x1 + j[1]
            if mag[yy, xx] >= 40:
                hits[k] = abs((gx[yy, xx] * np.cos(t[k]) + gy[yy, xx] * np.sin(t[k])) / mag[yy, xx]) > .92
        good = run = 0
        for k in range(720):
            run = run + 1 if hits[k] else 0
            if run and (k == 719 or not hits[k + 1]) and run >= 20:
                good += run
        score = good / 720.0
        if score > best:
            best = score
            if center is not None:
                center[:] = [cx, cy]
    return best


def scaled_record_evidence(image):
    """Recognize a record at the other common scanner resolution using three shapes."""
    doubled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    center = []
    if record_rim(doubled, center) < RECORD_RIM:
        return False
    gray = cv2.cvtColor(doubled, cv2.COLOR_BGR2GRAY)
    labels = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1,
                              minDist=30, param1=80, param2=35, minRadius=70, maxRadius=125)
    if labels is None:
        return False
    h, w = gray.shape
    for x, y, _ in labels[0]:
        x, y = round(float(x)), round(float(y))
        if x < 26 or y < 26 or x >= w - 26 or y >= h - 26:
            continue
        if math.hypot(x - center[0], y - center[1]) > 8 * PXMM:
            continue
        # A spindle hole is about 7 mm across and centered on the label.
        patch = gray[y-25:y+26, x-25:x+26]
        yy, xx = np.ogrid[-25:26,-25:26]
        distance = np.hypot(xx, yy)
        center = patch[distance <= 8]
        label = patch[(distance >= 15) & (distance <= 23)]
        if abs(float(np.median(center)) - float(np.median(label))) >= 15:
            return True
    return False


def related_paper_groups(groups, feats, rims, paper_notes):
    """Link adjacent scan rounds when the same printing supplies several matches."""
    i = 0
    while i + 1 < len(groups):
        first, second = groups[i:i+2]
        if any(rims.get(path, 0) >= RECORD_RIM or path not in feats for path in first+second):
            i += 1
            continue
        strong = []
        for left in first:
            for right in second:
                count, shift, angle = link(feats[right], feats[left])
                if count >= 80:
                    strong.append((left, right, count, shift, angle))
        linked_left = {x[0] for x in strong}
        linked_right = {x[1] for x in strong}
        repeat = any(count >= 180 and shift <= SAME_PLACE_MM and angle <= 5
                     for _, _, count, shift, angle in strong)
        corroborated = len(linked_left) >= 2 and len(linked_right) >= 2 and len(strong) >= 3
        duplicate = repeat and len(strong) >= 3
        if not (corroborated or duplicate):
            i += 1
            continue
        # Do not attach another capture to a complete scan round. The full
        # processing job should still use one suitable set of scans.
        a, b = extent_mm(first, feats), extent_mm(second, feats)
        union = extent_mm(first+second, feats)
        sizes = [scan_dpi.size_px(path) for path in first+second]
        total_pixels = sum(w*h for w,h in sizes) if all(sizes) else None
        fits = total_pixels is not None and total_pixels <= 700_000_000
        extends = a is not None and union is not None and (
            max(union) >= max(a) + 30 or min(union) >= min(a) + 30)
        safe_same_sheet = union is not None and min(union) <= 465 and max(union) <= 760
        if corroborated and extends and safe_same_sheet and fits and not (
                a and b and min(a) >= 305 and min(b) >= 305):
            groups[i] = first + second
            del groups[i+1]
            paper_notes[first[0]] = 'matching print extends the coverage of this paper stack'
            continue
        first_name = os.path.basename(first[0])
        if extends and safe_same_sheet and total_pixels is not None and not fits:
            paper_notes[second[0]] = (f'overlaps the stack starting with {first_name}; '
                                      f'together they exceed the full-resolution limit ({total_pixels / 1e6:.0f} Mpx)')
        elif duplicate:
            paper_notes[second[0]] = (f'one scan aligns to the same position in the stack starting with {first_name}; '
                                      'kept as a separate scan round')
        else:
            paper_notes[second[0]] = (f'printed details also appear in the stack starting with {first_name}; '
                                      'check whether they belong to the same part')
        i += 1
    return groups


def is_sticker(scan, work):
    """Feit: een sticker is een kleine gesloten vorm met afgeronde hoeken (of een cirkel) op een verder lege scan
    (zijn plastic hoes). Een cd-boekje of kaartje heeft scherpe hoeken; een cd is een cirkel van 120 mm."""
    out = os.path.join(work, 'sticker_probe.png')
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, 'scan_sticker.py'), scan, '--out', out], capture_output=True, text=True, timeout=180)
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError, subprocess.TimeoutExpired, OSError):
        return None
    if res.get('verdict') == 'FOUT' or not res.get('area_frac') or res['area_frac'] > .15:
        return None
    # Sleeve Transparant: een sticker is minstens 25 mm in beide richtingen (MOV: ±50 x 80 mm). TA clear 0003: een
    # vorm van 11 x 31 mm in het drukwerk van de hoes telde anders als sticker.
    if res.get('size_mm') and min(res['size_mm']) < 25:
        return None
    dpi = res.get('dpi') or 0
    if res.get('shape') == 'circle':
        if res.get('verdict') != 'GOED':
            return None                                     # rand van de cirkel maar deels te zien: geen sticker (Paradise 2014 Scan 6: 61%)
        d = (res.get('size_mm') or [0])[0]
        if dpi and 112 <= d <= 125:
            return None                                     # een cd, geen sticker
        return 'a round sticker'
    if dpi and res.get('corner_px', 0) / dpi * 25.4 < 1.5:
        return None                                         # scherpe hoeken: een kaartje of boekje
    return 'a small card with rounded corners (a sticker)'


def shape_in_print(scan, others, work):
    """Feit: staat de gevonden kleine vorm (scan_sticker) ook op een andere scan van dezelfde stapel, dan is hij deel
    van het drukwerk van de hoes, geen sticker erop. Vergelijken op 2 px/mm, binnen de afgeronde hoeken, in vier
    standen (een hoes wordt halverwege een halve slag gedraaid). Drukwerk: 0,91-0,98; sticker op de folie: 0,23-0,25."""
    out = os.path.join(work, 'shape_probe.png')
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, 'scan_sticker.py'), scan, '--out', out], capture_output=True, text=True, timeout=180)
        res = json.loads(r.stdout.strip().splitlines()[-1]); crop = cv2.imread(out)
    except (ValueError, IndexError, subprocess.TimeoutExpired, OSError):
        return False
    dpi = res.get('dpi') or scan_dpi.read(scan)
    if crop is None or not dpi:
        return False
    f = PXMM * 25.4 / dpi
    c = cv2.cvtColor(cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    m = max(4, int(.15 * min(c.shape))); c = c[m:-m, m:-m]
    if min(c.shape) < 8:
        return False
    for t in others:
        sm = small(t, work)
        if sm is None:
            continue
        g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        for k in range(4):
            q = np.ascontiguousarray(np.rot90(c, k))
            if q.shape[0] <= g.shape[0] and q.shape[1] <= g.shape[1] and float(cv2.matchTemplate(g, q, cv2.TM_CCOEFF_NORMED).max()) >= .6:
                return True
    return False


def card_size(scan, work):
    """Maat (mm) van het kaartje op één cd-scan, gemeten zoals bij het uitsnijden (scan_feiten, als achterkaart
    van een jewelcase: 136-154 x 114-120 mm). Geeft (breedte, hoogte) of None als zo'n kaart er niet in ligt."""
    import shutil as _sh
    d = tempfile.mkdtemp(dir=work); src = os.path.join(d, 'scan' + os.path.splitext(scan)[1])
    try:
        os.symlink(os.path.abspath(scan), src)
        dpi = scan_dpi.read(scan)
        if not dpi:
            return None
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(scan) as im:
            w, h = im.size
            im.convert('RGBA').save(os.path.join(d, 'raw.png'))
        json.dump({'scale': 1, 'dpi': dpi, 'scans': [{'scan': 'scan', 'corners': [[0, 0], [w, 0], [w, h], [0, h]]}]}, open(os.path.join(d, 'raw.png.scans.json'), 'w'))
        r = subprocess.run([sys.executable, os.path.join(HERE, 'scan_feiten.py'), os.path.join(d, 'raw.png'), 'back', '--format', 'cd'], capture_output=True, text=True, timeout=300)
        try:
            res = json.loads(r.stdout[r.stdout.index('{'):r.stdout.rindex('}') + 1]) if '{' in r.stdout else {}
        except ValueError:
            res = {}
        box = res.get('precise_mm') or res.get('box_mm')
        if res.get('verdict') == 'FOUT' or not box or len(box) < 4:
            return None
        return box[2], box[3]
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        _sh.rmtree(d, ignore_errors=True)


def real(M):
    """Een echte overeenkomst met een Discogs-foto heeft een schaal; schaal 0 is een ontaarde oplossing."""
    return M is not None and float(np.hypot(M[0, 0], M[1, 0])) > .05


def background(sm):
    """Achtergrond langs de scanrand: 'lid' (witte klep: licht en kleurloos) of 'sheet' (gekleurd of donker vel)."""
    h, w = sm.shape[:2]; b = max(3, int(.03 * min(h, w)))
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    white = (ring[:, 0] * 100 / 255 > 80) & (np.hypot(ring[:, 1] - 128, ring[:, 2] - 128) < 10)
    return 'lid' if white.mean() > .25 else 'sheet'


def complete(g, feats):
    """Bedekt deze stapel al een hele hoes (minstens 305 mm in beide richtingen)? Zo niet, dan mag een zwakke maar
    echte overlap erbij: een hoes die halverwege een halve slag is gedraaid, deelt weinig drukwerk tussen de twee
    helften (US wit voorkant: 39 punten tegen 151 binnen een helft) - maar alleen als de stapel daardoor een hele
    hoes wordt. Een insert met de hoes erop komt na een complete hoes-stapel en blijft apart; twee boekjespagina's
    van 29 cm worden samen geen hoes en blijven ook apart (US groen)."""
    e = extent_mm(g, feats) if len(g) > 1 else None
    return e is not None and min(e) >= 305


def stacks(scans, feats, discs):
    """Deelscans in bestandsvolgorde tot stapels: zie de feiten bovenaan."""
    order = sorted(scans, key=number_key)
    groups, strengths = [[order[0]]], [[]]
    for s in order[1:]:
        g = groups[-1]
        links = {x: link(feats[s], feats[x]) for x in g} if feats.get(s) and all(feats.get(x) for x in g) else {}
        best = max([v[0] for v in links.values()] or [0])
        new = (best < MIN_LINK
               or discs.get(s) != discs.get(g[0])
               or any(v[0] >= MIN_LINK and v[1] < SAME_PLACE_MM and v[2] < SAME_PLACE_DEG for v in links.values())
               or (strengths[-1] and best < WEAK * statistics.median(strengths[-1]) and (complete(g, feats) or not complete(g + [s], feats))))
        if new:
            groups.append([s]); strengths.append([])
        else:
            g.append(s); strengths[-1].append(best)
    # Sleeve 0.5.51 - feit: een vel waarvan één stuk bijna leeg is (Urban Solitude 2025 crystal clear, achterkant 0007:
    # crème papier, 7-10 punten met de scans ervoor) begon een nieuwe stapel, terwijl de scans daarna (0008, 0009) wél
    # duidelijk op de eerste stukken passen (53-82 punten, een andere plek op het vel). Twee opeenvolgende stapels die elk
    # nog geen hele hoes zijn, horen bij elkaar als een scan van de tweede echt overlapt met de eerste (minstens MIN_LINK
    # punten, minstens WEAK x de mediaan binnen die stapels) en ze samen precies één vel bedekken (305-470 mm beide kanten).
    merged = [groups[0]] if groups else []
    for g in groups[1:]:
        a = merged[-1]
        if all(feats.get(x) for x in a + g) and not complete(a, feats) and not complete(g, feats):
            cross = max(link(feats[y], feats[x])[0] for y in g for x in a)
            inside = [link(feats[y], feats[x])[0] for h in (a, g) for i, y in enumerate(h) for x in h[:i]]
            e = sheet_extent(a + g, feats)
            if (cross >= MIN_LINK and (not inside or cross >= WEAK * statistics.median(inside))
                    and e is not None and 305 <= min(e) and max(e) <= 470):
                merged[-1] = a + g
                continue
        merged.append(g)
    groups = merged
    # Feit: twee pagina's met dezelfde opmaak (kolommen, cijfers) delen wat punten, maar veel minder dan de
    # twee scans van één pagina onderling (US boekje x0003-4 en x0005-6: 1254 en 1402 binnen, hooguit 50
    # ertussen = 4%). Eén pagina in vier stukken deelt tussen de helften veel meer (GF gezicht: 100 en 338
    # binnen, 40 ertussen = 40%; TA ±36%; US-voorkant halve slag 26%). Een stapel van minstens 4 scans die
    # zo in twee delen (elk minstens 2 scans, op volgorde) uiteenvalt, zijn twee vellen.
    out = []
    for g in groups:
        best_k = None
        if len(g) >= 4 and all(feats.get(x) for x in g):
            L = {(a, b): link(feats[g[a]], feats[g[b]])[0] for a in range(len(g)) for b in range(a)}
            for k in range(2, len(g) - 1):
                wa = max(L[a, b] for a in range(k) for b in range(a))
                wb = max(L[a, b] for a in range(k, len(g)) for b in range(k, a))
                cross = max(L[a, b] for a in range(k, len(g)) for b in range(k))
                if cross < .15 * min(wa, wb) and (best_k is None or cross / min(wa, wb) < best_k[1]):
                    best_k = (k, cross / min(wa, wb))
        out += [g[:best_k[0]], g[best_k[0]:]] if best_k else [g]
    return out


CLOSEUP_MM = 140       # Sleeve Transparant: een losse uitsnede (label, sticker) is in beide richtingen hooguit zo groot
LOOKALIKE = str.maketrans({'А': 'A', 'В': 'B', 'С': 'C', 'Д': 'D', 'а': 'A', 'в': 'B', 'с': 'C'})   # Cyrillisch dat Vision soms leest
SIDE_WORDS = {'ONE': 'a', 'TWO': 'b', 'THREE': 'c', 'FOUR': 'd', '1': 'a', '2': 'b', '3': 'c', '4': 'd'}


def scan_mm(path):
    """Breedte en hoogte van de scan zelf in mm (pixels en dpi uit het bestand), of None."""
    sz = scan_dpi.size_px(path); dpi = scan_dpi.read(path)
    return (sz[0] / dpi * 25.4, sz[1] / dpi * 25.4) if sz and dpi else None


def label_circle(path, work):
    """Sleeve Transparant: een losse labelscan. Feit: de scanner snijdt bij tot alleen het label (±100-130 mm), en een
    label is een cirkel van 80-106 mm (lp-label ±100 mm). Geeft (x, y, r) in mm op de scan, of None. Gemeten op 4 px/mm
    (getest op 24 releases: elk label 89-100 mm; geen sticker, tekststrook of kaartje gaf zo'n cirkel)."""
    sz = scan_dpi.size_px(path); dpi = scan_dpi.read(path)
    if not sz or not dpi:
        return None
    out = os.path.join(work, f'{abs(hash(path))}_l.png')
    subprocess.run(['/usr/bin/sips', '--resampleWidth', str(max(64, int(sz[0] / dpi * 25.4 * 4))), '-s', 'format', 'png', path, '--out', out], capture_output=True)
    im = cv2.imread(out)
    if im is None:
        return None
    g = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 5)
    sat = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1], 5)
    c = None
    # tweede en derde poging: een gekleurd label op doorzichtig vinyl heeft weinig contrast in grijs (Fake It 2021
    # Scan 10: oranje op roze-doorzichtig); op verzadiging of met een iets lagere drempel wel (geen valse cirkels
    # op de 24 releases)
    for img_, p2 in ((g, 40), (sat, 40), (g, 30)):
        c = cv2.HoughCircles(img_, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=80, param2=p2, minRadius=160, maxRadius=212)
        if c is not None:
            break
    if c is None:
        return None
    x, y, r = (float(v) / 4 for v in c[0][0])
    return (x, y, r) if 0 < x < g.shape[1] / 4 and 0 < y < g.shape[0] / 4 else None


def label_side(path, circle):
    """Sleeve Transparant: de kant op een losse labelscan, gelezen op het label zelf (nauwkeurig lezen, 8 standen).
    Feit: 'SIDE A', 'SIDE ONE', 'A SIDE' is de kant. Een losse letter ('A', '- A -') ook, maar alleen als hij in minstens
    twee standen gelezen wordt (een losse D uit 'DEENA' kwam in één stand voor). Eén kant, anders None."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    dpi = scan_dpi.read(path)
    if not dpi:
        return None
    x, y, r = circle; k = dpi / 25.4
    try:
        with Image.open(path) as im:
            crop = np.asarray(im.convert('RGB').crop((int(max(0, (x - r) * k)), int(max(0, (y - r) * k)), int((x + r) * k), int((y + r) * k))))[:, :, ::-1]
    except (OSError, ValueError):
        return None
    if not crop.size:
        return None
    f = 2000 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    strong, letters = _read_label_sides(crop)
    if not strong and _LAST_READ_TIMED_OUT and _warm_up_text_recognition():
        # Feit (For Bitter Or Worse 2009, Scan 16): de eerste labelscan viel in de koude start van het nauwkeurige lezen
        # (60 s; daarna 3,7 s per scan); de standen die op 30 s afbraken lazen niets, en het label bleef bij de verkeerde
        # kant liggen. Na het opwarmen nog één keer lezen.
        strong, letters = _read_label_sides(crop)
    got = set(strong) or {sd for sd, n in letters.items() if n >= 2}
    return got.pop() if len(got) == 1 else None


_LAST_READ_TIMED_OUT = False
_WARMED = False


def _warm_up_text_recognition():
    """Het nauwkeurige lezen van macOS eenmalig laten opstarten (koud tot ±100 s). True als het daarna werkt."""
    global _WARMED
    if _WARMED:
        return False                                       # al opgewarmd: opnieuw lezen helpt dan niet
    _WARMED = True
    img = np.full((80, 400, 3), 255, np.uint8); cv2.putText(img, 'SIDE A', (10, 55), 0, 1.2, (0, 0, 0), 2)
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, 'w.png'); cv2.imwrite(pth, img)
        try:
            subprocess.run([OCR, pth, '--boxes'], capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired):
            return False
    return True


def _read_label_sides(crop):
    """De kantaanduiding op een label, gelezen in 8 standen. Geeft (sterke treffers, losse letters) per kant."""
    global _LAST_READ_TIMED_OUT
    _LAST_READ_TIMED_OUT = False
    strong, letters = {}, {}
    with tempfile.TemporaryDirectory() as td:
        for ang in range(0, 360, 45):
            h, w = crop.shape[:2]
            v = cv2.warpAffine(crop, cv2.getRotationMatrix2D((w / 2, h / 2), -ang, 1), (w, h), borderValue=(255, 255, 255))
            pth = os.path.join(td, 'v.png'); cv2.imwrite(pth, v)
            try:
                out = subprocess.run([OCR, pth, '--boxes'], capture_output=True, text=True, timeout=30).stdout
            except subprocess.TimeoutExpired:
                _LAST_READ_TIMED_OUT = True
                continue
            except OSError:
                continue
            for row in out.splitlines():
                parts = row.split('\t')
                if len(parts) < 2:
                    continue
                t = parts[1].translate(LOOKALIKE).upper().strip()
                m = (re.fullmatch(r'(?:SIDE|KANT|FACE)\s*[:.\-]?\s*(ONE|TWO|THREE|FOUR|[ABCD1-4])', t.strip(' -._:'))
                     or re.fullmatch(r'([ABCD])\s*-?\s*SIDE', t.strip(' -._:')))
                if m:
                    side = 'side' + SIDE_WORDS.get(m[1], m[1].lower()); strong[side] = strong.get(side, 0) + 1
                    continue
                m = re.fullmatch(r'[\-–—~•·]?\s*([ABCD])\s*[\-–—~•·]?', t)
                if m:
                    side = 'side' + m[1].lower(); letters[side] = letters.get(side, 0) + 1
    return strong, letters


def small_sticker(path, mm, work):
    """Sleeve Transparant: een losse stickerscan (de scanner snijdt bij tot de sticker). Feit: een kaartje met ronde
    hoeken dat het grootste deel van een kleine scan vult, of een kleine scan (hooguit 125 mm) waarvan het kaartje tot de
    rand loopt (dan is er geen rand rondom te vinden). Geeft de reden of None."""
    out = os.path.join(work, 'closeup_probe.png')
    # herkennen kan op ±2000 px (1200 dpi duurde langer dan de tijdgrens: Trails of Fails Scan 23)
    src, extra = path, []
    sz = scan_dpi.size_px(path); dpi0 = scan_dpi.read(path)
    if sz and dpi0 and max(sz) > 2000:
        src = os.path.join(work, 'closeup_small.png'); f = 2000 / max(sz)
        subprocess.run(['/usr/bin/sips', '--resampleWidth', str(int(sz[0] * f)), '-s', 'format', 'png', path, '--out', src], capture_output=True)
        extra = ['--dpi', str(dpi0 * f)]
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, 'scan_sticker.py'), src, '--out', out] + extra, capture_output=True, text=True, timeout=180)
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError, subprocess.TimeoutExpired, OSError):
        return None
    dpi = res.get('dpi') or 0
    if res.get('shape', '').startswith('rectangle') and res.get('size_mm') and min(res['size_mm']) >= 25 and dpi \
            and res.get('corner_px', 0) / dpi * 25.4 >= 1.5:
        return f'a separate scan of one small card with rounded corners ({res["size_mm"][0]:.0f} x {res["size_mm"][1]:.0f} mm): the sticker'
    if res.get('verdict') == 'FOUT' and max(mm) <= 125:
        return f'a separate small scan ({mm[0]:.0f} x {mm[1]:.0f} mm) of one small card that fills it: the sticker'
    return None


def place_label_scans(named, label_scans, feats, expected_sides=None):
    """Sleeve Transparant: losse labelscans naar hun kant. Feit 1: de kant die op het label staat (label_side). Feit 2:
    dezelfde opname op dezelfde plek als een gelezen labelscan heeft dezelfde kant. Feit 3: geen kant gelezen en hij
    ligt al bij een kant: daar laten. Feit 4: anders de kant van de plaatscan waar hij het meest mee overlapt. Een
    labelscan die zo nergens heen kan, is geen insert of sticker maar een label waarvan jij de kant kiest."""
    sides = {x[0]: x for x in named if x[0] in SIDES}
    read = {s: label_side(s, c) for s, c in label_scans.items()}
    for s in label_scans:
        if read[s] is None and feats.get(s):
            for t in label_scans:
                if t != s and read[t] and feats.get(t):
                    n, shift, deg = link(feats[s], feats[t])
                    if n >= MIN_LINK and shift < SAME_PLACE_MM and deg < SAME_PLACE_DEG:
                        read[s] = read[t]; break
    if not sides:
        # Alleen labels gescand, geen plaat (Fake It 2021): elk label is het label van zijn kant (vak labela, labelb ...).
        # Feit: de kant die erop staat; anders scanvolgorde (zoals bij de plaatkanten), dezelfde opname dezelfde kant.
        if any(('rim of a 30 cm record' in x[2] or 'printed label reads' in x[2]) and any(t not in label_scans for t in x[1]) for x in named):
            return named                                   # er zijn plaatscans (zonder vaste kant): niet raden
        # Plaatscans waarvan de rand niet te zien is (doorzichtig vinyl: Who's Your Momma 2022, roze): een grotere scan
        # (150-330 mm) in een insert/onbekend vak met een compleet label erop. Dan zijn er wél plaatkanten: elke plaatscan
        # naar de kant op zijn label, anders naar de gelezen labelscan waar hij het meest mee overlapt (en duidelijk
        # meer dan met een andere kant). Lukt dat niet voor elke plaatscan, dan niets veranderen (niet raden).
        with tempfile.TemporaryDirectory() as work6:
            recs = {}
            for x in named:
                if x[0] not in ('insert', 'unknown'):
                    continue
                for t in x[1]:
                    mm = scan_mm(t) if t not in label_scans else None
                    # en hij deelt drukwerk met een labelscan (hetzelfde label); een rond plaatje op een tekstvel niet
                    # (Fake It 2021: Scan 14-15, 0 punten met de labels)
                    if mm and 150 <= max(mm) <= 330 and feats.get(t) and any(
                            feats.get(u) and link(feats[t], feats[u])[0] >= 2 * MIN_LINK for u in label_scans):
                        c = label_circle(t, work6)
                        if c:
                            recs[t] = c
        if recs:
            side_of = {}
            for t, c in recs.items():
                sd = label_side(t, c)
                if sd is None and feats.get(t):
                    sc = {}
                    for u in label_scans:
                        if read[u] and feats.get(u):
                            sc[read[u]] = max(sc.get(read[u], 0), link(feats[t], feats[u])[0])
                    rk = sorted(sc.items(), key=lambda kv: -kv[1])
                    if rk and rk[0][1] >= 2 * MIN_LINK and (len(rk) == 1 or rk[0][1] >= 2 * rk[1][1]):
                        sd = rk[0][0]
                if sd is None:
                    return named
                side_of[t] = sd
            for u in label_scans:
                if read[u]:
                    side_of[u] = read[u]
            for x in named:
                x[1] = [t for t in x[1] if t not in side_of and t not in label_scans]
            named = [x for x in named if x[1]]
            for sd in sorted(set(side_of.values())):
                fs = sorted((t for t, v in side_of.items() if v == sd), key=number_key)
                named.append([sd, fs, f'record scans with a whole label (the rim of a clear record does not show) and separate label scans; '
                                      f'the label reads SIDE {sd[-1].upper()}'])
            for u in label_scans:
                if u not in side_of:
                    named.append(['unknown', [u], 'a separate scan of a record label; its side could not be read: choose it'])
            return named
        read0 = dict(read)
        taken = {read[s] for s in label_scans if read[s]}
        free = [sd for sd in (expected_sides or SIDES[:2]) if sd not in taken]
        by_side, left = {}, []
        for s in sorted(label_scans, key=number_key):
            sd = read[s]
            if sd is None and feats.get(s):
                # hetzelfde label nog eens gescand (iets anders bijgesneden of verschoven, niet gedraaid): zelfde kant
                # (Fake It 2021: Scan 9 en 10 zijn allebei label A)
                for sd_, fs_ in by_side.items():
                    if any(feats.get(t) and (lambda r: r[0] >= MIN_LINK and r[1] < 40 and r[2] < SAME_PLACE_DEG)(link(feats[s], feats[t])) for t in fs_):
                        sd = sd_; break
            if sd is None and free:
                sd = free.pop(0)
            read[s] = sd
            if sd:
                by_side.setdefault(sd, []).append(s)
            else:
                left.append(s)
        for x in named:
            x[1] = [t for t in x[1] if t not in label_scans]
        named = [x for x in named if x[1]]
        for sd, fs in sorted(by_side.items()):
            named.append(['label' + sd[-1], fs, f'a separate scan of the label of side {sd[-1].upper()}; no record scans: only the label'
                          + ('' if any(read0.get(t) for t in fs) else ' (side by scan order)')])
        for s in left:                                     # nooit een scan kwijtraken
            named.append(['unknown', [s], 'a separate scan of a record label; its side could not be read and all sides have a label: choose it'])
        return named
    where = {s: x for x in named for s in x[1]}
    record = [(s, x[0]) for x in named if x[0] in SIDES for s in x[1] if s not in label_scans]
    for s in sorted(label_scans, key=number_key):
        cur = where.get(s)
        side = read[s] if read[s] in sides else None
        how = f'its label reads SIDE {side[-1].upper()}' if side else None
        if side is None and cur is not None and cur[0] in SIDES:
            # ligt al bij een kant, maar de kant op het label is niet gelezen: laten liggen, wel zeggen (A en B lijken
            # vaak op elkaar, dus de overlap alleen bewijst de kant niet)
            cur[2] += f'; {os.path.basename(s)} is a separate scan of a label whose side could not be read: left here, check it'
            continue
        if side is None and feats.get(s):
            best = max(((link(feats[s], feats[r])[0], sd) for r, sd in record if feats.get(r)), default=(0, None))
            if best[0] >= MIN_LINK:
                side, how = best[1], f'it overlaps most with the record scans of side {best[1][-1].upper()}'
        if side is None:
            if cur is not None and all(t in label_scans for t in cur[1]) and cur[0] in ('insert', 'sticker', 'unknown'):
                cur[0], cur[2] = 'unknown', 'a separate scan of a record label; its side could not be read: choose it'
            continue
        if cur is sides[side]:
            continue
        if cur is not None:
            cur[1].remove(s)
        sides[side][1].append(s); sides[side][1].sort(key=number_key); where[s] = sides[side]
        sides[side][2] += f'; {os.path.basename(s)} is a separate scan of the label, placed here: {how}'
    # Plaatscan zonder zichtbare rand (doorzichtig vinyl: To Get Her Together 2022 Scan 17) in een insert of onbekend
    # vak: staat er een compleet label op en overlapt hij met de scans van één kant (de labelscans van die kant), dan
    # hoort hij bij die kant. Twee kanten even goed: niets doen.
    with tempfile.TemporaryDirectory() as work5:
        for x in [x for x in named if x[0] in ('insert', 'unknown') and x[1]]:
            for s in list(x[1]):
                if s in label_scans or not feats.get(s):
                    continue
                mm = scan_mm(s)
                circ = label_circle(s, work5) if mm and max(mm) <= 330 else None
                if circ is None:
                    continue
                score = {}
                for sd, g in sides.items():
                    score[sd] = max((link(feats[s], feats[t])[0] for t in g[1] if feats.get(t)), default=0)
                ranked = sorted(score.items(), key=lambda kv: -kv[1])
                if not ranked or ranked[0][1] < 2 * MIN_LINK:
                    continue                                # overlapt met geen enkele kant: geen plaatscan van deze plaat
                sd = label_side(s, circ)                    # de kant die op het label staat (A en B lijken vaak op elkaar)
                how = f'its label reads SIDE {sd[-1].upper()}' if sd in sides else None
                if how is None and (len(ranked) == 1 or ranked[0][1] >= 2 * ranked[1][1]):
                    sd, how = ranked[0][0], f'it overlaps with this side ({ranked[0][1]} points)'
                if how is None:
                    continue
                x[1].remove(s); sides[sd][1].append(s); sides[sd][1].sort(key=number_key)
                sides[sd][2] += f'; {os.path.basename(s)} shows a whole label and {how}: a record scan'
    return [x for x in named if x[1]]


def main():
    url, fmt, slots = sys.argv[1], sys.argv[2], [s for s in sys.argv[3].split(',') if s]
    scans = sys.argv[4:]
    set_order(scans)
    # Sleeve Transparant: feit van de gebruiker. Staat 'sticker' in de bestandsnaam van een scan, dan is het de
    # sticker, ook als er een stuk voorkant mee op de scan staat (dan herkent de app hem anders als voorkant).
    named_stickers = [s for s in scans if 'sticker' in os.path.basename(s).lower()]
    scans = [s for s in scans if s not in named_stickers]
    codes, imgs, roles = [], [], []
    if url:
        rid = re.search(r'/release/(\d+)', url); mid = re.search(r'/master/(\d+)', url)
        if not rid and mid:                                # master-link: de hoofduitgave daarvan
            main_ = json.loads(D.get(f'https://api.discogs.com/masters/{mid.group(1)}')).get('main_release')
            rid = re.search(r'/release/(\d+)', f'/release/{main_}') if main_ else None
        rel = json.loads(D.get(f'https://api.discogs.com/releases/{rid.group(1)}')) if rid else {}
        if not fmt:
            fmt = 'cd' if any((f.get('name') or '').lower() == 'cd' for f in rel.get('formats', [])) else 'vinyl'
        codes = [re.sub(r'\D', '', i.get('value', '')) for i in rel.get('identifiers', []) if i.get('type') == 'Barcode']
        for im in rel.get('images', []):
            try:
                arr = cv2.imdecode(np.frombuffer(D.get(im['uri']), np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    imgs.append(arr); roles.append(im.get('type', 'secondary'))
            except Exception:
                pass
    # Discogs' eigen hoesfoto: type 'primary'; zonder primary toont Discogs de eerste foto als hoes
    primary = roles.index('primary') if 'primary' in roles else (0 if imgs else None)
    # Discogs-foto's van de hoes: de hoofdfoto en de foto's die er duidelijk op lijken (vaak twee hoesfoto's)
    front_photos = set()
    if primary is not None:
        # alleen foto's waarin de hoes de hele foto vult (een foto van een tekstvel met de hoes er klein op telt niet:
        # Queen 2021, foto's 2 en 3)
        front_photos = {primary} | {n for k, n, M in D.matches(imgs[primary], imgs)
                                    if k >= 40 and real(M) and imgs[primary].shape[1] * float(np.hypot(M[0, 0], M[1, 0])) >= .7 * imgs[n].shape[1]}
    facts, why, feats, discs, rims, photo, backs = {}, {}, {}, {}, {}, {}, {}
    shown_mm = {}                                        # zo breed staat de hoesfoto op deze scan (mm)
    scale_candidates = set()
    scaled_records = set()
    paper_notes = {}
    with tempfile.TemporaryDirectory() as work:
        # Verkleinen en barcode lezen zijn losse programma's (sips, sleeve_ocr): drie scans tegelijk
        from concurrent.futures import ThreadPoolExecutor
        def prep(s):
            return small(s, work), thumb(s, work), (barcodes(s) if codes else [])
        with ThreadPoolExecutor(3) as pool:
            pre = dict(zip(scans, pool.map(prep, scans)))
        for n, s in enumerate(scans):
            print(f'SLEEVE_STEP looking at scan {n + 1} of {len(scans)}', file=sys.stderr, flush=True)
            sm, (im, scale), bcs = pre[s]
            if sm is not None:
                feats[s] = features(sm)
                backs[s] = background(sm)
                if fmt != 'cd':
                    rims[s] = record_rim(sm)
                    if fmt == 'vinyl' and .08 <= rims[s] < RECORD_RIM and scaled_record_evidence(sm):
                        scale_candidates.add(s)
            found = []
            if codes and any(same_barcode(b, c) for b in bcs for c in codes):
                found.append(('back', 'the release barcode is on this scan'))
            if im is None:
                continue
            if imgs:
                m = [x for x in D.matches(im, imgs) if real(x[2])]
                if m:
                    photo[s] = (m[0][1], m[0][0])          # de Discogs-foto die het best op deze scan past
                # Sleeve Transparant: een scan met de sticker op de hoes past soms beter op een andere Discogs-foto
                # (GF green 0001: 294 punten op een stickerfoto, 37 op de hoesfoto); dan de best passende hoesfoto.
                fm = next((x for x in m if x[1] in front_photos), None)
                if fm is not None and fm is not m[0] and fm[0] >= MIN_FRONT:
                    m = [fm] + [x for x in m if x is not fm]
                if m and m[0][1] in front_photos and m[0][0] >= MIN_FRONT:
                    # Feit: op de voorkant staat de hoesfoto op ware grootte (±31 cm); een tekstvel of insert met de
                    # hoes erop toont hem kleiner (Queen 2021: twee tekstvellen met de hoesfoto). Grootte van de
                    # Discogs-foto in deze scan: zijn breedte via de overeenkomst (schaal) en de resolutie in mm.
                    dpi_ = scan_dpi.read(s)
                    sM = float(np.hypot(m[0][2][0, 0], m[0][2][1, 0]))
                    shown = (imgs[m[0][1]].shape[1] / sM) / (dpi_ * scale / 25.4) if dpi_ and sM > 0 else None
                    if shown:
                        shown_mm[s] = shown
                    if shown is None or shown >= FRONT_MIN_MM.get(fmt, 96):
                        found.append(('front', f'matches the Discogs cover photo ({m[0][0]} points' + (f', shown {shown:.0f} mm wide)' if shown else ')')))
                    else:
                        why[s] = f'shows the cover photo only {shown:.0f} mm wide: a print of the cover, not the sleeve'
            if fmt == 'cd':
                dpi = scan_dpi.read(s)
                if dpi and has_cd(im, dpi * scale / 25.4):
                    discs[s] = True
                    found.append(('cd', 'a 120 mm disc lies on this scan'))
            if len(found) == 1:            # twee feiten die elkaar tegenspreken: niet kiezen
                facts[s] = found[0]
            elif found:
                why[s] = 'conflicting facts: ' + ', '.join(p for p, _ in found)
        # Sleeve Transparant: geen enkele scan past (genoeg) op de hoesfoto van deze release (slechte foto op Discogs,
        # GF green 2022: 25-29 punten). Feit: andere uitgaven van hetzelfde album (zelfde master) hebben dezelfde hoes;
        # dan hun hoesfoto's, met dezelfde eisen (genoeg punten, op ware grootte).
        if imgs and rel.get('master_id') and not any(f[0] == 'front' for f in facts.values()):
            extra = other_version_covers(rel, fmt)
            for s in scans:
                if s in facts or s in why or not extra:
                    continue
                im, scale = pre[s][1]
                if im is None:
                    continue
                m = [x for x in D.matches(im, extra) if real(x[2])]
                if not m or m[0][0] < MIN_FRONT:
                    continue
                dpi_ = scan_dpi.read(s)
                sM = float(np.hypot(m[0][2][0, 0], m[0][2][1, 0]))
                shown = (extra[m[0][1]].shape[1] / sM) / (dpi_ * scale / 25.4) if dpi_ and sM > 0 else None
                if shown is None or shown >= FRONT_MIN_MM.get(fmt, 96):
                    facts[s] = ('front', f'matches the cover photo of another release of this album on Discogs ({m[0][0]} points'
                                + (f', shown {shown:.0f} mm wide)' if shown else ')'))
    usable = [s for s in scans if s in feats] or scans
    # Sleeve Transparant: losse uitsneden (vinyl). Een labelscan blijft gewoon meedoen in de stapels (hij hoort bij
    # zijn kant en overlapt vaak met de plaatscans); na de kanten gaat hij naar de juiste kant (place_label_scans).
    # Een stickerscan gaat uit de reeks voordat de stapels gemaakt worden: hij overlapt met niets en brak anders een
    # stapel in tweeën (Paradise 2014: een strook-sticker tussen de scans van de hoes). Niet als hij overlapt met een
    # buurscan: dan is het een stuk van een groter vel (Hotel New York: tekststroken in delen).
    label_scans, sticker_closeups = {}, []
    if fmt == 'vinyl':
        order_ = sorted(usable, key=number_key)
        with tempfile.TemporaryDirectory() as work4:
            for i_, s in enumerate(order_):
                mm = scan_mm(s)
                if not mm or max(mm) > CLOSEUP_MM or min(mm) < 20 or rims.get(s, 0.0) >= RECORD_RIM or s in facts:
                    continue
                c = label_circle(s, work4)
                if c:
                    label_scans[s] = c
                    continue
                nb = [t for t in order_[max(0, i_ - 1):i_ + 2] if t != s]
                if any(feats.get(s) and feats.get(t) and link(feats[s], feats[t])[0] >= MIN_LINK for t in nb):
                    continue
                why_ = small_sticker(s, mm, work4)
                if why_:
                    sticker_closeups.append((s, why_))
        usable = [s for s in usable if s not in {x for x, _ in sticker_closeups}]
    # A single apparent circle on a sleeve is insufficient. Require a
    # second independently scanned view in the same overlap stack.
    if scale_candidates:
        for group in stacks(usable, feats, discs):
            supported = set(group) & scale_candidates
            if len(supported) >= 2:
                scaled_records.update(supported)
        for path in scaled_records:
            rims[path] = RECORD_RIM
    # alleen scans van een plaat (de rand van een lp is te zien): optie A bestaat alleen bij vinyl. Een hoes kan
    # toevallig scans met en zonder witte klep langs de rand hebben (Deena inside).
    rec = [s for s in usable if rims.get(s, 0.0) >= RECORD_RIM]
    lids = [s for s in rec if backs.get(s) == 'lid']; sheets = [s for s in rec if backs.get(s) == 'sheet']
    if len(lids) >= 2 and len(sheets) >= 2:
        # Optie A (doorschijnend vinyl): per positie een scan met de witte klep en een met een vel op de plaat. Een
        # vel-scan ligt op dezelfde plek als zijn klep-scan: eerst de klep-scans tot stapels, dan elke vel-scan bij de
        # stapel waar hij het best op past. (Anders begon bij elke vel-scan een nieuwe stapel: 'zelfde plek'.)
        groups = stacks(lids, feats, discs)
        rest = [s for s in usable if s not in lids and s not in sheets]
        for sh in sorted(sheets, key=number_key):
            best = max(((link(feats[sh], feats[x])[0], gi) for gi, g in enumerate(groups) for x in g), default=(0, None))
            if best[0] >= MIN_LINK:
                groups[best[1]].append(sh)
            else:
                groups.append([sh])
        groups += stacks(rest, feats, discs) if rest else []
        groups.sort(key=lambda g: min(number_key(x) for x in g))       # scanvolgorde (kant A voor kant B)
    else:
        groups = stacks(usable, feats, discs) if scans else []
    if fmt == 'vinyl':
        groups = related_paper_groups(groups, feats, rims, paper_notes)
    def best_photo(g):
        got = [photo[s] for s in g if s in photo]
        return max(got, key=lambda x: x[1]) if got else (None, 0)
    named = []
    for g in groups:
        parts = {facts[s][0] for s in g if s in facts}
        if 'back' in parts and 'front' in parts:
            # Feit: een barcode staat alleen op de achterkant; een achterkant kan wel de hoes klein afbeelden
            parts = {'back'}
        part, reason, score = 'unknown', 'no fact for this part yet: choose it', 0
        if fmt != 'cd' and g and statistics.median(rims.get(s, 0.0) for s in g) >= RECORD_RIM:
            # de rand van een plaat weegt zwaarder dan een Discogs-foto: een label toont vaak de hoes
            part, reason = 'side', 'the rim of a 30 cm record is on these scans'
            if any(s in scaled_records for s in g):
                reason = 'the rim, round label and spindle hole agree at twice the stated scale; check the scan DPI before export'
        elif len(parts) == 1:
            part = parts.pop(); src = next(s for s in g if s in facts and facts[s][0] == part); reason = facts[src][1] + f' ({os.path.basename(src)})'
            score = best_photo(g)[1]
        elif len(parts) > 1:
            reason = 'conflicting facts: ' + ', '.join(sorted(parts))
        named.append([part, g, reason, score])
    # Feit: de stapel met de barcode is de back; daarmee is ook bekend welke Discogs-foto de back toont.
    # Een andere stapel die het best op diezelfde foto past (opnieuw gescand) is ook de back.
    backs = [x for x in named if x[0] == 'back']
    bps = {best_photo(x[1])[0] for x in backs}
    if len(bps) == 1:                                       # alle stapels met de barcode passen op dezelfde foto
        bp = bps.pop()
        for x in named:
            if x[0] == 'unknown' and bp is not None and best_photo(x[1])[0] == bp and bp not in front_photos and best_photo(x[1])[1] >= MIN_FRONT:
                x[0], x[2] = 'back', f'matches the Discogs photo of the back, like the scan with the barcode'
    # De hoes bestaat één keer, maar een insert of poster kan precies dezelfde afbeelding dragen. Lijken
    # meer stapels op de hoes, dan is niet vast te stellen welke de hoes is: geen naam (jij kiest).
    fronts = [x for x in named if x[0] == 'front']
    if len(fronts) > 1 and fmt != 'cd':
        # Feit: een lp-hoes is minstens 305 mm in beide richtingen. Wat de scans van een stapel samen bedekken,
        # moet dan ook minstens zo groot zijn; is het in één richting kleiner, dan is het niet de hoes maar een
        # kleiner vel met de hoes erop (Urban Solitude groen: een afdruk van de hoes in 2 scans).
        for x in fronts:
            e = extent_mm(x[1], feats)
            if e is not None and min(e) < 305:
                x[0], x[2] = 'unknown', f'looks like the cover, but these scans cover only {min(e):.0f} mm one way: smaller than a sleeve (305 mm)'
        fronts = [x for x in named if x[0] == 'front']
    if len(fronts) > 1 and fmt != 'cd':
        # Feit: een boekje of insert zit ín de hoes en is dus kleiner; staat de hoesfoto erop, dan kleiner dan op
        # de hoes zelf (TA gold: hoes 315-317 mm, voorkant van het boekje 304-306 mm). Is één stapel duidelijk de
        # grootste afdruk (minstens 5 mm groter dan elke andere, en niet groter dan een hoes), dan is dat de voorkant.
        med = {id(x): float(np.median([shown_mm[t] for t in x[1] if t in shown_mm])) for x in fronts if any(t in shown_mm for t in x[1])}
        if len(med) == len(fronts):
            big = max(fronts, key=lambda x: med[id(x)])
            if med[id(big)] <= 330 and all(med[id(big)] - med[id(x)] >= 5 for x in fronts if x is not big):
                for x in fronts:
                    if x is not big:
                        x[0], x[2] = 'unknown', (f'shows the cover photo {med[id(x)]:.0f} mm wide, the front {med[id(big)]:.0f} mm: '
                                                 'a smaller print of the cover (booklet or insert)')
        fronts = [x for x in named if x[0] == 'front']
    if len(fronts) > 1:
        for x in fronts:
            x[0], x[2] = 'unknown', f'{len(fronts)} stacks look like the cover (a print of it on an insert?): choose which is the front'
    # De back (barcode) bestaat één keer. Opeenvolgende back-stapels: opnieuw gescand, één onderdeel.
    # Niet opeenvolgend: de sterkste overeenkomst houdt de naam.
    for part in ('back',):
        i = 0
        while i < len(named) - 1:
            repeated = 'separate scan round' in paper_notes.get(named[i + 1][1][0], '')
            # Sleeve Transparant: bedekt elke stapel al de hele hoes, dan zijn het twee volledige scanrondes, geen helften:
            # niet samenvoegen (Hotel New York 2022: 8 scans van de achterkant pasten samen niet); de beste blijft (hieronder)
            both_whole = all(complete(named[j][1], feats) for j in (i, i + 1) if all(t in feats for t in named[j][1]))
            if named[i][0] == part and named[i + 1][0] == part and not repeated and not both_whole:
                named[i][1] += named[i + 1][1]; named[i][3] = max(named[i][3], named[i + 1][3]); del named[i + 1]
            else:
                i += 1
        same = [x for x in named if x[0] == part]
        if len(same) > 1:
            keep = max(same, key=lambda x: x[3])
            for x in same:
                if x is not keep:
                    x[0], x[2] = 'unknown', f'also looks like the {part}, but {os.path.basename(keep[1][0])} matches better: choose it'
    # Een release heeft één front en één back. Zijn beide gevonden, dan is elk ander vel (geen plaat, geen cd)
    # een insert, of de binnenkant van een uitklaphoes als het samen breder is dan 45 cm (±62 x 31 cm).
    # Sleeve Transparant: detailscan van het venster in de voorkant (US white: de EYES-scan 0009). Feit: één losse scan die
    # met de voorkant overlapt (zelfde drukwerk) en een volledig rechthoekig venster toont. Eigen vak 'front'; de
    # bridge gebruikt hem als detail voor het venster (scan_window_detail.merge_detail), niet als extra stuk hoes.
    fronts_ = [x for x in named if x[0] == 'front']
    if fmt == 'vinyl' and len(fronts_) == 1:
        import scan_window_detail as WD
        for x in named:
            if x[0] != 'unknown' or len(x[1]) != 1 or x[1][0] not in feats:
                continue
            overlap = max((link(feats[x[1][0]], feats[f])[0] for f in fronts_[0][1] if f in feats), default=0)
            # een scan met een sticker erop is de sticker (TA clear 0019: de stickerrand leek een venster)
            if overlap >= MIN_LINK and WD.has_window(x[1][0]) and not sticker_on(x[1][0]):
                x[0], x[2] = 'front', (f'a detail scan of the window in the front ({overlap} points overlap with the front, '
                                       'a whole rectangular opening): used for the window')
    # Sticker: een losse scan met een kleine gesloten vorm met afgeronde hoeken of een cirkel (zijn plastic hoes eromheen)
    if sticker_closeups:
        # dezelfde sticker twee keer gescand (overlap): één vak; de app kent één sticker-vak per uitgave
        sgroups = []
        for s, why_ in sticker_closeups:
            hit = next((g for g in sgroups if feats.get(s) and any(feats.get(t) and link(feats[s], feats[t])[0] >= MIN_LINK for t in g[1])), None)
            if hit:
                hit[1].append(s)
            else:
                sgroups.append(['sticker', [s], why_, 0])
        for k_, g in enumerate(sgroups):
            if k_ or named_stickers or any(x[0] == 'sticker' for x in named):
                g[0], g[2] = 'unknown', g[2].replace(': the sticker', '') + '; another sticker is already found: choose it'
        named += sgroups
    own_sticker = next((x for x in named if x[0] == 'sticker' and x[2].startswith('a separate')), None)
    if not named_stickers and (own_sticker or not any(x[0] == 'sticker' for x in named)):
        with tempfile.TemporaryDirectory() as work3:
            for x in named:
                if not own_sticker and x[0] == 'unknown' and len(x[1]) == 1 and x[1][0] not in label_scans:
                    why_ = is_sticker(x[1][0], work3)
                    if why_:
                        x[0], x[2] = 'sticker', f'{why_} lies on this scan'
            # Sleeve Transparant: de sticker op de hoes gescand (een stuk voorkant mee op de scan) komt in de stapel
            # van de voorkant of achterkant terecht. Feit: op die ene scan ligt een kleine vorm met afgeronde hoeken;
            # op de gewone stukken niet. Alleen als het precies één scan is, en de stapel zonder hem nog de hele hoes
            # (≥ 305 mm beide kanten) bedekt: dan is het een losse scan van de sticker, geen stuk van de hoes.
            # (heeft de sticker een eigen losse scan, dan blijft de scan van de hoes mét sticker gewoon bij de hoes: zonder
            # die scan miste de voorkant 5,5% (Hotel New York 2023, 00 3); de hoes is hoe hij is, met sticker)
            if not any(x[0] == 'sticker' for x in named) and fmt == 'vinyl':
                hits = []
                for x in named:
                    if x[0] in ('front', 'back') and len(x[1]) >= 2:
                        for s_ in x[1]:
                            if facts.get(s_, ('',))[0] == 'back' or s_ in label_scans:
                                continue                    # de scan met de barcode is de achterkant (Paradise 2014: Scan 6)
                            why_ = is_sticker(s_, work3)
                            if why_:
                                hits.append((x, s_, why_))
                if len(hits) > 1:
                    # Sleeve 0.5.51: een vorm met afgeronde hoeken in het drukwerk (een tegel op de foto, Together Alone
                    # 2016 zwart 0002) staat ook op de andere scans van die stapel; een sticker op de krimpfolie (0001) niet.
                    hits = [h for h in hits if not shape_in_print(h[1], [t for t in h[0][1] if t != h[1]], work3)]
                if len(hits) == 1:
                    x, s_, why_ = hits[0]
                    rest = [t for t in x[1] if t != s_]
                    e = extent_mm(rest, feats) if all(t in feats for t in rest) else None
                    if e is not None and min(e) >= 305:
                        x[1] = rest
                        x[2] = x[2].replace(f' ({os.path.basename(s_)})', '')
                        if own_sticker:
                            named.append(['unknown', [s_], f'{why_} lies on this scan, with part of the {x[0]} (the {x[0]} is whole without it); '
                                                           f'the sticker also has its own scan ({os.path.basename(own_sticker[1][0])}): not needed', 0])
                        else:
                            named.append(['sticker', [s_], f'{why_} lies on this scan, with part of the {x[0]} (the {x[0]} is whole without it)', 0])
    # De maat is wat de deelscans samen bedekken (glas, dus ruimer dan het vel): een vel van 30 cm in 4 delen
    # 330-460 mm, een opengeklapte hoes ±670 mm (gemeten op de vaste toets). Ligt er een binnenkant in de
    # release, dan krijgt geen andere stapel 'insert': een los stuk van die binnenkant zou anders insert heten.
    if fmt != 'cd' and any(x[0] == 'front' for x in named) and any(x[0] == 'back' for x in named):
        open_ = [x for x in named if x[0] == 'unknown' and (x[2].startswith('no fact') or 'smaller than a sleeve' in x[2] or 'smaller print of the cover' in x[2])
                 and 'same position' not in paper_notes.get(x[1][0], '') and not all(t in label_scans for t in x[1])]   # tweede scanronde, labelscan: geen insert
        size = {id(x): extent_mm(x[1], feats) for x in open_}
        for x in open_:
            if size[id(x)] and max(size[id(x)]) > 580:
                x[0], x[2] = 'inside', f'front and back are found; these scans cover {max(size[id(x)]):.0f} mm: the inside of a gatefold'
        inside_scans = [t for x in named if x[0] == 'inside' for t in x[1]]
        for x in open_:
            if x[0] == 'unknown' and size[id(x)] and max(size[id(x)]) < 520:
                # Sleeve Transparant: ligt er een binnenkant, dan is een stapel die drukwerk met die binnenkant deelt een
                # los stuk ervan (geen insert); een stapel zonder gedeeld drukwerk is een eigen vel (Deena & Jim wit:
                # twee songtekstvellen naast de uitklaphoes)
                if inside_scans and any(feats.get(t) and feats.get(u) and link(feats[t], feats[u])[0] >= MIN_LINK
                                        for t in x[1] for u in inside_scans):
                    continue
                x[0], x[2] = 'insert', 'front and back are found and this is another sheet: an insert'
    # Sleeve 0.5.51 - feit: zonder plaat (geen labelkant, geen labelscan) en zonder achterkant (geen barcode) zijn
    # meerdere losse vellen geen hoes maar losse bladen: een boekje of inserts (de boekjes van Together Alone, Urban
    # Solitude, Graduated Fool en Hotel New York, apart gescand in extra_scans_all_editions: pagina 1 toont de hoesfoto
    # en werd 'front', de rest bleef open). Eén vel alleen blijft zoals het is (een losse voorkant kan ook).
    if fmt != 'cd' and not any(x[0] in ('back', 'sidea', 'sideb', 'sidec', 'sided', 'labela', 'labelb', 'inside') for x in named) \
            and not label_scans and not any(rims.get(t, 0.0) >= RECORD_RIM for x in named for t in x[1]):
        sheets_ = [x for x in named if x[0] in ('front', 'unknown') and not all(t in label_scans for t in x[1])
                   and 'same position' not in paper_notes.get(x[1][0], '')]
        sizes_ = [extent_mm(x[1], feats) if len(x[1]) > 1 and all(t in feats for t in x[1]) else None for x in sheets_]
        if len(sheets_) >= 2 and all(e is None or max(e) < 520 for e in sizes_):
            for x in sheets_:
                cover_ = x[0] == 'front'
                x[0], x[2] = 'insert', ('no record and no sleeve back among these scans: a separate printed sheet (booklet or insert)'
                                        + ('; it shows the cover picture' if cover_ else ''))
    # Cd (jewelcase): er zijn precies twee delen van ±150 mm breed, de achterkaart en het inlay (de twee kanten van
    # het kaartje in de lade). Is de achterkaart gevonden, dan is een andere losse scan met zo'n kaart het inlay.
    # Een vierkant kaartje van 120 mm kan insert, inside of de achterkant van een promo zijn: geen naam.
    if fmt == 'cd' and any(x[0] == 'back' for x in named) and not any(x[0] == 'inlay' for x in named):
        wide = []
        with tempfile.TemporaryDirectory() as work2:
            for x in named:
                if x[0] == 'unknown' and len(x[1]) == 1:
                    sz = card_size(x[1][0], work2)
                    if sz and 145 <= max(sz) <= 154 and 114 <= min(sz) <= 120:   # ±150 x 118 (een boekje van 120 mm kwam op de ondergrens 136 uit)
                        wide.append((x, sz))
        if len(wide) == 1:
            x, sz = wide[0]
            x[0], x[2] = 'inlay', f'the back is found; this card is {max(sz):.0f} x {min(sz):.0f} mm, the other card of that size in a jewel case: the inlay'
    named += [['sticker', [s], f'the file name says sticker ({os.path.basename(s)})', 0] for s in named_stickers]
    named = [x[:3] for x in named]
    if fmt == 'cd':                       # één cd-schijf: meer stapels met een cd is geen feit meer
        cds = [x for x in named if x[0] == 'cd']
        if len(cds) > 1:
            for x in cds:
                x[0], x[2] = 'unknown', 'more than one scan has a disc'
    if fmt == 'vinyl':
        named = assign_printed_sides(named)
    # Sleeve Transparant: Discogs noemt het aantal kanten (slots). Kloppen de plaatstapels daar niet mee en is de kant
    # niet van het label gelezen, dan is 'op scanvolgorde' geen feit meer (Deena & Jim zwart: 3 stapels, 4 kanten; de
    # labels lijken op elkaar en scans van twee kanten kwamen in één stapel). Dan niet raden: jij deelt ze in.
    expected = [s_ for s_ in slots if s_ in SIDES]
    pending = [x for x in named if x[0] == 'side']
    if fmt == 'vinyl' and expected and pending and len(pending) + sum(x[0] in SIDES for x in named) != len(expected):
        n_ = len(pending) + sum(x[0] in SIDES for x in named)
        for x in pending:
            x[0], x[2] = 'unknown', (f'{x[2]}; the record scans form {n_} stacks but this release has {len(expected)} sides on Discogs, '
                                     'and the side could not be read on the labels: put these scans into the side slots yourself')
    k = 0
    for x in named:                       # kanten in scanvolgorde (na het verwerken controleert het label)
        if x[0] == 'side':
            if k < len(SIDES):
                x[0] = SIDES[k]; x[2] += f'; side {SIDES[k][-1].upper()} by scan order (the label is checked after processing)'
            else:
                x[0] = 'unknown'
            k += 1
    if fmt == 'vinyl' and label_scans:
        named = place_label_scans(named, label_scans, feats, expected)
    # vangnet: elke scan die erin ging, staat in een vak (anders 'onbekend')
    placed = {t for x in named for t in x[1]}
    for s_ in sys.argv[4:]:
        if s_ not in placed:
            named.append(['unknown', [s_], 'no fact for this part yet: choose it']); placed.add(s_)
    for item in named:
        part, g, reason = item
        if g and g[0] in paper_notes:
            item[2] = reason + '; ' + paper_notes[g[0]]
        for s in g:
            why.setdefault(s, reason)
    print(json.dumps({'groups': [{'part': p, 'files': g, 'why': r} for p, g, r in named], 'why': why}, ensure_ascii=False))


if __name__ == '__main__':
    main()
