#!/usr/bin/env python3
"""Rechtop zetten op feiten, zonder regels.

Feiten: drukwerk (hoes, label) is gemaakt om gelezen te worden, en welke tekst
erop staat is bekend (artiest, titel, tracks, label van de release op Discogs).
Lees het onderdeel in alle vier de standen met de tekstherkenning van macOS; in de
stand waarin die bekende tekst terug te lezen is, staat het rechtop. Is er geen
duidelijke winnaar: TWIJFEL, de gebruiker beslist.
"""
import argparse, difflib, json, math, os, re, subprocess, sys, tempfile, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OCR = os.path.join(HERE, 'sleeve_ocr')
READABLE_MIN = 40     # zonder bekende tekst: zoveel letters echte tekst minstens in de beste stand (zoals voor een hoes)
CLEAR = 2.0          # de beste stand moet minstens zoveel keer meer tekst opleveren dan de op één na beste
MATCH = 0.85         # zo gelijk moet een gelezen stuk tekst aan een bekend woord zijn


def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())


def found(known, line):
    """Komt het bekende stuk tekst (genormaliseerd) in de gelezen regel voor?"""
    if len(known) < 3 or len(line) < len(known) - 1:
        return False
    best = max(difflib.SequenceMatcher(None, known, line[i:i + len(known)]).ratio() for i in range(0, max(1, len(line) - len(known) + 1)))
    return best >= MATCH


DEAD_FLAG = os.path.join(tempfile.gettempdir(), 'sleeve-ocr-unavailable')


def ocr_unavailable():
    """Reageerde de tekstherkenning van macOS de afgelopen 10 minuten helemaal niet (ook snel niet)?"""
    try:
        return time.time() - os.path.getmtime(DEAD_FLAG) < 600
    except OSError:
        return False


def run_ocr(args, timeout):
    """Eén aanroep van sleeve_ocr met een harde tijdgrens. Reageert ook het snelle lezen niet
    (Mac mini: minutenlang niets), dan 10 minuten lang geen tekst meer lezen: dan bepalen de
    Discogs-foto of jouw keuze de stand, in plaats van dat elk onderdeel minuten wacht."""
    if ocr_unavailable():
        return ''
    try:
        return subprocess.run([OCR] + args, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        if '--boxes' not in args:                          # snel lezen hing: tekstherkenning is er niet
            try:
                open(DEAD_FLAG, 'w').close()
            except OSError:
                pass
        return ''


def read_rotations(path, max_side=2200):
    """Per stand (graden dat het beeld met de klok mee moet): de gelezen regels."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 4:                                   # plaat: buiten de cirkel donker maken
        a = im[:, :, 3:4] / 255.0
        im = (im[:, :, :3] * a + 40 * (1 - a)).astype(np.uint8)
    s = max_side / max(im.shape[:2])
    small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'x.png'); cv2.imwrite(p, small)
        out = ''
        for attempt in range(2):                           # Vision geeft soms (de eerste keer) niets terug
            out = run_ocr([p], 15)
            if out.strip() or ocr_unavailable():
                break
            time.sleep(0.3)
    lines = {0: [], 90: [], 180: [], 270: []}
    for line in out.splitlines():
        try:
            deg, _, text = line.split('\t', 2)
        except ValueError:
            continue
        lines[int(deg)].append(text)
    return lines



def faint_known_lines(image, words):
    """Measure low-contrast print twice; never change export pixels."""
    if not words:
        return []
    readings = []
    for size in (1600, 2200):
        scale = min(1., size / max(image.shape[:2]))
        view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(view[:, :, :3], cv2.COLOR_BGR2GRAY)
        low, high = np.percentile(gray, [.5, 99.5])
        if not 3 <= high - low < 64:
            return []
        contrast = np.clip((gray.astype(np.float32)-low)*255/(high-low), 0, 255).astype(np.uint8)
        lines = ocr_lines(cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR))
        readings.append([line for line in lines if any(found(norm(word), norm(line['text'])) for word in words)])
    candidates = []
    for quarter in (0, 90, 180, 270):
        sets = []
        for lines in readings:
            sets.append({norm(word) for word in words if any(
                int(round(line['direction']/90)) % 4 * 90 == quarter
                and found(norm(word), norm(line['text'])) for line in lines)})
        shared = sets[0] & sets[1]
        if sum(map(len, shared)) >= 12:
            candidates.append(quarter)
    if len(candidates) != 1:
        return []
    return [line for line in readings[0]
            if int(round(line['direction']/90)) % 4 * 90 == candidates[0]]


def readable_text_score(lines):
    """Per stand: hoeveel echte woorden Vision leest (ook tekst die niet van de release is).
    Feit: tekst leest alleen als woorden (Hoofdletter+kleine letters, of alles hoofdletters, met
    een klinker) als hij rechtop staat; ondersteboven of gekanteld wordt het een mengsel als 'aJaAas'."""
    score, hits = {}, {}
    for deg, texts in lines.items():
        good = []
        for t in texts:
            n = sum(len(tok) for tok in t.split() if WORD.match(tok) and re.search('[aeiouyAEIOUY]', tok))
            if n:
                good.append((n, t))
        hits[deg] = [t for _, t in sorted(good, reverse=True)]; score[deg] = sum(n for n, _ in good)
    return score, hits


WORD = re.compile(r'^[(\'"]?(?:[A-Z]?[a-z]{2,}|[A-Z]{3,})[.,;:!?)\'"]?$')


def sparse_text_orientation(path, first):
    """Confirm a short title by reading the same words at a second scale."""
    second = read_rotations(path, max_side=3200)
    return corroborated_short_text(first, second)


def corroborated_short_text(first, second):
    """Keep the word-count gate; corroborate a short OCR error by its full line."""
    candidates = {}
    for angle in first:
        def tokens(lines):
            return {token.lower() for text in lines for token in re.findall(r"[A-Za-z]{3,}", text)
                    if re.search(r"[aeiouy]", token, re.I)}
        common = tokens(first[angle]) & tokens(second[angle])
        if len(common) >= 2 and sum(map(len, common)) >= 12:
            candidates[angle] = common
            continue
        # A short word can be misread while the full phrase agrees. Require
        # two exact repeated words, a complete reading meeting the existing
        # length gate, and the same fuzzy-match limit used for known text.
        def valid_words(text):
            return {norm(token) for token in text.split()
                    if WORD.match(token) and re.search(r'[aeiouy]', token, re.I)}
        for left in first[angle]:
            for right in second[angle]:
                left_words, right_words = valid_words(left), valid_words(right)
                shared = left_words & right_words
                if (len(shared) >= 2
                        and max(sum(map(len, left_words)), sum(map(len, right_words))) >= 12
                        and difflib.SequenceMatcher(None, norm(left), norm(right)).ratio() >= MATCH):
                    candidates[angle] = shared
    if len(candidates) != 1:
        return None
    angle, words = next(iter(candidates.items()))
    return {'rotate_clockwise': angle, 'verdict': 'LEESBAAR',
            'read_upright': sorted(words) + ['same short text at two scales']}


def numbered_track_lines(lines, shape):
    """Identify numbered track rows inside the label, excluding circular text."""
    height, width = shape[:2]
    result = []
    for line in lines:
        x0, y0, x1, y1 = line['box']
        cx, cy = (x0+x1)/2, (y0+y1)/2
        if (re.match(r"^\s*0?[1-9]\d?\s*[.)]?\s*[A-Za-z]", line['text'])
                and len(re.findall(r"[A-Za-z]", line['text'])) >= 6
                and abs(cx-width/2) < .38*width and abs(cy-height/2) < .38*height
                and line['width'] > 2.5*line['height']):
            result.append(line)
    return result


def known_text_score(lines, words):
    """Per stand: welke bekende woorden terug te lezen zijn, en hun totale lengte."""
    score, hits = {}, {}
    for deg, texts in lines.items():
        got = [w for w in words if any(found(norm(w), norm(t)) for t in texts)]
        hits[deg] = got; score[deg] = sum(len(norm(w)) for w in got)
    return score, hits


# sleeve_ocr leest met CGImagePropertyOrientation .up/.right/.down/.left (0/90/180/270);
# '.right' betekent: het beeld moet 90 graden met de klok mee om rechtop te staan.
LABEL_MM = 105.0     # feit: een 12"-label is ±100 mm; iets ruimer zodat de rand meedoet
ROTATE = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def read_image(im):
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'x.png'); cv2.imwrite(p, im)
        out = ''
        for attempt in range(2):                           # Vision geeft soms (de eerste keer) niets terug
            out = run_ocr([p], 15)
            if out.strip() or ocr_unavailable():
                break
            time.sleep(0.3)
    lines = {0: [], 90: [], 180: [], 270: []}
    for line in out.splitlines():
        try:
            deg, _, text = line.split('\t', 2)
        except ValueError:
            continue
        lines[int(deg)].append(text)
    return lines


def text_level(label, angle, r_px, hole_px):
    """Feit: de tekstregels op een label zijn recht en evenwijdig. Rond de grove hoek
    in stappen van 0,1 graad draaien; waar de regels het scherpst horizontaal
    liggen (rij-voor-rij het grootste verschil in drukwerk) staat het label recht."""
    g = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    ink = (mag > np.percentile(mag, 90)).astype(np.float32)
    n = label.shape[0]; c = n / 2
    keep = np.zeros_like(ink); cv2.circle(keep, (int(c), int(c)), int(r_px * .85), 1, -1)
    cv2.circle(keep, (int(c), int(c)), int(hole_px * 2), 0, -1)
    ink *= keep
    best = None
    for a in np.arange(angle - 4, angle + 4.01, 0.1):
        R = cv2.getRotationMatrix2D((c, c), -a, 1.0)
        prof = cv2.warpAffine(ink, R, (n, n)).sum(1)
        score = float(np.var(prof))
        if best is None or score > best[0]:
            best = (score, float(a))
    return best[1]


SLOW_FLAG = os.path.join(tempfile.gettempdir(), 'sleeve-accurate-ocr-slow')


def accurate_is_slow():
    """Hing het nauwkeurige lezen de afgelopen 10 minuten? (gedeeld tussen alle onderdelen)"""
    try:
        return time.time() - os.path.getmtime(SLOW_FLAG) < 600
    except OSError:
        return False


def mark_accurate_slow():
    try:
        open(SLOW_FLAG, 'w').close()
    except OSError:
        pass


def ocr_lines(view):
    """Gelezen regels met hun kader in pixels (x0, y0, x1, y1), via sleeve_ocr --boxes."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'x.png'); cv2.imwrite(p, view)
        out = ''
        slow = accurate_is_slow()
        for attempt in range(3):                           # Vision geeft soms (vaak de eerste keer) niets terug: opnieuw
            # Nauwkeurig lezen draait op de Neural Engine: normaal 0,5 s, maar het kan minuten
            # hangen als die bezet is of zijn model nog voorbereidt (Mac mini: 4,5 min voor één cd).
            # Daarom kort proberen; hangt het, dan 10 minuten lang meteen snel lezen (CPU).
            mode = '--fast-boxes' if (slow or attempt >= 1) else '--boxes'
            t0 = time.time()
            # SLEEVE_OCR_ACCURATE_WAIT: een tweede poging mag langer wachten (koude start TA zwart sticker: 98 s, daarna 0,4 s)
            out = run_ocr([p, mode], float(os.environ.get('SLEEVE_OCR_ACCURATE_WAIT', 8)) if mode == '--boxes' else 15)
            if mode == '--boxes' and not out.strip() and time.time() - t0 >= float(os.environ.get('SLEEVE_OCR_ACCURATE_WAIT', 8)) - .5:
                mark_accurate_slow(); slow = True
            if out.strip() or ocr_unavailable():
                break
            time.sleep(0.3 * (attempt + 1))
    h, w = view.shape[:2]; lines = []
    for row in out.splitlines():
        try:
            conf, text, quad = row.split('\t')
            pts = np.array([[float(v) for v in q.split(',')] for q in quad.split(';')])
        except ValueError:
            continue
        xs, ys = pts[:, 0] * w, (1 - pts[:, 1]) * h
        P = np.c_[xs, ys]                                  # tl, tr, br, bl
        direction = math.degrees(math.atan2(P[0][1] - P[1][1], P[1][0] - P[0][0]))   # leesrichting, tegen de klok in vanaf rechts
        height = (np.linalg.norm(P[0] - P[3]) + np.linalg.norm(P[1] - P[2])) / 2   # letterhoogte, ook als de regel scheef staat
        lines.append({'text': text, 'box': (xs.min(), ys.min(), xs.max(), ys.max()), 'height': height,
                      'width': (np.linalg.norm(P[0] - P[1]) + np.linalg.norm(P[3] - P[2])) / 2, 'direction': direction})
    return lines


def line_angle(view, box, around):
    """Hoek van één tekstregel: een ruim stuk rond de regel draaien in stappen van
    0,1 graad en daarna precies de regel eruit nemen (zo vallen er geen lege hoeken
    in de meting); recht is waar de boven- en onderkanten van de letters per rij het
    scherpst samenvallen."""
    x0, y0, x1, y1 = box; hgt = y1 - y0; wid = x1 - x0
    pad = int(.08 * wid + hgt)                                 # ruim genoeg voor 4 graden over de halve regel
    X0, X1 = int(max(0, x0 - pad)), int(min(view.shape[1], x1 + pad))
    Y0, Y1 = int(max(0, y0 - pad)), int(min(view.shape[0], y1 + pad))
    g = cv2.cvtColor(view[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    c = ((x0 + x1) / 2 - X0, (y0 + y1) / 2 - Y0)
    bx0, bx1 = int(x0 - X0), int(x1 - X0); by0, by1 = int(y0 - X0 * 0 - Y0 - .4 * hgt), int(y1 - Y0 + .4 * hgt)
    best = None
    for a in np.arange(around - 4, around + 4.01, 0.1):
        R = cv2.getRotationMatrix2D(c, a, 1.0)
        band = cv2.warpAffine(gy, R, (g.shape[1], g.shape[0]))[max(0, by0):by1, max(0, bx0):bx1]
        sc = float(np.var(band.sum(1)))
        if best is None or sc > best[0]:
            best = (sc, float(a))
    return best[1]


def label_lines(view, words):
    """Regels op het label; vindt Vision geen tracklist, dan nog eens op dubbele grootte
    lezen (kleine of omcirkelde cijfers) en de kaders terugschalen."""
    lines = ocr_lines(view)
    def track_count(rows):
        return sum(any(found(norm(w), norm(l['text'])) for w in words[2:]) for l in rows)
    if track_count(lines) >= 2:
        return lines
    big = cv2.resize(view, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    more = ocr_lines(big)
    for l in more:
        l['box'] = tuple(v / 2 for v in l['box']); l['height'] /= 2; l['width'] /= 2
    return more if (track_count(more), len(more)) > (track_count(lines), len(lines)) else lines


def parallel_label_text(view, lines):
    """Measure a consensus of separate central text rows, never a lone title.

    Each baseline's angular interval allows one pixel over the measured row
    width plus half the existing 0.1-degree search step. Competing consensuses
    remain uncertain. The outer circular lettering cannot supply these rows.
    """
    h, w = view.shape[:2]
    rows = []
    for line in sorted(lines, key=lambda l: -l['width']):
        x0, y0, x1, y1 = line['box']
        if (len(re.findall('[A-Za-z]', line['text'])) < 6
                or line['width'] <= 2.5*line['height']
                or math.hypot((x0+x1)/2-w/2, (y0+y1)/2-h/2) >= .38*min(w,h)):
            continue
        # Several OCR fragments on one baseline are still only one row.
        if any(min(y1, other['box'][3]) > max(y0, other['box'][1]) for other in rows):
            continue
        rows.append(line)
    if len(rows) < 3:
        return None
    offsets = np.asarray([line_angle(view, line['box'], 0.0) for line in rows])
    uncertainty = np.degrees(np.arctan(1/np.asarray([line['width'] for line in rows]))) + .05
    low, high = offsets-uncertainty, offsets+uncertainty
    endpoints = np.sort(np.r_[low, high])
    probes = (endpoints[:-1]+endpoints[1:])/2
    groups = set()
    for angle in probes:
        indices = np.flatnonzero((low <= angle) & (high >= angle))
        # Distinct fragments of one sloping row must remain one observation.
        # Project onto the normal of this baseline and undo the axis-aligned
        # bounding box's expansion to recover its actual text-height band.
        theta = math.radians(angle); bands=[]; independent=[]
        for i in indices:
            x0,y0,x1,y1=rows[i]['box']; cx,cy=(x0+x1)/2,(y0+y1)/2
            centre=cy*math.cos(theta)-cx*math.sin(theta)
            height=max(1, ((y1-y0)*math.cos(theta)-(x1-x0)*abs(math.sin(theta)))/math.cos(2*theta))
            band=(centre-height/2-.5, centre+height/2+.5)
            if any(min(band[1],other[1])>max(band[0],other[0]) for other in bands):
                continue
            bands.append(band);independent.append(int(i))
        groups.add(tuple(independent))
    count = max(map(len, groups), default=0)
    best = [group for group in groups if len(group) == count]
    if count < 3 or len(best) != 1:
        return None
    group = list(best[0])
    return float((low[group].max()+high[group].min())/2), count


def level_label(label, coarse, words):
    """Feit: de tracklist op een label staat recht. Draai het label naar de grove
    stand en zet de regels van de tracklist (bekend van Discogs) precies horizontaal.
    Is er geen tracklist te lezen, dan de grootste tekst (artiest of titel).
    Meten, bijdraaien en opnieuw meten tot de regels recht liggen (de grove stand
    kan tot ±7,5 graad naast zitten)."""
    n = label.shape[0]; c = (n / 2, n / 2); angle = coarse; how = 'none'
    for _ in range(4):
        R = cv2.getRotationMatrix2D(c, -angle, 1.0)
        view = cv2.warpAffine(label, R, (n, n), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        lines = [l for l in label_lines(view, words) if len(re.findall(r'\w', l['text'])) >= 3]
        titles = [l for l in lines if any(found(norm(w), norm(l['text'])) for w in words[:2])]
        tracks = [l for l in lines if any(found(norm(w), norm(l['text'])) for w in words[2:])]
        # Feit (André, 8 nov): de tracklist staat op een label (bijna) altijd recht.
        # Die is leidend; alleen als er geen tracklist te lezen is, de grootste tekst
        # (artiest of titel).
        # een tracktitel die ook de albumtitel is (bv. 'Set This Thing On Fire') telt niet als tracklistregel
        tracks = [l for l in tracks if l not in titles or len(tracks) == 1 and not titles]
        if not words:
            tracks = numbered_track_lines(lines, view.shape)
            if not tracks:
                # Sleeve Transparant: geen genummerde regels gelezen (Vision las bij een scherp label het nummer los, of als
                # andere letter: 'TOO LONG', 'З HAIL'); de speelduren (m:ss) van de tracklist staan ook recht onder elkaar
                hh_, ww_ = view.shape[:2]
                tracks = [l for l in lines if re.match(r'^\s*\d{1,2}[:.]\d{2}\s*$', l['text'])
                          and abs((l['box'][0] + l['box'][2]) / 2 - ww_ / 2) < .38 * ww_ and abs((l['box'][1] + l['box'][3]) / 2 - hh_ / 2) < .38 * hh_]
                if len(tracks) < 3:
                    tracks = []
        # Once track rows established the angle, preserve their measurement
        # if a later OCR pass misses them. This fallback must not replace it.
        parallel = (parallel_label_text(view, lines)
                    if not tracks and (how == 'none' or how.startswith('parallel label text')) else None)
        if tracks:
            off = float(np.median([line_angle(view, l['box'], 0.0) for l in tracks]))
            how = 'tracklist (%d lines)' % len(tracks)
        elif parallel:
            off, count = parallel
            how = 'parallel label text (%d rows)' % count
        elif titles and how == 'none':
            big = max(titles, key=lambda l: l['height'])
            off, how = line_angle(view, big['box'], 0.0), 'largest text: ' + big['text']
        else:
            break
        angle = (angle - off) % 360
        if abs(off) < 0.15:
            break
    if how.startswith('parallel label text') and abs(off) >= .15:
        how = 'none'       # A clipped search peak is not a converged baseline.
    return angle, how


RIM_MM = 44.0        # Sleeve Transparant: randtekst van een 12"-label loopt rond buiten deze straal (mm vanaf het midden)


def disc_angle(path, words, px_mm):
    """Plaat: alleen het label lezen (midden, volle resolutie), in stappen rondgedraaid.
    De hoek waarin de bekende tekst het best terug te lezen is, is rechtop.
    Geeft (hoek met de klok mee, score per geteste hoek, gelezen woorden).
    Sleeve Transparant, zonder bekende tekst: feit, de kleine randtekst loopt rond het label
    en leest daardoor in elke stand een stukje; de tracklist en titel staan recht en lezen
    maar in één stand. Is de winnaar niet duidelijk, dan nog eens zonder de rand lezen; dat
    telt alleen als het dan wel duidelijk is (anders blijft het TWIJFEL)."""
    first = _disc_angle(path, words, px_mm)
    if _clear(first, words):
        return first
    second = _disc_angle(path, words, px_mm, RIM_MM)
    if _clear(second, words):
        print('label: rim text left out, upright from the straight text', file=sys.stderr)
        return second[0], second[1], second[2] + ['rim text left out']
    if words:
        # Vision is not invariant under a half-turn. If neither view gives
        # reliable text, read an exact half-turn of the same label pixels.
        # Accept only multiple actual track rows; a title remains uncertain.
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h, w = image.shape[:2]
        r = min(int(LABEL_MM / 2 * px_mm), h // 2, w // 2)
        label = image[h//2-r:h//2+r, w//2-r:w//2+r]
        with tempfile.TemporaryDirectory() as folder:
            flipped = os.path.join(folder, 'label-half-turn.png')
            cv2.imwrite(flipped, cv2.rotate(label, cv2.ROTATE_180))
            third = _disc_angle(flipped, words, px_mm)
        tracks = any(re.search(r'level: track(?:list| direction) \(([2-9]|[1-9][0-9]+) lines\)', text)
                     for text in third[2])
        if tracks and _clear(third, words):
            return ((float(third[0]) + 180) % 360,
                    {(float(angle) + 180) % 360: score for angle, score in third[1].items()},
                    third[2] + ['track rows confirmed in a half-turn OCR view'])
    return first


def _clear(res, words):
    best, per_angle, got = res
    far = [v for k, v in per_angle.items() if abs(((k - best + 180) % 360) - 180) >= 30]
    top = max(per_angle.values(), default=0)
    enough = 1 if words else READABLE_MIN
    return (top >= enough and top >= CLEAR * max(max(far, default=0), 1)
            and 'title only' not in got and 'level: none' not in got)


def _disc_angle(path, words, px_mm, rim_mm=None):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    h, w = im.shape[:2]; r = min(int(LABEL_MM / 2 * px_mm), h // 2, w // 2)
    label = im[h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r].copy()
    mask = np.zeros(label.shape[:2], np.uint8); cv2.circle(mask, (r, r), min(r, int(rim_mm * px_mm)) if rim_mm else r, 255, -1)
    label[mask == 0] = 255
    small = cv2.resize(label, (1240, 1240), interpolation=cv2.INTER_AREA) if label.shape[0] > 1300 else label
    # Vision leest tekst in elke stand en geeft de leesrichting per regel: de tracklist
    # (of anders artiest/titel) zegt dan meteen hoe het label gedraaid ligt.
    lines = label_lines(small, words)
    tracks = [l for l in lines if any(found(norm(w), norm(l['text'])) for w in words[2:])]
    heads = [l for l in lines if any(found(norm(w), norm(l['text'])) for w in words[:2])]
    use = tracks if len(tracks) >= 2 else []          # alleen de tracklist geeft hier meteen de stand; anders rondgedraaid zoeken
    if not heads and words:
        # de titel over twee regels ('TO GET HER' / 'TOGETHER'): een regel van minstens 6 tekens die in artiest of titel staat
        heads = [l for l in lines if len(norm(l['text'])) >= 6 and any(norm(l['text']) in norm(w) for w in words[:2])]
    # Sleeve Transparant: artiest/titel-regels en hun leesrichting (Vision leest ze in elke stand, maar de leesrichting zegt
    # hoe ze liggen). Controle op het rondgedraaid zoeken hieronder (To Get Her Together 2022: alleen 'TO GET HER TOGETHER').
    head_dir = None
    sides_ = [l for l in lines if re.fullmatch(r'(side|kant)[abcd1-4]|(side|kant)(one|two|three|four)', norm(l['text']))]
    if sides_:
        # Sleeve Transparant: 'SIDE A' staat altijd recht; een titel kan schuin ontworpen zijn (To Get Her Together:
        # titel 18 graden schuin, 'SIDE A' recht). Dan 'SIDE A' als leesrichting.
        heads = sides_
    if not heads:
        # Sleeve Transparant: geen artiest/titel herkend: lange rechte regels (breed t.o.v. hoog: tracklist op één regel,
        # 'SIDE A', copyright) hebben samen één leesrichting; de randtekst loopt rond en spreidt alle kanten op
        # (Sad Singalong Songs 2023: '1. THE RULES 2. PRETENDING AS ALWAYS')
        longs = [l for l in lines if l['width'] >= 6 * max(l['height'], 1) and len(norm(l['text'])) >= 8]
        if len(longs) >= 2:
            dd = np.array([l['direction'] for l in longs])
            best_ = max(range(len(dd)), key=lambda i: int((np.abs(((dd - dd[i] + 180) % 360) - 180) < 8).sum()))
            group = [l for l in longs if abs(((l['direction'] - dd[best_] + 180) % 360) - 180) < 8]
            if len(group) >= 2 and len(group) >= .6 * len(longs):
                heads = group
    if heads and (words or len(heads) >= 2):
        hd = np.array([l['direction'] for l in heads]); ref_ = hd[0]
        if np.median(np.abs(((hd - ref_ + 180) % 360) - 180)) < 10:
            head_dir = (float(np.median(((hd - ref_ + 180) % 360) - 180)) + ref_) % 360
    if use:
        dirs = np.array([l['direction'] for l in use])
        ref = dirs[0]; spread = np.abs(((dirs - ref + 180) % 360) - 180)
        if np.median(spread) < 10:
            est = (float(np.median(((dirs - ref + 180) % 360) - 180)) + ref) % 360   # zoveel met de klok mee draaien
            fine, how = level_label(small, est, words)
            if not how.startswith('tracklist') and len(use) >= 3 and spread.max() <= 1.0:
                # Three or more actual track rows agree on their read direction.
                # A later OCR miss must not replace that fact with a logo/title.
                fine, how = est, 'track direction (%d lines)' % len(use)
            got = [w for w in words if any(found(norm(w), norm(l['text'])) for l in lines)]
            # Alleen artiest/titel gelezen, geen tracklist: een titel kan schuin ontworpen zijn
            # (Set This Thing On Fire, kant B). Dan niet vertrouwen maar twijfelen.
            note = [] if use is tracks and how.startswith(('tracklist', 'track direction')) else ['title only']
            if use is tracks and how.startswith('none'):
                # Sleeve Transparant: twee of meer tracklistregels met dezelfde leesrichting zijn het feit, ook als het
                # fijne rechtzetten niets vond (Sad Singalong Songs 2023: tracks op één lange regel): die richting
                fine, note = est, []
            return fine, {round(est, 1): len(use)}, got + ['direction from %d lines' % len(use), 'level: ' + how] + note
    tried = {}
    def score_at(base):
        n_ = small.shape[0]                                       # op ±300 dpi zoeken: snel, en Vision leest daar goed
        R = cv2.getRotationMatrix2D((n_ / 2, n_ / 2), -base, 1.0)    # met de klok mee draaien
        view = cv2.warpAffine(small, R, (n_, n_), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        # Geen bekende tekst (geen Discogs-link): dan telt hoeveel echte woorden Vision leest. Feit: tekst leest alleen rechtop.
        sc, hits = (known_text_score(read_image(view), words) if words else readable_text_score(read_image(view)))
        for q, v in sc.items():
            tried[(base + q) % 360] = (v, hits[q])
    for base in range(0, 90, 15):
        score_at(base)
    coarse = max(tried, key=lambda k: tried[k][0])
    for base in np.arange(coarse - 6, coarse + 6.1, 3):
        score_at(float(base) % 90)
    best = max(tried, key=lambda k: (tried[k][0], -abs(((k - coarse + 180) % 360) - 180)))
    if head_dir is not None and abs(((best - head_dir + 180) % 360) - 180) > 30:
        # het zoeken wijst de andere kant op dan de leesrichting van artiest/titel: de leesrichting is het feit
        fine, how = level_label(small, head_dir, words)
        if how.startswith('none'):
            fine = head_dir
        return fine, {round(head_dir, 1): len(heads)}, ['direction from the title (%d lines)' % len(heads), 'level: ' + how, 'title only']
    # Sleeve Transparant: controle na het rondzoeken: in de gevonden stand nog eens lezen. Lezen 'SIDE x' of artiest/titel
    # daar ondersteboven (leesrichting ±180), dan een halve slag erbij (To Get Her Together 2022 kant B: 'Music On Vinyl'
    # wees de verkeerde kant op; het label lag eerst 60 graden scheef en 'SIDE B' was toen niet te lezen)
    side_seen = False
    try:
        n0 = small.shape[0]
        def keys_at(angle_):
            v_ = cv2.warpAffine(small, cv2.getRotationMatrix2D((n0 / 2, n0 / 2), -angle_, 1.0), (n0, n0), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
            l_ = ocr_lines(v_)
            return sum(1 for l in l_ if re.fullmatch(r'(side|kant)[abcd1-4]|(side|kant)(one|two|three|four)', norm(l['text']))
                       or (words and len(norm(l['text'])) >= 6 and any(norm(l['text']) in norm(w) for w in words[:2])))
        here, flipped = keys_at(best), keys_at((best + 180) % 360)
        if flipped > here:
            best = (best + 180) % 360
        side_seen = max(here, flipped) > 0
    except Exception:
        pass
    fine, how = level_label(small, best, words)          # fijne hoek op ±300 dpi: Vision leest de regels daar het best
    print('label level:', how, f'({fine:.1f} degrees)', file=sys.stderr)
    if how.startswith('none') and side_seen:
        # stand bevestigd door 'SIDE x' of artiest/titel, alleen niet fijn uitgelijnd: deze stand tonen (met twijfel)
        return best, {round(k, 1): v[0] for k, v in sorted(tried.items())}, tried[best][1] + ['level: ' + how, 'title only']
    return fine, {round(k, 1): v[0] for k, v in sorted(tried.items())}, tried[best][1] + ['level: ' + how]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image'); p.add_argument('--out')
    p.add_argument('--pixels', help='source pixels to rotate after measuring the label on image')
    p.add_argument('--words', required=True, help='JSON list with the known text of the release (Discogs)')
    p.add_argument('--disc', action='store_true', help='record side: read the label, any angle')
    p.add_argument('--dpi', type=float, default=300.0)
    p.add_argument('--lanczos', action='store_true', help='Sleeve Transparant: plaat draaien met Lanczos (scherper dan kubisch)')
    a = p.parse_args()
    words = json.load(open(a.words))
    if a.disc:
        best, per_angle, got = disc_angle(a.image, words, a.dpi / 25.4)
        top = sorted(per_angle.values(), reverse=True)
        # tweede kandidaat: de beste score minstens 30 graden verderop
        far = [v for k, v in per_angle.items() if abs(((k - best + 180) % 360) - 180) >= 30]
        enough = 1 if words else READABLE_MIN                  # zonder bekende tekst moet er genoeg echte tekst gelezen zijn
        verdict = 'GOED' if top[0] >= enough and top[0] >= CLEAR * max(max(far, default=0), 1) else 'TWIJFEL'
        if 'title only' in got or 'level: none' in got:
            verdict = 'TWIJFEL'                        # label alleen op de titel rechtgezet: jij kijkt
        result = {'rotate_clockwise': round(best, 1), 'known_text_best': top[0], 'next_best_elsewhere': max(far, default=0), 'read_upright': got, 'verdict': verdict}
        print(json.dumps(result, ensure_ascii=False))
        if a.out:
            im = cv2.imread(a.pixels or a.image, cv2.IMREAD_UNCHANGED); hh, ww = im.shape[:2]
            R = cv2.getRotationMatrix2D((ww / 2, hh / 2), -best, 1.0)
            cv2.imwrite(a.out, cv2.warpAffine(im, R, (ww, hh), flags=cv2.INTER_LANCZOS4 if a.lanczos else cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0)))
        return
    # Hoes: Vision leest tekst in elke stand en geeft de leesrichting per regel. Bekende
    # tekst van de release (feit) in een duidelijke richting = zo moet de hoes draaien
    # (een hoes staat altijd op een kwartslag).
    im0 = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)[:, :, :3]
    k0 = 1600 / max(im0.shape[:2])
    view0 = cv2.resize(im0, None, fx=k0, fy=k0, interpolation=cv2.INTER_AREA) if k0 < 1 else im0
    known = [l for l in ocr_lines(view0) if any(found(norm(w), norm(l['text'])) for w in words)]
    if not known:
        known = faint_known_lines(im0, words)
    votes = {}
    for l in known:
        q = int(round(l['direction'] / 90.0)) % 4 * 90               # leesrichting op een kwartslag
        votes[q] = votes.get(q, 0) + len(norm(l['text']))
    if votes:
        ranked = sorted(votes, key=votes.get, reverse=True)
        best = ranked[0]; second = votes[ranked[1]] if len(ranked) > 1 else 0
        verdict = 'GOED' if votes[best] >= CLEAR * max(second, 1) else 'TWIJFEL'
        result = {'rotate_clockwise': best, 'known_text_per_rotation': votes, 'read_upright': [l['text'] for l in known][:12] + ['direction from Vision'], 'verdict': verdict}
    else:
        rot = read_rotations(a.image)
        score, hits = known_text_score(rot, words)
        ranked = sorted(score, key=score.get, reverse=True)
        best, second = ranked[0], ranked[1]
        verdict = 'GOED' if score[best] > 0 and score[best] >= CLEAR * max(score[second], 1) else 'TWIJFEL'
        result = {'rotate_clockwise': best, 'known_text_per_rotation': score, 'read_upright': hits[best], 'verdict': verdict}
        if score[best] == 0:
            # geen tekst van de release: dan de stand waarin Vision andere tekst duidelijk leest
            rs, rh = readable_text_score(rot)
            ranked = sorted(rs, key=rs.get, reverse=True)
            best, second = ranked[0], ranked[1]
            if rs[best] >= 40 and rs[best] >= CLEAR * max(rs[second], 1):
                result = {'rotate_clockwise': best, 'readable_text_per_rotation': rs,
                          'read_upright': rh[best][:8] + ['readable text (not of the release)'], 'verdict': 'LEESBAAR'}
            elif not words:
                sparse = sparse_text_orientation(a.image, rot)
                if sparse:
                    result = sparse
                    best = sparse['rotate_clockwise']
    print(json.dumps(result, ensure_ascii=False))
    if a.out:
        im = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
        if ROTATE[best] is not None:
            im = cv2.rotate(im, ROTATE[best])
        cv2.imwrite(a.out, im)


if __name__ == '__main__':
    main()
