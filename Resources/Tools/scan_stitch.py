#!/usr/bin/env python3
"""Meerdere flatbed-deelscans van één hoes of plaat samenvoegen tot één beeld.

Gemaakt voor lp-hoezen en platen die niet in één keer op de scanner passen.
Anders dan een gewone panorama-merge:
  - alleen verschuiving + draaiing (een flatbed vervormt niet), dus geen
    "uitgerekte" of kromme stukken
  - het witte scannerdeksel rond de hoes telt niet mee en wordt nooit over de
    hoes heen geplakt
  - kleur/grijs gelijktrekken: elk stuk krijgt per kleurkanaal een zacht
    verlopende correctie (vaste zweem + lichtverloop van de lamp), gemeten in
    de overlap; ook voor de lastige grijstinten
  - naden zacht laten overlopen

Gebruik:
  python3 scan_stitch.py uit.png deel1.jpg deel2.jpg deel3.jpg ...
      [--geen-kleur]   kleuren niet gelijktrekken
      [--vorm hoes|plaat] standaard bepaalt hij dat zelf
      [--referentie N] stuk N (1 = eerste) is de kleurreferentie (standaard:
                       het stuk met de meeste overlap)
  De oriëntatie van het resultaat is die van het eerste deel.

Uitvoer: png met transparantie waar geen scan was. Daarna eventueel
scan_cutout.py (cover/vinyl) en de watermerkstap.
"""
import argparse, os, sys, gc, contextlib, math
from scan_progress import progress
from scan_blend import detail_blur
from scan_fields import GainPlane, CorrectedImage, FieldGain
from scan_memory import MappedScanStore,MappedArrayStore,render_tile_size,drop_array_pages,flush_array_rows,MemoryLease
import numpy as np
import cv2

MAX_SIDE = 3600        # rekenschaal: langste zijde voor uitlijnen/kleur/vorm (uitvoer is volle resolutie)
TILE = 2048           # blokgrootte voor het opbouwen op volle resolutie (geheugen)
MATCH_SIZE = 3000      # langste zijde voor het zoeken naar overeenkomsten (hoog = nauwkeurig)
DETAIL_SIGMA = 6       # px: fijner dan dit = detail (uit één scan), grover = kleur/licht (overvloeien)
FEATHER = 700          # px: zo ver van de scanrand weegt een pixel pas volledig mee
                       # (randen van de glasplaat zijn donkerder: lamp + schaduw)
EDGE = 150             # px van de rand die niet meetelt bij het kleurmeten
QUALITY_WINDOW = 48    # px (rekenschaal): vakjes waarin de eindcontrole naden en scherpte meet
QUALITY_MIN_SPOTS = 3  # pas vanaf zoveel plekken een melding (één plek is vaak structuur, geen tekst)
QUALITY_RESPONSE = 0.3 # minimale betrouwbaarheid van de verschuivingsmeting
QUALITY_SHIFT = 4.0   # px (uitvoer): zoveel verschuiving op een naad door tekst wordt gemeld
SEAM_CORRIDOR = 40    # px (rekenschaal): zo ver mag een naad van de scherpste-scan-grens afwijken
REFINE_MAX_ERROR = 4.0 # px (rekenschaal): paren die volgens de boom verder afwijken, tellen niet mee bij het bijstellen
INK_EDGE = 0.35       # deel van de sterkste randen dat als tekst/drukwerk telt (hoeskader)
INK_OUTSIDE = 0.03    # zoveel drukwerk mag buiten het hoeskader vallen; meer = kader te klein
INK_MARGIN = 0.01     # ruimer kader begint minstens zo ver (deel van de zijde) buiten de tekst
INK_WINDOW = 15       # px (op 0,2): vak waarin drukwerk randen in beide richtingen heeft
INK_BOTH = 0.15       # zoveel van dat vak met randen per richting
HOLE_RADIUS = 0.022   # asgat (straal) als deel van de plaatstraal; iets kleiner dan 0,026 voor de marge
SPREAD_ASPECT = 1.7   # samengevoegd vlak minstens zo langwerpig: uitgeklapte binnenkant (2:1)
LP_MM = 304.8         # bekende maat 12": vangnet als de papierrand nergens te zien is
INK_MIN_AREA = 0.002  # minder drukwerk dan dit deel van het canvas: geen controle (alleen lijnen)
REFINE_SLACK = 0.5     # px: bijstellen mag geen enkel paar meer dan dit verslechteren
SEAM_SIDE = 1200       # px: rekenschaal voor het kiezen van de naden
WINNER_SMOOTH = 8     # px (rekenschaal): keuzekaart per deelscan samenhangend maken
SHARP_SIGMA_MM = 0.7   # scherpte meten over ±0,7 mm
DETAIL_RATIO = 1.8     # zoveel scherper moet een scan zijn om bij kleine tekst te winnen
TONE_SIGMA = 30        # px (rekenschaal): alleen deze heel grove toon wordt tussen deelscans gemengd;
                       # alles fijner (ook letters) komt uit de gekozen deelscan
MAX_SLOPE = 0.06      # lichtverloop per deelscan: max ~6% over de hele breedte/hoogte
MAX_CAST = 0.04        # vaste kleurzweem per kanaal: max ~4%


def open_srgb(path, factor=1.0):
    """Open with the established resize/profile path, avoiding RGB copies."""
    import io
    from PIL import Image,ImageCms
    Image.MAX_IMAGE_PIXELS=None
    image=Image.open(path)
    profile=image.info.get('icc_profile')
    if image.mode not in ('RGB','RGBA','L'):image=image.convert('RGB')
    if factor != 1:
        image=image.resize((max(1,round(image.width*factor)),max(1,round(image.height*factor))),Image.LANCZOS)
    if image.mode != 'RGB':image=image.convert('RGB')
    if profile:
        source=ImageCms.ImageCmsProfile(io.BytesIO(profile))
        if 'srgb' not in ImageCms.getProfileDescription(source).lower():
            ImageCms.profileToProfile(image,source,ImageCms.createProfile('sRGB'),outputMode='RGB',inPlace=True)
    return image


LICHTKOLOM = None   # alleen met --lichtkolom (transparant vinyl): lichtcorrectie per scannerkolom, zie scan_lichtkolom.py


def load_into(path,target):
    """Decode/profile once; copy RGB to mapped BGR in bounded row bands."""
    with open_srgb(path) as image:
        if target.shape != (image.height,image.width,3):raise ValueError('Decoder target dimensions changed')
        for start in range(0,image.height,256):
            stop=min(image.height,start+256)
            with image.crop((0,start,image.width,stop)) as band:
                target[start:stop]=cv2.cvtColor(np.asarray(band),cv2.COLOR_RGB2BGR)
            if LICHTKOLOM is not None:
                import scan_lichtkolom
                scan_lichtkolom.apply(target,LICHTKOLOM,rows=(start,stop,image.height))
            flush_array_rows(target,start,stop)


def load(path,f=1.0,consume_cache=False):
    """Read BGR in sRGB, optionally resized; no source file is modified."""
    try:
        if LICHTKOLOM is not None:
            import scan_lichtkolom
            with open_srgb(path,f) as image:
                bgr=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            if not scan_lichtkolom.apply(bgr,LICHTKOLOM):
                print(f'lichtkolom: {os.path.basename(path)} hoort niet bij dit glas; niet gecorrigeerd',flush=True)
            return bgr
        from scan_previews import get
        cached=get(path,f,consume=consume_cache)
        if cached is not None:return cached
        with open_srgb(path,f) as image:
            return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    except Exception as error:
        sys.exit(f'can’t read {path}: {error}')


def load_previews(path, factors, inspector=None):
    """Decode once, then create independent previews with the existing pixel path."""
    import io
    from PIL import Image, ImageCms
    Image.MAX_IMAGE_PIXELS = None
    outputs=[]
    with Image.open(path) as original:
        icc=original.info.get('icc_profile')
        image=original if original.mode in ('RGB','RGBA','L') else original.convert('RGB')
        profile=ImageCms.ImageCmsProfile(io.BytesIO(icc)) if icc else None
        convert=profile is not None and 'srgb' not in ImageCms.getProfileDescription(profile).lower()
        for factor in factors:
            preview=image
            if factor != 1:
                preview=image.resize((max(1,round(image.width*factor)),max(1,round(image.height*factor))),Image.LANCZOS)
            preview=preview.convert('RGB')
            if convert:
                preview=ImageCms.profileToProfile(preview,profile,ImageCms.createProfile('sRGB'),outputMode='RGB')
            outputs.append(cv2.cvtColor(np.asarray(preview),cv2.COLOR_RGB2BGR))
        if inspector is not None:return outputs,inspector(outputs,original)
    return outputs


def lid_colour(edge):
    """Kleur van het deksel uit de randpixels + hoeveel donkerder het mag zijn.

    De rand is vaak grotendeels hoes, dus niet de mediaan: groepeer de
    randkleuren en neem de meest neutrale groep (deksel is wit/grijs); is een
    grijze hoes even neutraal, dan de lichtste. Ligt er zo'n even neutrale
    grijze hoes naast, dan mag 'deksel' maar half zo donker worden als het
    verschil, anders loopt het de hoes in."""
    px = edge.reshape(-1, 3).astype(np.float32)
    groups = []
    for k in (2, 3):
        _, lab_, cen = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3, cv2.KMEANS_PP_CENTERS)
        for c in range(k):
            if (lab_.ravel() == c).mean() >= 0.08:
                L, a, b = cen[c]
                groups.append((int(np.hypot(a - 128, b - 128) / 5), -L, np.median(px[lab_.ravel() == c], axis=0)))
    if not groups:
        return np.median(px, axis=0), 45.0
    groups.sort(key=lambda g: (g[0], g[1]))
    ch, negL, lid = groups[0]
    darker = [-g[1] for g in groups if g[0] <= ch and -g[1] < -negL - 8]
    down = 45.0 if not darker else float(np.clip(0.5 * (-negL - max(darker)), 8, 45))
    return lid, down


def background_mask(im):
    """True = scannerachtergrond: vlakken in de kleur van het deksel, egaal, die
    aan de rand vastzitten. Kleur (Lab) en niet helderheid, want een lichte hoes
    is even licht als het deksel; randen (schaduwlijn langs de hoes) houden het
    tegen zodat het niet de hoes in loopt."""
    small = cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = np.concatenate([lab[2], lab[-3], lab[:, 2], lab[:, -3]])
    lid, down = lid_colour(edge)                           # kleur van het deksel
    dC = np.sqrt(((lab[..., 1:] - lid[1:]) ** 2).sum(axis=2))    # kleurzweem
    dL = np.where(lab[..., 0] < lid[0], (lid[0] - lab[..., 0]) * 45.0 / down,   # donkerder: beperkt
                  lab[..., 0] - lid[0])                                     # lichter mag
    gf = g.astype(np.float32)
    var = cv2.blur(gf * gf, (9, 9)) - cv2.blur(gf, (9, 9)) ** 2
    barrier = cv2.dilate(cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 15, 40), np.ones((3, 3), np.uint8)) > 0
    cand = ((dC < 6) & (dL < 45) & (var < 80) & ~barrier).astype(np.uint8)
    n, lab_ = cv2.connectedComponents(cand)
    border = set(np.unique(np.concatenate([lab_[0], lab_[-1], lab_[:, 0], lab_[:, -1]]))) - {0}
    bg = np.isin(lab_, list(border))
    bg = cv2.dilate(bg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    # alleen grote vlakken zijn hoes/plaat (stofjes op het deksel niet)
    n, lv, st, _ = cv2.connectedComponentsWithStats((~bg).astype(np.uint8))
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] > 0.03 * bg.size
    bg = ~keep[lv]
    bg = cv2.resize(bg.astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    m = 6                                                # scannerrand zelf ook weg
    bg[:m] = bg[-m:] = True; bg[:, :m] = bg[:, -m:] = True
    return bg


def features(im, valid):
    s = MATCH_SIZE / max(im.shape[:2])
    small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    vm = cv2.resize(valid.astype(np.uint8) * 255, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create(nfeatures=12000)
    kp, des = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), vm)
    pts = np.float32([k.pt for k in kp]) / s
    return pts, des


def faint_print(im):
    """Feit: ook flauw drukwerk is drukwerk (lichte letters op wit papier). Het plaatselijke contrast
    oprekken maakt het zichtbaar voor het zoeken naar overeenkomsten; het beeld zelf verandert niet."""
    L = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2LAB)[:, :, 0]
    return cv2.cvtColor(cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8)).apply(L), cv2.COLOR_GRAY2BGR)


def pair_matches(fa, fb):
    """Overeenkomende punten (src in b, dst in a) of None."""
    pa, da = fa; pb, db = fb
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.72 * x[1].distance]
    if len(good) < 15:
        return None
    return pb[[g.queryIdx for g in good]], pa[[g.trainIdx for g in good]]


def rigid_fit(src, dst):
    """Rigide transformatie (draaiing + verschuiving) die src op dst legt: (M, inliers, rms)."""
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=5000, confidence=0.999)
    if M is None:
        return None, 0, 1e9
    sc = np.hypot(M[0, 0], M[1, 0])
    if not 0.97 < sc < 1.03:          # flatbed: schaal hoort 1 te zijn
        return None, 0, 1e9
    # Uniform scaling supports scans exported at slightly different sizes.
    # No shear or perspective: a circle remains a circle.
    k = inl.ravel() > 0
    M[:,2] = np.mean(dst[k] - src[k] @ M[:,:2].T, axis=0)
    r = np.sqrt((((src[k] @ M[:, :2].T) + M[:, 2] - dst[k]) ** 2).sum(1).mean())
    return np.vstack([M, [0, 0, 1]]), int(k.sum()), r


STRETCH_GAIN = 0.8    # rekcorrectie alleen bij minstens 20% betere passing

def scanner_stretch(pairs):
    """Een flatbed rekt de ene richting net iets anders dan de andere (~0,1-0,5%).
    Liggen stukken 90 graden gedraaid, dan past alleen schuiven/draaien niet meer
    precies (letters verspringen). Zoek de verhouding y/x die alle paren het
    best laat passen; dezelfde scanner, dus één waarde voor alle scans."""
    # (12 nov geprobeerd: alle punten met afgekapte afwijking. Dat zette de rek ook aan waar het ruis was en
    # maakte TA, Fake, HNY 1-2 mm kleiner in de vaste toets; teruggezet, 13 nov.)
    def score(sy):
        D = np.array([1.0, sy], np.float32); tot = []
        for src, dst in pairs.values():
            M, k, r = rigid_fit(src * D, dst * D)
            if M is not None and k >= 15:
                tot.append(r)
        return np.mean(tot) if tot else 1e9
    grid = np.arange(0.994, 1.0061, 0.001)
    best = min(grid, key=score)
    fine = np.arange(best - 0.001, best + 0.00101, 0.00025)
    best = min(fine, key=score)
    return float(best), score(1.0), score(best)


def pair_errors(H, links):
    """Gemiddelde passingsfout (px) per paar onder de plaatsing H."""
    out = {}
    for (i, j), (src, dst) in links.items():
        a = np.c_[dst, np.ones(len(dst))] @ H[i][:2].T
        b = np.c_[src, np.ones(len(src))] @ H[j][:2].T
        out[(i, j)] = float(np.hypot(*(a - b).T).mean())
    return out


def refine_placement(H, ref, pairs, D):
    """De boom plaatst elk deel via één ander deel; bij vier kwartslag-delen
    stapelen kleine fouten zich op tot een naad waar letters ~6 px verspringen
    (Deena & Jim tekstvel: buren 0,6-1,2 px, het niet-gekoppelde paar 2,1 px).
    Daarom alle overlappende paren tegelijk: per deel draaiing, schaal en
    verschuiving (lineair, kleinste kwadraten), het referentiedeel staat vast.
    Alleen overnemen als geen enkel paar slechter wordt dan het slechtste was."""
    keys = sorted(k for k in H if k != ref)
    if not keys:
        return H
    links = {}
    for (i, j), (src, dst) in pairs.items():
        if i < j and i in H and j in H:
            q_src, q_dst = src * D, dst * D
            M, inl = cv2.estimateAffinePartial2D(q_src, q_dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=5000, confidence=0.999)
            if M is not None and inl.sum() >= 15:
                k = inl.ravel() > 0
                links[(i, j)] = (q_src[k], q_dst[k])
    # Een paar dat volgens de boom ver afwijkt, is eerder een valse koppeling
    # (herhaald patroon) dan een echte overlap: niet laten meetrekken.
    first = pair_errors(H, links)
    links = {key: value for key, value in links.items() if first[key] <= REFINE_MAX_ERROR}
    if len(links) < len(keys) + 1:
        return H                       # geen lus: de boom is al de enige oplossing
    col = {k: 4 * n for n, k in enumerate(keys)}
    rows, rhs = [], []
    def terms(k, p, sign, row_x, row_y, fixed):
        if k == ref:
            fixed[0] += sign * (H[ref][0, :2] @ p + H[ref][0, 2]); fixed[1] += sign * (H[ref][1, :2] @ p + H[ref][1, 2])
            return
        c = col[k]; x, y = p
        row_x[c:c + 4] += sign * np.array([x, -y, 1, 0]); row_y[c:c + 4] += sign * np.array([y, x, 0, 1])
    for (i, j), (src, dst) in links.items():
        for p_dst, p_src in zip(dst, src):
            row_x = np.zeros(4 * len(keys)); row_y = np.zeros(4 * len(keys)); fixed = [0.0, 0.0]
            terms(i, p_dst, 1, row_x, row_y, fixed); terms(j, p_src, -1, row_x, row_y, fixed)
            rows += [row_x, row_y]; rhs += [-fixed[0], -fixed[1]]
    sol = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)[0]
    refined = dict(H)
    for k in keys:
        a, b, tx, ty = sol[col[k]:col[k] + 4]
        if not 0.97 < np.hypot(a, b) < 1.03:
            return H
        refined[k] = np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])
    before, after = pair_errors(H, links), pair_errors(refined, links)
    if max(after.values()) > max(before.values()) or any(after[key] > before[key] + REFINE_SLACK for key in links):
        return H
    if max(before.values()) - max(after.values()) >= 0.25:
        print(f'placement adjusted over all overlaps: worst pair {max(before.values()):.2f} -> {max(after.values()):.2f} px', flush=True)
    return refined


def place(n, feats):
    """Paarsgewijs uitlijnen en via een opspannende boom (meeste inliers) plaatsen."""
    pairs = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                pm = pair_matches(feats[i], feats[j])  # j op i
                if pm is not None:
                    pairs[(i, j)] = pm
    sy, r0, r1 = scanner_stretch(pairs)
    # Feit, geen schatting: alleen corrigeren als de scans er aantoonbaar beter door passen.
    # Zonder winst is de gevonden rek ruis (Fire-back: +0,40% en -0,23% op dezelfde scanner)
    # en zou hij de hoes 1-2 mm hoger of lager maken.
    if not (r1 <= STRETCH_GAIN * r0 and r0 - r1 >= 0.15):
        sy, r1 = 1.0, r0
    D = np.array([1.0, sy], np.float32)
    T = {}; W = np.zeros((n, n), int)
    for (i, j), (src, dst) in pairs.items():
        M, k, _ = rigid_fit(src * D, dst * D)
        if M is not None:
            T[(i, j)] = M; W[i, j] = k
    ref = int(np.argmax(W.sum(axis=1)))
    H = {ref: np.eye(3)}; frontier = [ref]
    while frontier:
        best = None
        for i in H:
            for j in range(n):
                if j not in H and W[i, j] > 0 and (best is None or W[i, j] > best[2]):
                    best = (i, j, W[i, j])
        if best is None:
            break
        i, j, _ = best
        H[j] = H[i] @ T[(i, j)]; frontier.append(j)
    H = refine_placement(H, ref, pairs, D)
    Ds = np.diag([1.0, sy, 1.0])                       # eerst de scannerrek eruit
    H = {k: v @ Ds for k, v in H.items()}
    if abs(sy - 1) >= 0.0005:
        print(f'scanner stretch corrected: y/x = {sy:.4f} (fit {r0:.2f} -> {r1:.2f} px)')
    missing = [k for k in range(n) if k not in H]
    return H, ref, W, missing


def thin_line_contrast(gray, kernel):
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    light = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return dark, light


def line_crossings(gray, width, length, horizontal, tolerance):
    """Return dark and light line strength across the expected line position."""
    middle = gray.shape[0] // 2
    cross = slice(middle - tolerance, middle + tolerance + 1)
    kernel = np.ones((width, width), np.uint8)
    direction = np.ones((1, length) if horizontal else (length, 1), np.uint8)
    result = []
    for contrast in thin_line_contrast(gray, kernel):
        line = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, direction)
        result.append(line[cross, middle] if horizontal else line[middle, cross])
    return np.asarray(result)


def broad_line_agreement(grays, masks, first, second, x, y, radius, horizontal,
                         width, length, tolerance):
    bounds = np.s_[y-radius:y+radius+1, x-radius:x+radius+1]
    if (x < radius or y < radius or y + radius >= grays[0].shape[0]
            or x + radius >= grays[0].shape[1]
            or not masks[0][bounds].all() or not masks[1][bounds].all()):
        return (-1, -1)
    strength = np.minimum(
        line_crossings(grays[first][bounds], width, length, horizontal, tolerance),
        line_crossings(grays[second][bounds], width, length, horizontal, tolerance))
    return tuple(int(row.argmax()) if row.max() >= 13 else -1 for row in strength)


def flat_patch(gray):
    low, high = np.percentile(gray, (5, 95))
    return high - low <= 5 and float(gray.max()) - float(gray.min()) <= 10


def corroborated_flat(grays, masks, ks, source, x, y, radius):
    """Require two flat observations and no second observation of the detail."""
    left, top = x - radius, y - radius
    right, bottom = x + radius + 1, y + radius + 1
    if left < 0 or top < 0 or bottom > grays[0].shape[0] or right > grays[0].shape[1]:
        return False
    area = np.s_[top:bottom, left:right]
    count = 0
    for i, k in enumerate(ks):
        if i == source or not masks[k][area].all():
            continue
        if not flat_patch(grays[i][area]):
            return False
        count += 1
    return count >= 2


def seam_line_conflicts(grays, masks, ks, best, support, px_mm, locations=None, evidence=None):
    """Find a thin, continuous dark line cut by a source change inside a cover."""
    thickness = max(3, round(.45 * px_mm) | 1)
    length = max(7, round(2.4 * px_mm) | 1)
    interior = support >= .99
    pairs = []
    horizontal = best[:, 1:] != best[:, :-1]
    vertical = best[1:, :] != best[:-1, :]
    for a, k in enumerate(ks):
        for b in range(a + 1, len(ks)):
            other = ks[b]
            across = ((best[:, :-1] == a) & (best[:, 1:] == b)) | ((best[:, :-1] == b) & (best[:, 1:] == a))
            down = ((best[:-1, :] == a) & (best[1:, :] == b)) | ((best[:-1, :] == b) & (best[1:, :] == a))
            seam = np.zeros(best.shape, bool)
            seam[:, 1:] = horizontal & across
            seam[1:, :] |= vertical & down
            seam &= interior & masks[k] & masks[other]
            if not seam.any():
                continue
            yy, xx = np.nonzero(seam)
            pad = length + thickness
            box = np.s_[max(0, yy.min() - pad):yy.max() + pad + 1, max(0, xx.min() - pad):xx.max() + pad + 1]
            local, axes = [], []
            for i in (a, b):
                g = grays[i][box]
                contrast = thin_line_contrast(g, np.ones((thickness, thickness), np.uint8))
                hline, vline = (np.maximum(*(cv2.morphologyEx(thin, cv2.MORPH_OPEN, kernel)
                                             for thin in contrast))
                                for kernel in (np.ones((1, length), np.uint8),
                                               np.ones((length, 1), np.uint8)))
                local.append(np.maximum(hline, vline))
                axes.append(hline >= vline)
            # A real line can land a few pixels away in an overlapping scan.
            # If it is visible nearby, this is a placement check, not a vanishing line.
            tolerance = max(1, round(.32 * px_mm))
            nearby = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
            left, right = local
            only_left = (left >= 13) & (cv2.dilate(right, nearby) <= 5)
            only_right = (right >= 13) & (cv2.dilate(left, nearby) <= 5)
            found = seam[box] & (only_left | only_right)
            if found.sum() >= 2:
                pairs.append((int(k), int(other), int(found.sum())))
                if locations is not None:
                    y, x = np.nonzero(found)
                    horizontal_line = np.where(only_left[y, x], axes[0][y, x], axes[1][y, x])
                    reporting_source = np.where(only_left[y, x], a, b)
                    x, y = x + box[1].start, y + box[0].start
                    radius = length // 2 + thickness + tolerance
                    flat = [corroborated_flat(grays, masks, ks, src, col, row, radius)
                            for src, col, row in zip(reporting_source, x, y)]
                    broad_width = max(thickness, round(.75 * px_mm) | 1)
                    broad = [broad_line_agreement(grays, (masks[k], masks[other]), a, b,
                                                 col, row, radius, direction, broad_width,
                                                 length, tolerance)
                             for col, row, direction in zip(x, y, horizontal_line)]
                    locations[k, other] = np.column_stack((x, y, horizontal_line, flat, broad))
                    if evidence is not None:
                        # Sleeve Transparant: ook het stukje uit de andere scan bewaren. De lijn staat maar in één
                        # scan (anders belicht of wazig, TA yellow 0004); volgt het resultaat de andere scan, dan klopt het.
                        other_source = np.where(only_left[y - box[0].start, x - box[1].start], b, a)
                        for store, chosen in (((k, other), reporting_source), (('other', k, other), other_source)):
                            patches = []
                            for source, col, row in zip(chosen, x, y):
                                bounds = np.s_[row-radius:row+radius+1, col-radius:col+radius+1]
                                if (row >= radius and col >= radius
                                        and row+radius < best.shape[0] and col+radius < best.shape[1]):
                                    patches.append(grays[source][bounds].copy())
                                else:
                                    patches.append(None)
                            evidence[store] = patches
    return pairs


def faithful_source_patch(rendered, source, tolerance):
    """Check the actual printed structure instead of requiring a straight line."""
    rendered = cv2.GaussianBlur(rendered.astype(np.float32), (0, 0), 1.1)
    source = cv2.GaussianBlur(source.astype(np.float32), (0, 0), 1.1)
    margin = max(1, tolerance)
    if min(source.shape) < 2*margin+9 or rendered.shape != source.shape:
        return False
    template = source[margin:-margin, margin:-margin]
    if float(template.std()) < 5:
        return False
    similarity = cv2.matchTemplate(rendered, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, offset = cv2.minMaxLoc(similarity)
    if score < .97:
        return False
    x, y = offset
    sample = rendered[y:y+template.shape[0], x:x+template.shape[1]]
    def agrees(observed):
        residual = observed-template
        residual -= np.median(residual)
        contrast = float(observed.std())/max(float(template.std()), 1)
        correlation = float(np.mean((observed-observed.mean())*(template-template.mean())))
        correlation /= max(float(observed.std()*template.std()), 1e-6)
        # Sleeve Transparant: de toegestane afwijking schaalt met het contrast van het drukwerk (TA yellow, met
        # lichtcorrectie: overeenkomst .98, contrast .97-1.03, maar 6-8 niveaus korrel); minimaal de oude 4 niveaus.
        return bool(correlation >= .97 and .9 <= contrast <= 1.1
                    and np.quantile(np.abs(residual), .9) <= max(4, .3 * float(template.std())))
    if agrees(sample):
        return True
    # Integer template matching leaves up to half a pixel of raster phase.
    # Refine translation only; no stretching, contrast loss or weaker limits.
    try:
        _, correction = cv2.findTransformECC(template, sample, np.eye(2, 3, dtype=np.float32),
                                              cv2.MOTION_TRANSLATION,
                                              (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-6),
                                              None, 3)
    except cv2.error:
        return False
    dx, dy = correction[:, 2]
    if max(abs(dx), abs(dy)) > .75:
        return False
    if x+dx < 0 or y+dy < 0 or x+dx+template.shape[1] > rendered.shape[1] or y+dy+template.shape[0] > rendered.shape[0]:
        return False
    yy, xx = np.indices(template.shape, dtype=np.float32)
    refined = cv2.remap(rendered, xx+x+dx, yy+y+dy, cv2.INTER_LINEAR)
    return agrees(refined)


def unresolved_seam_points(output, locations, transform, px_mm, diagnostics=None, evidence=None):
    """Check line continuity and corroborated flat paper in the rendered pixels."""
    thickness = max(3, round(.45 * px_mm) | 1)
    length = max(7, round(2.4 * px_mm) | 1)
    tolerance = max(1, round(.32 * px_mm))
    radius = length // 2 + thickness + tolerance
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    broad_width = max(thickness, round(.75 * px_mm) | 1)
    unresolved = 0
    for point_index, (x, y, horizontal, flat, dark_position, light_position) in enumerate(locations):
        sx = transform[0, 0] * (xx + x) + transform[0, 1] * (yy + y) + transform[0, 2]
        sy = transform[1, 0] * (xx + x) + transform[1, 1] * (yy + y) + transform[1, 2]
        failed = sx.min() < 0 or sy.min() < 0 or sx.max() >= output.shape[1] - 1 or sy.max() >= output.shape[0] - 1
        if failed:
            unresolved += 1
            if diagnostics is not None:
                diagnostics.append((x, y, 'outside merged image'))
            continue
        patch = sample_analysis_patch(output, sx, sy, transform)
        if patch.shape[2] == 4 and patch[:, :, 3].min() < 250:
            unresolved += 1
            if diagnostics is not None:
                diagnostics.append((x, y, 'near transparent pixels'))
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGRA2GRAY)
        strength = line_crossings(gray, thickness, length, horizontal, tolerance)
        if strength.max() >= 13 or (flat and flat_patch(gray)):
            continue
        # Sleeve Transparant: rasterdruk. Feit: met het raster weggepoetst (vervaging over 0,35 mm, groter dan een rasterpunt)
        # is er op deze plek niets te zien: geen lijn, geen rand, alleen vlak drukwerk. Dan kan een naad daar niets breken.
        # Alleen als ook de bronscans daar niets tonen: een vlakke plek in het resultaat kan ook een verloren lijn zijn.
        def plain(img_):
            sm_ = cv2.GaussianBlur(np.asarray(img_, np.float32), (0, 0), max(1., .35 * px_mm))
            c_ = sm_[thickness:-thickness, thickness:-thickness] if min(sm_.shape) > 2 * thickness + 4 else sm_
            return float(np.percentile(c_, 95) - np.percentile(c_, 5)) <= 10
        if evidence is not None and point_index < len(evidence):
            src_ = evidence[point_index]
            src_ = [q for q in (src_ if isinstance(src_, (list, tuple)) else [src_]) if q is not None]
            if plain(gray) and src_ and all(plain(q) for q in src_):
                continue
            # Feit: met het raster weggepoetst (0,35 mm) toont het resultaat hetzelfde als een bronscan (foto, verloop):
            # wat je ziet is wat er gescand is; alleen het rasterpatroon wisselt bij de naad (TGHT 2022: hoes, inserts)
            def soft(img_):
                return cv2.GaussianBlur(np.asarray(img_, np.float32), (0, 0), max(1., .35 * px_mm))
            g_ = soft(gray)
            if any(q.shape == gray.shape and (lambda d_: float(np.quantile(np.abs(d_ - np.median(d_)), .9)) <= 4)(g_ - soft(q))
                   and float(np.corrcoef(g_.ravel(), soft(q).ravel())[0, 1]) >= .9 for q in src_ if soft(q).std() >= 2) \
                    or any(q.shape == gray.shape and soft(q).std() < 2 and g_.std() < 2.5 for q in src_):
                continue
        if max(dark_position, light_position) >= 0:
            broader = line_crossings(gray, broad_width, length, horizontal, tolerance)
            if any(pos >= 0 and row[max(0, pos-1):pos+2].max() >= 13
                   for pos, row in zip((dark_position, light_position), broader)):
                continue
        if evidence is not None and point_index < len(evidence):
            source_patch = evidence[point_index]
            candidates = source_patch if isinstance(source_patch, (list, tuple)) else [source_patch]
            if any(p is not None and faithful_source_patch(gray, p, tolerance) for p in candidates):
                continue
        if diagnostics is not None:
            diagnostics.append((x, y, 'line continuity not confirmed'))
        unresolved += 1
    return unresolved


def sample_analysis_patch(output, sx, sy, transform):
    """Compare at the source analysis scale, integrating native pixels first.

    A bilinear point sample of a 600/1200-dpi print aliases its ink raster;
    source evidence was already reduced before placement. This changes only the
    inspection pixels, never the exported image or the acceptance thresholds.
    """
    density = max(np.linalg.norm(transform[:, 0]), np.linalg.norm(transform[:, 1]))
    pad = max(1, math.ceil(density))
    left, top = max(0, int(sx.min())-pad), max(0, int(sy.min())-pad)
    right = min(output.shape[1], math.ceil(float(sx.max()))+pad+1)
    bottom = min(output.shape[0], math.ceil(float(sy.max()))+pad+1)
    area = output[top:bottom, left:right]
    if density > 1.01:
        height, width = area.shape[:2]
        view = cv2.resize(area, (max(1, round(width/density)), max(1, round(height/density))),
                          interpolation=cv2.INTER_AREA)
        mx = (sx-left+.5)*(view.shape[1]/width)-.5
        my = (sy-top+.5)*(view.shape[0]/height)-.5
    else:
        view, mx, my = area, sx-left, sy-top
    return cv2.remap(view, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)


def stitch_quality(img, masks, ks, best, height, width, scale, support=None, px_mm=None, locations=None, evidence=None):
    """Eindcontrole van het samenvoegen, op rekenschaal. Telt plekken waar
    (1) een naad door tekst loopt en de twee deelscans daar zichtbaar
    verschoven liggen (hoes lag niet plat), en (2) de gebruikte deelscan in
    tekst veel minder scherp is dan een andere deelscan van die plek.
    scale: uitvoerpixels per rekenpixel, voor de gemelde verschuiving."""
    grays, sharp = [], []
    for k in ks:
        g = cv2.cvtColor(np.clip(img[k][0:height], 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        grays.append(g)
        sharp.append(cv2.GaussianBlur(cv2.Laplacian(g, cv2.CV_32F) ** 2, (0, 0), 2) * masks[k])
    peak = np.max(sharp, axis=0)
    covered = np.any([masks[k] for k in ks], axis=0)
    report = {'seam_text_shift': 0, 'seam_text_max_px': 0.0, 'blurry_text': 0}
    if not covered.any():
        return report
    ink = peak > 25 * max(float(np.median(peak[covered])), 1.0)
    chosen = np.zeros((height, width), np.float32)
    for n in range(len(ks)):
        chosen[best == n] = sharp[n][best == n]
    seam = np.zeros((height, width), bool)
    seam[:, 1:] |= best[:, 1:] != best[:, :-1]
    seam[1:, :] |= best[1:, :] != best[:-1, :]
    seam &= covered
    half = QUALITY_WINDOW // 2
    window = cv2.createHanningWindow((QUALITY_WINDOW, QUALITY_WINDOW), cv2.CV_32F)
    for y in range(half, height - half, half):
        for x in range(half, width - half, half):
            sl = np.s_[y - half:y + half, x - half:x + half]
            if ink[sl].mean() < 0.03:
                continue
            if (chosen[sl][ink[sl]].mean() < 0.5 * peak[sl][ink[sl]].mean()):
                report['blurry_text'] += 1
            if not seam[sl].any():
                continue
            labels, counts = np.unique(best[sl], return_counts=True)
            pair = [int(v) for v in labels[np.argsort(-counts)][:2] if v < len(ks)]
            if len(pair) < 2 or min(masks[ks[v]][sl].mean() for v in pair) < 0.95:
                continue
            # OpenCV applies the window in-place. Keep the source observations
            # intact for later line checks and saved final-image evidence.
            (dx, dy), response = cv2.phaseCorrelate(grays[pair[0]][sl].copy(),
                                                   grays[pair[1]][sl].copy(), window)
            if response < QUALITY_RESPONSE or max(abs(dx), abs(dy)) > QUALITY_WINDOW / 6:
                continue            # geen duidelijke match (weinig contrast) of uitslag tegen de vakrand: ruis
            shift = float(np.hypot(dx, dy)) * scale
            if shift >= QUALITY_SHIFT:
                report['seam_text_shift'] += 1
                report['seam_text_max_px'] = round(max(report['seam_text_max_px'], shift), 1)
    if support is not None and px_mm:
        report['seam_line_conflicts'] = seam_line_conflicts(grays, masks, ks, best, support, px_mm, locations, evidence)
    return report


def quality_messages(report):
    """Meldingen voor de eindcontrole in de app (leeg = niets gevonden)."""
    # Eén losse treffer is meestal structuur (goud vinyl, foto) en geen tekst.
    # 'blurry_text' staat alleen in het logboek: die meting gaf nog vals alarm.
    messages = []
    if report.get('seam_text_shift', 0) >= QUALITY_MIN_SPOTS:
        messages.append(f"At {report['seam_text_shift']} spots a seam runs through text with up to ~{report['seam_text_max_px']:.0f} px shift; "
                        'the cover probably wasn’t flat in a partial scan. Check the text or rescan that part.')
    return messages


def seam_regions(img, masks, ks, height, width):
    """Per plek één deelscan, met naden waar de scans het meest op elkaar lijken
    (wit papier, tussen regels) in plaats van dwars door tekst. Graph cut op
    ~SEAM_SIDE px. Geeft per pixel het scannummer (255 = geen keuze) of None."""
    f = min(1.0, SEAM_SIDE / max(height, width))
    w, h = max(1, round(width * f)), max(1, round(height * f))
    try:
        images, small = [], []
        for k in ks:
            images.append(cv2.UMat(cv2.resize(img[k][0:height], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)))
            small.append(cv2.UMat(cv2.resize(masks[k].astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)))
        found = cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD').find(images, [(0, 0)] * len(ks), small)
    except cv2.error as error:
        print(f'seams: graph cut unsuccessful ({error.code}); used a coherent choice', flush=True)
        return None
    choice = np.full((h, w), 255, np.uint8)
    for n, m in enumerate(found):
        m = m.get() if hasattr(m, 'get') else np.asarray(m)
        choice[(m > 0) & (choice == 255)] = n
    choice = cv2.resize(choice, (width, height), interpolation=cv2.INTER_NEAREST)
    for n, k in enumerate(ks):
        choice[(choice == n) & ~masks[k]] = 255              # nooit buiten de eigen scan
    return choice


TONE_FIELD_MM = 10.0   # zo glad is het toonveld per deelscan (mm); fijn genoeg voor het licht dat langs de glasrand afloopt
TONE_FIELD_MAX = 0.12  # hooguit 12% lichter of donkerder
TONE_EDGE_MM = 1.0     # zo ver van de rand van een deelscan telt hij niet mee (randschaduw van de scanner, ±0,5 mm)


GLASS_NODES = (8, 11)  # vakken over het glas (breedte x hoogte, ±27 mm): fijn genoeg voor de afloop langs de rand
GLASS_MAX = 0.15       # hooguit 15% correctie door het lichtpatroon


GLASS_FOR_RECORDS = os.environ.get('SLEEVE_GLASS_RECORDS') == '1'   # proef 14 nov: lichtpatroon ook op dekkend vinyl (doorschijnend: kleur schijnt door)
SEAM_LEVEL = False   # uit: verschoof de toon langs de scanranden en daarmee de gevonden hoesrand (Queen-insert,
if os.environ.get('SLEEVE_SEAM_LEVEL') == '1':
    SEAM_LEVEL = True  # optie 'Level tone across the seams' (0.5.33): alleen voor de pixels, de randen komen uit een gewone run
                     # Deena front 1-1,4 mm in de vaste toets, 13 nov); op de backs maakte het nauwelijks verschil
GLASS_FILE = '.sleeve-glas.json'   # het gemeten lichtpatroon, bij de scans bewaard (zelfde scanner, zelfde sessie)


def glass_apply(img, ks, H, Tsh, sizes, height, width, pattern, q=8):
    """Het lichtpatroon van het glas (vakken over het glas, factor rond 1) uit elke deelscan halen."""
    hs, ws = max(8, height // q), max(8, width // q)
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float64)
    cx, cy = xx * q + q / 2, yy * q + q / 2
    grid = cv2.resize(np.asarray(pattern, np.float32), (256, 256), interpolation=cv2.INTER_CUBIC)
    for k in ks:
        Hi = np.linalg.inv(Tsh @ H[k])
        u = (Hi[0, 0] * cx + Hi[0, 1] * cy + Hi[0, 2]) / sizes[k][0]
        v = (Hi[1, 0] * cx + Hi[1, 1] * cy + Hi[1, 2]) / sizes[k][1]
        f_ = cv2.remap(grid, (np.clip(u, 0, 1) * 255).astype(np.float32), (np.clip(v, 0, 1) * 255).astype(np.float32), cv2.INTER_LINEAR)
        gain = (1.0 / np.maximum(f_, .5)).astype(np.float32)
        full = cv2.resize(gain, (width, height), interpolation=cv2.INTER_LINEAR)
        old = img[k].gain
        base_field = old.field.astype(np.float32) if isinstance(old, FieldGain) else 1.0
        img[k].gain = FieldGain(old.plane if isinstance(old, FieldGain) else old, (full * base_field).astype(np.float16))


def glass_load(path, sizes, ks):
    """Een eerder (op een hoes) gemeten lichtpatroon, als het bij scans van dezelfde maat hoort (hetzelfde glas,
    zelfde resolutie; de rekenschaal mag verschillen, die is een vaste factor)."""
    import json
    try:
        data = json.load(open(path))
    except (OSError, ValueError, TypeError):
        return None
    w0, h0 = data.get('scan_size', [0, 0])
    if not w0 or not h0:
        return None
    for k in ks:
        w_, h_ = sizes[k]
        if abs(w_ / h_ - w0 / h0) > .01 * (w0 / h0):
            return None                                    # andere verhouding: niet het hele glas, of een ander glas
    return data


def glass_fields(img, masks, ks, H, Tsh, sizes, height, width, px_mm, q=8, log=print, save=None):
    """Feit: elke deelscan komt van hetzelfde glas met dezelfde lamp; het licht valt op elke plek van
    het glas in elke scan even sterk (bij deze scanner is het rechterkwart ±3% donkerder, gemeten op
    12 scans, 12 nov). En: een egaal stuk papier of achtergrond heeft overal dezelfde kleur. Dus: in
    elke scan, per egale kleur, hoe licht is die op elke plek van het glas ten opzichte van zijn eigen
    midden; het midden daarvan over alle scans is het lichtpatroon. Dat wordt uit elke scan gehaald, ook
    waar maar één scan ligt (een hoek op de glasrand). Alleen als alle scans even groot zijn (het hele
    glas) en er genoeg egaal vlak is; anders niets."""
    if len(ks) < 2 or not px_mm:
        return False
    dims = {sizes[k] for k in ks}
    ws0, hs0 = max(s_[0] for s_ in dims), max(s_[1] for s_ in dims)
    if any(abs(w_ - ws0) > .01 * ws0 or abs(h_ - hs0) > .01 * hs0 for w_, h_ in dims):
        return False                                       # niet het hele glas gescand: plek op het glas onbekend
    hs, ws = max(8, height // q), max(8, width // q)
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float64)
    cx, cy = xx * q + q / 2, yy * q + q / 2
    nx, ny = GLASS_NODES
    sums = np.zeros((ny, nx)); cnt = np.zeros((ny, nx)); vals = [[[] for _ in range(nx)] for _ in range(ny)]
    maps = {}
    for k in ks:
        bgr = np.empty((height, width, 3), np.uint8)
        for y0 in range(0, height, 512):
            y1 = min(height, y0 + 512)
            bgr[y0:y1] = np.clip(img[k][y0:y1], 0, 255).astype(np.uint8)
        small = cv2.resize(bgr, (ws, hs), interpolation=cv2.INTER_AREA); bgr = None
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        m = cv2.resize(masks[k].astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST)
        m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(round(1.5 * px_mm / q)))) > 0
        L = lab[:, :, 0]
        sm = cv2.GaussianBlur(L, (0, 0), max(1.0, 1.5 * px_mm / q))
        flat = m & (np.abs(L - sm) < 2.0) & (L > 60) & (L < 252)
        Hi = np.linalg.inv(Tsh @ H[k])
        u = (Hi[0, 0] * cx + Hi[0, 1] * cy + Hi[0, 2]) / sizes[k][0]
        v = (Hi[1, 0] * cx + Hi[1, 1] * cy + Hi[1, 2]) / sizes[k][1]
        maps[k] = (u, v)
        if flat.sum() < 200:
            continue
        # egale kleuren: de grootste groepen gelijke tint. Op tint (a, b) en grof op helderheid: het lichtverloop
        # zelf (een paar procent) mag een kleur niet in tweeën splitsen, anders meet je het niet meer.
        key = np.round(lab[flat] / np.array([40.0, 6.0, 6.0], np.float32)).astype(np.int32)
        uniq, inv_, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        fy, fx = np.nonzero(flat)
        for g in np.argsort(counts)[::-1][:3]:
            if counts[g] < .08 * flat.sum() or counts[g] < 150:
                break
            centre = np.median(lab[flat][inv_.ravel() == g], 0)
            close = flat & (np.linalg.norm(lab[:, :, 1:] - centre[1:], axis=2) < 5.0) & (np.abs(lab[:, :, 0] - centre[0]) < 25.0)
            if close.sum() < 150:
                continue
            ref_ = float(np.median(L[close]))
            r_ = L[close] / max(ref_, 1.0)
            uu, vv = u[close], v[close]
            inside = (uu >= 0) & (uu < 1) & (vv >= 0) & (vv < 1)
            ci = np.minimum((uu[inside] * nx).astype(int), nx - 1); cj = np.minimum((vv[inside] * ny).astype(int), ny - 1)
            for a_, b_, val in zip(cj, ci, r_[inside]):
                vals[a_][b_].append(val)
    med = np.full((ny, nx), np.nan); n_ = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            if len(vals[j][i]) >= 40:
                med[j, i] = float(np.median(vals[j][i])); n_[j, i] = len(vals[j][i])
    if np.isnan(med).mean() > .5:
        log('scanner light pattern: too little even area to measure; not used')
        return False
    w_ = (~np.isnan(med)).astype(np.float32); vfill = np.nan_to_num(med, nan=1.0).astype(np.float32)
    num = cv2.GaussianBlur(vfill * w_, (0, 0), 1.0); den = cv2.GaussianBlur(w_, (0, 0), 1.0)
    pattern = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 1.0)
    pattern = pattern / float(np.median(pattern[~np.isnan(med)]))
    lim = np.log1p(GLASS_MAX); pattern = np.exp(np.clip(np.log(pattern), -lim, lim))
    if os.environ.get('STITCH_GLASS'):
        print('GLASS rows (top->bottom), columns left->right, % (measured cells):')
        for j in range(ny):
            print('  ' + ' '.join(('%+5.1f' % (100 * (pattern[j, i] - 1))) if not np.isnan(med[j, i]) else '   . ' for i in range(nx)))
    log(f'scanner light pattern from even areas: {100 * (pattern.min() - 1):+.1f}% to {100 * (pattern.max() - 1):+.1f}% over the glass')
    glass_apply(img, ks, H, Tsh, sizes, height, width, pattern, q)
    if save:
        import json, datetime
        try:
            tmp = save + f'.{os.getpid()}.tmp'
            json.dump({'pattern': np.round(pattern, 5).tolist(), 'scan_size': [int(ws0), int(hs0)], 'samples': int(n_.sum()),
                       'measured': datetime.datetime.now().isoformat(timespec='seconds')}, open(tmp, 'w'))
            os.replace(tmp, save)
        except OSError:
            pass                                           # alleen-lezen map: dan niet bewaren
    return True


def tone_fields(img, masks, ks, height, width, px_mm, lighter, q=8):
    """Feit: twee deelscans van hetzelfde stuk tonen dezelfde toon. Het lichtvlak per deelscan (gain_correct)
    vangt een vaste zweem en een rechtlijnig verloop; wat overblijft (een hoes die niet overal even plat
    ligt, het licht dat per stand anders valt) gaf een harde stap op de naad (Urban Solitude, backs).
    Per deelscan een glad veld dat hem in de overlap gelijk maakt aan de andere, en dat buiten de overlap
    geleidelijk uitdooft. Bij papier is de lichtste de maat (schaduw en een opgetilde hoes maken alleen
    donkerder); bij een plaat het gemiddelde (glans maakt juist lichter)."""
    if len(ks) < 2 or not px_mm:
        return
    hs, ws = max(8, height // q), max(8, width // q)
    pq = px_mm / q
    L, M = {}, {}
    for k in ks:
        g = np.empty((height, width), np.uint8)
        for y0 in range(0, height, 512):
            y1 = min(height, y0 + 512)
            g[y0:y1] = cv2.cvtColor(np.clip(img[k][y0:y1], 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gs = cv2.resize(g, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32); g = None
        m = cv2.resize(masks[k].astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST)
        m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(round(TONE_EDGE_MM * pq)))) > 0
        sig = max(1.0, 3.0 * pq)
        mb = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sig)
        L[k] = cv2.GaussianBlur(gs * m, (0, 0), sig) / np.maximum(mb, 1e-6); M[k] = m & (mb > .5)
    stack = np.stack([np.where(M[k], L[k], np.nan) for k in ks])
    count = np.sum(~np.isnan(stack), 0)
    with np.errstate(all='ignore'):
        cons = np.nanmax(stack, 0) if lighter else np.nanmean(stack, 0)
    big = max(2.0, TONE_FIELD_MM * pq); lim = np.log1p(TONE_FIELD_MAX)
    for k in ks:
        w = (M[k] & (count >= 2) & (L[k] > 20)).astype(np.float32)
        if w.sum() < 50:
            continue
        r = np.zeros((hs, ws), np.float32)
        r[w > 0] = np.clip(np.log(cons[w > 0] / np.maximum(L[k][w > 0], 1e-3)), -lim, lim)
        num = cv2.GaussianBlur(r * w, (0, 0), big); den = cv2.GaussianBlur(w, (0, 0), big)
        field = num / np.maximum(den, .01 * float(den.max()))   # in en vlak bij de overlap de volle correctie; verder weg dooft hij uit
        full = cv2.resize(np.exp(np.clip(field, -lim, lim)).astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        if os.environ.get('STITCH_TONE'):np.save(os.environ['STITCH_TONE']+f'_{k}.npy',np.stack([L[k],np.where(M[k],1,0).astype(np.float32),cons.astype(np.float32),field.astype(np.float32),w]))   # hulpbestand
        old = img[k].gain
        base_field = old.field.astype(np.float32) if isinstance(old, FieldGain) else 1.0
        img[k].gain = FieldGain(old.plane if isinstance(old, FieldGain) else old, (full * base_field).astype(np.float16))


LEVEL_MAX = 0.15       # seam leveling: hooguit 15% lichter of donkerder per plek


def seam_level(img, masks, ks, best, height, width, px_mm, q=8, log=print):
    """Seam leveling (zoals 'naadloze tonen' in Photoshop). Feit: het beeld heeft op een naad geen sprong;
    alleen de deelscans verschillen daar in licht. Per plek is er één deelscan gekozen (best). Gezocht:
    één glad correctieveld C over het hele beeld, zodat (gekozen scan + C) op elke naad doorloopt zoals
    de scans zelf daar verlopen, en binnen elk stuk zo vlak mogelijk is (het verloop binnen een scan blijft).
    Daarna wordt elke deelscan op dat resultaat afgestemd, ook waar hij niet gekozen is (voor het mengen).
    Alleen helderheid (één factor voor alle kanalen); kleur regelt het lichtvlak per scan al."""
    if len(ks) < 2 or not px_mm:
        return False
    hs, ws = max(8, height // q), max(8, width // q)
    sig = max(1.0, 2.0 * px_mm / q)
    Ls, Ms = [], []
    for k in ks:
        g = np.empty((height, width), np.uint8)
        for y0 in range(0, height, 512):
            y1 = min(height, y0 + 512)
            g[y0:y1] = cv2.cvtColor(np.clip(img[k][y0:y1], 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gs = cv2.resize(g, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32); g = None
        m = cv2.resize(masks[k].astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST)
        m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(round(0.8 * px_mm / q)))) > 0
        mb = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sig)
        Lk = cv2.GaussianBlur(gs * m, (0, 0), sig) / np.maximum(mb, 1e-6)
        Ls.append(np.log(np.maximum(Lk, 4.0))); Ms.append(m & (mb > .6))
    Ls = np.stack(Ls); Ms = np.stack(Ms)
    R = cv2.resize(best.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    R = np.clip(R, 0, len(ks) - 1)
    V = np.take_along_axis(Ms, R[None], 0)[0]              # de gekozen scan is hier bruikbaar
    yy, xx = np.mgrid[0:hs, 0:ws]
    def edges(axis):
        if axis == 1:
            p_, q_ = (yy[:, :-1], xx[:, :-1]), (yy[:, 1:], xx[:, 1:])
        else:
            p_, q_ = (yy[:-1, :], xx[:-1, :]), (yy[1:, :], xx[1:, :])
        i, j = R[p_], R[q_]
        both = V[p_] & V[q_]
        same = i == j
        # naad: beide scans moeten op beide plekken bruikbaar zijn; het gewenste verschil in C is
        # het verschil tussen de scans daar (gemiddeld over beide plekken)
        ok_i = Ms[i, p_[0], p_[1]] & Ms[i, q_[0], q_[1]]; ok_j = Ms[j, p_[0], p_[1]] & Ms[j, q_[0], q_[1]]
        gq = .5 * ((Ls[i, q_[0], q_[1]] - Ls[j, q_[0], q_[1]]) + (Ls[i, p_[0], p_[1]] - Ls[j, p_[0], p_[1]]))
        w = np.where(same, 1.0, np.where(ok_i & ok_j, 1.0, 0.0)) * both
        g_ = np.where(same, 0.0, gq) * (w > 0)
        return w.astype(np.float64), g_.astype(np.float64)
    wh, gh = edges(1); wv, gv = edges(0)
    nseam = int(((R[:, :-1] != R[:, 1:]) & (wh > 0)).sum() + ((R[:-1, :] != R[1:, :]) & (wv > 0)).sum())
    if nseam < 10:
        return False
    lam = 1e-3
    Vf = V.astype(np.float64)
    def A(C):
        out = lam * C * Vf
        d = (C[:, 1:] - C[:, :-1]) * wh; out[:, :-1] -= d; out[:, 1:] += d
        d = (C[1:, :] - C[:-1, :]) * wv; out[:-1, :] -= d; out[1:, :] += d
        return out
    b = np.zeros((hs, ws))
    t = wh * gh; b[:, :-1] -= t; b[:, 1:] += t
    t = wv * gv; b[:-1, :] -= t; b[1:, :] += t
    # geconjugeerde gradiënten (matrixvrij)
    C = np.zeros((hs, ws)); r_ = b - A(C); pv = r_.copy(); rs = float((r_ * r_).sum())
    for it in range(1500):
        Ap = A(pv); alpha = rs / max(float((pv * Ap).sum()), 1e-30)
        C += alpha * pv; r_ -= alpha * Ap; rs_new = float((r_ * r_).sum())
        if rs_new < 1e-10 * max(float((b * b).sum()), 1e-30):
            break
        pv = r_ + (rs_new / rs) * pv; rs = rs_new
    C -= np.median(C[V])                                    # gemiddeld niets lichter of donkerder
    lim = np.log1p(LEVEL_MAX)
    T = np.take_along_axis(Ls, R[None], 0)[0] + C           # doel: gekozen scan + correctie
    log(f'seam leveling: {nseam} seam steps, {it + 1} iterations, correction {100 * (np.exp(np.percentile(C[V], 2)) - 1):+.1f}% to {100 * (np.exp(np.percentile(C[V], 98)) - 1):+.1f}%')
    big = max(1.0, 3.0 * px_mm / q)
    for n, k in enumerate(ks):
        m = Ms[n] & V
        d = np.where(m, np.clip(T - Ls[n], -lim, lim), 0.0).astype(np.float32)
        wgt = m.astype(np.float32)
        num = cv2.GaussianBlur(d * wgt, (0, 0), big); den = cv2.GaussianBlur(wgt, (0, 0), big)
        dk = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0.0)
        # buiten wat deze scan met het doel deelt: de correctie van de dichtstbijzijnde gedeelde plek, uitdovend
        gain = np.exp(np.clip(dk, -lim, lim)).astype(np.float32)
        full = cv2.resize(gain, (width, height), interpolation=cv2.INTER_LINEAR)
        old = img[k].gain
        base_field = old.field.astype(np.float32) if isinstance(old, FieldGain) else 1.0
        img[k].gain = FieldGain(old.plane if isinstance(old, FieldGain) else old, (full * base_field).astype(np.float16))
    return True



LOCAL_ALIGN = os.environ.get('SLEEVE_LOCAL_ALIGN', '1') == '1'   # 0.5.45: deelscans plaatselijk op elkaar laten vallen
LOCAL_MIN_MM = 0.15     # kleinere verschillen laten we zoals ze zijn
LOCAL_MAX_MM = 2.0      # meer is geen buiging van het papier maar een fout in de meting


LOCAL_Q = 2             # velden op 1/2 schaal van het canvas


def local_fields(warped, frames, order, px_mm, q=LOCAL_Q, log=print):
    """Plaatselijke verschuiving per deelscan (canvas, 1/q schaal, in canvaspixels). De deelscans liggen als geheel op
    elkaar (schuiven + draaien op de print in het midden). Een hoes ligt niet in elke scan even plat: aan de rand kan
    dezelfde rand van het papier in twee scans tienden van millimeters verschillen (Queen 2021 back: 0,7 mm; waar de
    scans wisselen, een trapje in de rand). Feit: in de overlap tonen beide scans hetzelfde papier; waar iets te zien
    is (print, de rand) meet de optische stroom hoeveel de ene ten opzichte van de andere ligt. Elke scan schuift naar
    het gemiddelde van de scans op die plek: bij twee scans de helft, bij drie of vier een derde of een kwart van elk
    paarverschil (15 nov: altijd de helft schoof bij 3-4 scans 1,5-2 keer te ver; fake back: 0,07-0,34 mm over, nu
    0,01-0,02 mm). Waar niets te meten is of het verschil klein is, blijft alles zoals het was."""
    if len(order) < 2 or not px_mm:
        return {}
    Hc, Wc = frames[order[0]].shape[:2]
    sw, sh = max(1, Wc // q), max(1, Hc // q)
    pm = px_mm / q                                          # pixels per mm op deze schaal
    er = np.ones((max(3, int(1.0 * pm)) | 1,) * 2, np.uint8)
    g, m = {}, {}
    for k in order:
        w8 = np.clip(warped[k], 0, 255).astype(np.uint8)
        g[k] = cv2.resize(cv2.cvtColor(w8, cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
        m[k] = cv2.erode(cv2.resize(frames[k].astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST), er) > 0
    acc = {k: np.zeros((sh, sw, 2), np.float32) for k in order}
    used = False
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); dis.setFinestScale(0); dis.setPatchStride(2)
    sig = max(1.0, 2.0 * pm)
    for i, a_ in enumerate(order):
        for b_ in order[i + 1:]:
            ov = m[a_] & m[b_]
            if ov.sum() < (20 * pm) ** 2:
                continue
            ys, xs = np.where(ov); pad = int(5 * pm)
            y0, y1 = max(0, ys.min() - pad), min(sh, ys.max() + pad + 1); x0, x1 = max(0, xs.min() - pad), min(sw, xs.max() + pad + 1)
            ga, gb, o_ = np.ascontiguousarray(g[a_][y0:y1, x0:x1]), np.ascontiguousarray(g[b_][y0:y1, x0:x1]), ov[y0:y1, x0:x1]
            F = dis.calc(ga, gb, None)                      # ga(x) ~ gb(x + F)
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
            gbw = cv2.remap(gb, xx + F[..., 0], yy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            err = cv2.GaussianBlur(np.abs(ga.astype(np.float32) - gbw.astype(np.float32)), (0, 0), 1.5)
            gx = cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.5)
            mag = np.hypot(F[..., 0], F[..., 1])
            # betrouwbaar: waar iets te zien is en de stroom de twee scans ook echt op elkaar legt (na verschuiven klein verschil)
            # Optical flow may confuse lighting with shape when one scan
            # casts a shadow along an opening. Require both sources to show
            # an edge with the same direction and light-to-dark orientation.
            bx = cv2.Sobel(gbw, cv2.CV_32F, 1, 0, ksize=3)
            by = cv2.Sobel(gbw, cv2.CV_32F, 0, 1, ksize=3)
            common_strength = cv2.GaussianBlur(np.hypot(bx, by), (0, 0), 1.5)
            shared_edge = cv2.GaussianBlur(gx * bx + gy * by, (0, 0), 1.5)
            gradient_product = cv2.GaussianBlur(np.hypot(gx, gy) * np.hypot(bx, by), (0, 0), 1.5)
            common = (common_strength > 40) & (shared_edge > .8 * gradient_product)
            conf = (o_ & (grad > 40) & common & (err < 25) & (mag < LOCAL_MAX_MM * pm)).astype(np.float32)
            if conf.sum() < 200:
                continue
            wb = cv2.GaussianBlur(conf, (0, 0), sig)
            Fs = cv2.GaussianBlur(F * conf[..., None], (0, 0), sig) / np.maximum(wb, 1e-6)[..., None]
            t = np.clip(wb / 0.05, 0, 1) * cv2.GaussianBlur(o_.astype(np.float32), (0, 0), sig)
            Fs *= t[..., None]
            if float(np.percentile(np.hypot(Fs[..., 0], Fs[..., 1])[o_], 99)) < LOCAL_MIN_MM * pm:
                continue
            acc[a_][y0:y1, x0:x1] -= Fs / 2
            acc[b_][y0:y1, x0:x1] += Fs / 2
            used = True
            log(f'NOTE: scans {a_ + 1} and {b_ + 1} laid on each other locally (up to {float(np.hypot(Fs[..., 0], Fs[..., 1]).max()) / pm:.2f} mm)')
    if not used:
        return {}
    # gemiddelde over de scans op elke plek: som van de paarverschillen / aantal scans (zacht, zoals Fs)
    ns = sum(cv2.GaussianBlur(m[k].astype(np.float32), (0, 0), sig) for k in order)
    share = (2.0 / np.maximum(2.0, ns))[..., None]
    for k in order:
        acc[k] *= share
    out = {}
    for k in order:
        d = acc[k]
        if float(np.abs(d).max()) < .5 * LOCAL_MIN_MM * pm:
            continue
        out[k] = (d * q).astype(np.float32)                 # in canvaspixels
    return out


def apply_field(arr, d, q, interp):
    """arr (canvas) opnieuw bemonsteren: nieuw(x) = oud(x + d(x)); d op 1/q schaal in canvaspixels."""
    H_, W_ = arr.shape[:2]
    D = cv2.resize(d, (W_, H_), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:H_, 0:W_].astype(np.float32)
    return cv2.remap(arr, (xx + D[..., 0]).astype(np.float32), (yy + D[..., 1]).astype(np.float32), interp, borderMode=cv2.BORDER_CONSTANT)


TONE_INSET_MM = 2.5     # de grove toon wordt alleen gemeten op het onderdeel zelf, zo ver binnen de gevonden rand
                        # (= de schaduwmaat van scan_feiten.trim_edge_lines; ±1 sigma van TONE_SIGMA)


def tone_support(out, ox, oy, meta, part, fmt, px_mm, height, width, workbase):
    """Feit: wat buiten de gevonden rand ligt haalt het eindresultaat nooit; de grove toon wordt daarom alleen op het
    onderdeel zelf gemeten. Deksel, achtergrond en randschaduw verschillen per scan en lieten per scan een andere waas of
    een trapje in de rand van het onderdeel komen (hoes, insert, plaat, cd).
    De rand wordt gezocht met scan_feiten op de gewone menging (zelfde code en regels als de route erna); `out` is die
    menging op de rekenschaal, bijgesneden op wat gescand is (ox, oy = de bijsnijding op het canvas).
    Uitkomst (canvasschaal, één kaart): 0 buiten de gevonden rand, 0,5 op het onderdeel binnen TONE_INSET_MM van de rand,
    1 op het onderdeel vanaf TONE_INSET_MM. Buiten de rand blijft de toon precies zoals voorheen: dat haalt het eindresultaat
    nooit, maar de randzoeker in de route kijkt wel naar dat beeld (GF voorkant: een paar grijswaarden anders in de schaduw
    buiten de rand verschoof een zwakke rand 2,2 mm). None als er geen rand gevonden is."""
    import json, shutil, subprocess, tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    d = tempfile.mkdtemp(prefix='toon_', dir=os.path.dirname(os.path.abspath(workbase)) or '.')
    try:
        img = os.path.join(d, 'mix.png'); crop = os.path.join(d, 'crop.png')
        # de rand zoeken op de helft van de rekenschaal: 4x minder pixels, en de rand wordt toch 2,5 mm binnen gebruikt
        half = cv2.resize(out, (max(1, out.shape[1] // 2), max(1, out.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        qx, qy = out.shape[1] / half.shape[1], out.shape[0] / half.shape[0]
        meta = dict(meta, dpi=meta['dpi'] / 2, scans=[dict(sc, corners=[[x / qx, y / qy] for x, y in sc['corners']]) for sc in meta['scans']])
        if not cv2.imwrite(img, half, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            return None
        json.dump(meta, open(img + '.scans.json', 'w'))
        cmd = [sys.executable, os.path.join(here, 'scan_feiten.py'), img, part, '--out', crop] + (['--format', fmt] if fmt else [])
        subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        F = json.load(open(crop + '.feiten.json'))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if F.get('verdict') not in ('GOED', 'TWIJFEL'):
        return None
    x_h, x_w = out.shape[:2]
    O = np.zeros((height, width), np.uint8)
    if F.get('cut_circle_px'):
        cx, cy, r = F['cut_circle_px']; cx = cx * qx + ox; cy = cy * qy + oy; r *= (qx + qy) / 2     # terug naar de rekenschaal
        cv2.circle(O, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1)
        ho = F.get('hole_offset_mm') or [0, 0]
        hr = (F.get('hole_diameter_mm') or 7.24) / 2 + 0.9          # het middengat hoort niet bij het onderdeel
        cv2.circle(O, (int(round(cx + ho[0] * px_mm)), int(round(cy + ho[1] * px_mm))), int(round(hr * px_mm)), 0, -1)
    elif F.get('precise_mm') and 'angle' in F:
        L, T, Wd, Hd = [v * px_mm for v in F['precise_mm']]
        Rm = cv2.getRotationMatrix2D((x_w / 2, x_h / 2), F['angle'], 1.0); Ri = cv2.invertAffineTransform(Rm)
        pts = np.float32([[L, T], [L + Wd, T], [L + Wd, T + Hd], [L, T + Hd]])
        pts = (pts @ Ri[:, :2].T + Ri[:, 2] + np.float32([ox, oy])).astype(np.int32)
        cv2.fillPoly(O, [pts], 1)
    else:
        return None
    if not O.any():
        return None
    dist = cv2.distanceTransform(O, cv2.DIST_L2, 5) / px_mm
    print(f'tone measured on the {part} only, {TONE_INSET_MM} mm inside the found edge (edge verdict {F["verdict"]})', flush=True)
    return (0.5 * (dist > 0) + 0.5 * (dist >= TONE_INSET_MM)).astype(np.float32)


def tone_blur(x, m, sup, sigma, blur):
    """Grove toon van x (gewicht m). Met sup (steunkaart 0..1): alleen gemeten waar sup telt; waar er te weinig steun in
    de buurt is (de scan ziet daar bijna niets van het onderdeel) terugval op alles wat de scan laat zien."""
    def plain():
        return blur(x * m[..., None], sigma) / np.maximum(blur(m, sigma), 1e-6)[..., None]
    if sup is None:
        return plain()
    inside = np.minimum(2 * sup, 1)                                         # kaart: 0 buiten, 0,5 binnen de inset, 1 daarbinnen
    ak = m * np.maximum(2 * sup - 1, 0); sa = blur(ak, sigma)
    t = np.minimum(np.maximum((sa - .02) * 12.5, 0), 1) * inside            # buiten de rand: de gewone toon, zoals voorheen
    rob = blur(x * ak[..., None], sigma) / np.maximum(sa, 1e-6)[..., None]
    if not ((t < 1) & (m > 0)).any():           # overal genoeg steun (het midden van het onderdeel): de gewone toon is niet nodig
        return rob
    t = t[..., None]
    return plain() * (1 - t) + rob * t


def scans_meta(order, a_sizes, Tsh, H, O, a, f, ref_dpi, small=False):
    """Waar elke scan in het (bijgesneden) uitvoerbeeld ligt: de .scans.json naast het resultaat (small: de kleine schaal)."""
    klein = a.klein or small
    outlines = []
    for k in order:
        hk, wk = int(a_sizes[k][1]), int(a_sizes[k][0])
        cor = np.float32([[0, 0], [wk, 0], [wk, hk], [0, hk]]).reshape(-1, 1, 2)
        cor = cv2.perspectiveTransform(cv2.perspectiveTransform(cor, Tsh @ H[k]), O).reshape(-1, 2)
        outlines.append({'scan': os.path.basename(a.delen[k]), 'corners': (cor / (1 if (klein or f >= 1) else f)).round(1).tolist()})
    return {'scale': 1 if (klein or f >= 1) else 1 / f, 'dpi': (ref_dpi or 0) * (f if (klein and f < 1) else 1), 'scans': outlines}


def feather_weight(mask):
    """Gewicht dat vanaf de rand van het geldige deel zacht oploopt."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return (np.minimum(d, FEATHER) / FEATHER) ** 2 + (mask * 1e-4)


def gain_correct(im, mask, comp, cmask, step=16, allocator=None, release=None, fit_only=False):
    """Pas im aan zodat het in de overlap gelijk is aan comp.

    Per kanaal een vlak a + b*x + c*y in log(comp/im), robuust geschat op
    rustige (geen randen), niet-overbelichte pixels. Een vlak kan nooit vlekken
    of blokken maken; het vangt een vaste kleurzweem plus het lichtverloop.
    """
    h, w = mask.shape
    sm = lambda x: cv2.resize(x, (w // step, h // step), interpolation=cv2.INTER_AREA)
    ov = ((mask & cmask) & (cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) > EDGE)
          & (cv2.distanceTransform(cmask.astype(np.uint8), cv2.DIST_L2, 5) > EDGE)).astype(np.uint8)
    if im.dtype == np.uint8:
        kernel=cv2.getGaussianKernel(25,3,cv2.CV_32F)
        A=sm(cv2.sepFilter2D(im,cv2.CV_32F,kernel,kernel))
    else:
        A = sm(cv2.GaussianBlur(im, (0, 0), 3))
    if release:release()
    B = sm(cv2.GaussianBlur(comp, (0, 0), 3))
    if release:release()
    ovs = sm(ov.astype(np.float32)) > 0.99
    g = cv2.cvtColor(np.clip(A, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    ok = ovs & (A.min(axis=2) > 8) & (A.max(axis=2) < 247) & (B.min(axis=2) > 8) & (B.max(axis=2) < 247)
    if ok.sum() > 200:
        ok &= grad < np.percentile(grad[ok], 60)       # alleen egale stukken (geen naadfouten)
    if ok.sum() < 100:
        return None if fit_only else (im, None)
    yy, xx = np.nonzero(ok)
    X = np.stack([np.ones_like(xx), xx / (w / step), yy / (h / step)], 1).astype(np.float64)
    fx=np.arange(w,dtype=np.float32)[None,:]
    # Eén gedeeld lichtverloop voor alle kanalen, per kanaal alleen een vaste
    # zweem. Het vlak wordt in de (smalle) overlap gemeten maar over de hele
    # scan doorgetrokken: losse hellingen per kanaal gaven daar rode/grijze
    # gloed op wit papier. De helling is begrensd, gemeten rond de overlap.
    r = np.log(B[ok] / A[ok])
    lum = r.mean(axis=1)
    sel = np.ones(len(lum), bool)
    for _ in range(3):                                  # uitschieters eruit
        coef, *_ = np.linalg.lstsq(X[sel], lum[sel], rcond=None)
        res = lum - X @ coef
        sel = np.abs(res) < max(2.5 * np.std(res[sel]), 0.01)
    coef[1:] = np.clip(coef[1:], -MAX_SLOPE, MAX_SLOPE)
    coef[0] = np.median(lum[sel] - X[sel, 1:] @ coef[1:])
    coefficients=[]
    for c in range(3):
        cast = np.clip(np.median(r[sel, c] - lum[sel]), -MAX_CAST, MAX_CAST)
        coefficients.append(np.array([coef[0] + cast, coef[1], coef[2]]))
    if fit_only:return GainPlane(coefficients,(h,w))
    out=allocator("corrected",im.shape) if allocator else np.empty_like(im)
    gm=allocator("gain",im.shape) if allocator else np.empty_like(im)
    # Write all three channels before flushing this band, instead of rereading
    # and flushing the same mapped pages once per channel. Arithmetic is unchanged.
    for y0 in range(0,h,256):
        y1=min(h,y0+256);fy=np.arange(y0,y1,dtype=np.float32)[:,None]
        for c,coef in enumerate(coefficients):
            gmap=np.exp(coef[0]+coef[1]*fx/w+coef[2]*fy/h)
            gm[y0:y1,:,c]=np.clip(gmap,0.85,1.15)
            out[y0:y1,:,c]=im[y0:y1,:,c]*gm[y0:y1,:,c]
        if release:release()
    return out, gm


def edge_maps(warped, frames, order, size, sc=0.2):
    """Randen per scan (binnen het eigen scanvak, dus geen scanranden), het
    sterkste per plek; plus een gemiddeld grijsbeeld. Op verkleinde schaal."""
    w, h = int(size[0] * sc), int(size[1] * sc)
    Gx = np.zeros((h, w), np.float32); Gy = np.zeros((h, w), np.float32)
    acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
    for k in order:
        g = cv2.resize(cv2.cvtColor(warped[k], cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        f = cv2.erode(cv2.resize(frames[k].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST), np.ones((7, 7), np.uint8)) > 0
        Gx = np.maximum(Gx, np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) * f)
        Gy = np.maximum(Gy, np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)) * f)
        acc += g * f; cnt += f
    lim = max(np.percentile(Gx, 99.5), np.percentile(Gy, 99.5))
    Gx = np.minimum(Gx, lim); Gy = np.minimum(Gy, lim)   # tekst niet laten domineren
    return Gx, Gy, acc / np.maximum(cnt, 1), cnt > 0


def ring_support(G, pts):
    """Gemiddelde randsterkte langs een omtrek (punten op verkleinde schaal)."""
    h, w = G.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1); y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    # beste waarde in een kleine omgeving (de rand ligt niet op de pixel precies)
    Gm = cv2.dilate(G, np.ones((5, 5), np.uint8))
    return float(Gm[y, x].mean())


def find_circle(gray, cover, G, sc=0.2):
    """De plaat: de grootste duidelijke cirkel. Geeft (cx, cy, r) op verkleinde schaal of None."""
    h, w = gray.shape
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    cs = cv2.HoughCircles(g8, cv2.HOUGH_GRADIENT, dp=2, minDist=min(w, h), param1=60, param2=40,
                          minRadius=int(0.30 * min(w, h)), maxRadius=int(0.55 * max(w, h)))
    if cs is None:
        return None
    best = None
    for cx, cy, r in cs[0][:5]:
        t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        pts = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)
        sup = ring_support(G, pts)
        if best is None or sup > best[0]:
            best = (sup, (float(cx), float(cy), float(r)))
    return best


def centre_hole(gray, cover, circle):
    """Asgat midden in de cirkel: daar zie je het deksel doorheen. Het label
    eromheen wijkt duidelijk af van het deksel. Een hoes met een rond motief
    heeft dat niet. Alleen bruikbaar als het label van kleur verschilt."""
    cx, cy, r = circle
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]; d = np.hypot(xx - cx, yy - cy)
    outer = cover & (d > 1.05 * r)
    ring = (d > .06 * r) & (d < .12 * r)
    if outer.sum() < 50 or ring.sum() < 50:
        return False
    lid, label = float(np.median(gray[outer])), float(np.median(gray[ring]))
    if abs(label - lid) < 20:
        return False
    hole_r = max(2.0, HOLE_RADIUS * r); best = 1e9; step = max(1, int(hole_r / 3))
    for oy in range(-int(.04 * r), int(.04 * r) + 1, step):
        for ox in range(-int(.04 * r), int(.04 * r) + 1, step):
            spot = np.hypot(xx - cx - ox, yy - cy - oy) < hole_r
            if spot.any():
                best = min(best, abs(float(gray[spot].mean()) - lid))
    return best < .35 * abs(label - lid)


def text_ink(Gx, Gy, gradient):
    """Drukwerk: randen in beide richtingen dicht bij elkaar (letters, beeld).
    Een rechte lijn (hoesrand, deksel, scannerstreep) heeft er maar één."""
    t = INK_EDGE * np.percentile(gradient, 99.5)
    k = (INK_WINDOW, INK_WINDOW)
    fx = cv2.boxFilter((np.abs(Gx) > t / 2).astype(np.float32), -1, k)
    fy = cv2.boxFilter((np.abs(Gy) > t / 2).astype(np.float32), -1, k)
    return ((fx > INK_BOTH) & (fy > INK_BOTH) & (gradient > t)).astype(np.float32)


def edge_box(Gx, Gy, ink, angle, c, ratio=1.0):
    """Kader op de eerste duidelijke rand buiten de tekst, per zijde (gedraaid
    over 'angle'). Geeft een kandidaat zoals find_square die kent, of None."""
    h, w = Gx.shape
    R = cv2.getRotationMatrix2D(c, angle, 1.0)
    rotated = cv2.warpAffine(ink, R, (w, h))
    Py = np.abs(cv2.warpAffine(Gy, R, (w, h))).sum(axis=1); Px = np.abs(cv2.warpAffine(Gx, R, (w, h))).sum(axis=0)
    edges = []
    for P, span in ((Py, ink_span(rotated.sum(axis=1))), (Px, ink_span(rotated.sum(axis=0)))):
        if span is None:
            return None
        m = max(2, int(INK_MARGIN * (span[1] - span[0])))      # buitenste tekstregel is geen papierrand
        before, after = P[:max(0, span[0] - m)], P[span[1] + m:]
        if len(before) < 3 or len(after) < 3:
            return None
        # eerst naar buiten lopend: de papierrand ligt vóór de rand van het scanbereik
        lo = next((i for i in range(len(before) - 1, -1, -1) if before[i] >= .5 * before.max()), None)
        hi = next((i for i in range(len(after)) if after[i] >= .5 * after.max()), None)
        if lo is None or hi is None:
            return None
        edges.append((lo, span[1] + m + hi))
    (y0, y1), (x0, x1) = edges
    Sx, Sy = x1 - x0, y1 - y0
    if min(Sx, Sy) <= 0 or abs(max(Sx, Sy) / min(Sx, Sy) - ratio) > .10 * ratio:
        return None
    # Een hoes is vierkant (binnenkant: 2 x 1): de te korte zijde aanvullen, aan de
    # kant waar op die afstand de duidelijkste rand ligt (Hotel New York: 3716 x 3516).
    short = max(min(Sx, Sy), max(Sx, Sy) / ratio); size_y, size_x = (ratio * short, short) if Sy > Sx else (short, ratio * short)
    def square_up(lo, hi, P, S):
        S = int(round(S))
        if hi - lo >= S:
            return lo, hi
        options = [(lo, lo + S), (hi - S, hi)]
        options = [(a, b) for a, b in options if a >= 0 and b < len(P)]
        if not options:
            return lo, hi
        return max(options, key=lambda ab: P[ab[0]] + P[ab[1]])
    y0, y1 = square_up(y0, y1, Py, size_y); x0, x1 = square_up(x0, x1, Px, size_x)
    return (0.0, angle, int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def known_box(Gx, Gy, ink, angle, c, ratio, S):
    """Kader van de bekende maat (S px op verkleinde schaal) rond de tekst,
    gedraaid over 'angle'. Binnen de ruimte die de tekst laat, de plek waar de
    randen het sterkst zijn (is de papierrand aan één kant wel te zien, dan
    sluit het kader daarop aan). None als de tekst er niet in past of het kader
    buiten het canvas valt."""
    h, w = ink.shape
    R = cv2.getRotationMatrix2D(c, angle, 1.0)
    rotated = cv2.warpAffine(ink, R, (w, h))
    Py = np.abs(cv2.warpAffine(Gy, R, (w, h))).sum(axis=1); Px = np.abs(cv2.warpAffine(Gx, R, (w, h))).sum(axis=0)
    spans = ink_span(rotated.sum(axis=0)), ink_span(rotated.sum(axis=1))
    if None in spans:
        return None
    (ax, bx), (ay, by) = spans
    Sx, Sy = (int(round(S * ratio)), int(round(S))) if bx - ax > by - ay else (int(round(S)), int(round(S * ratio))) if ratio > 1 else (int(round(S)), int(round(S)))
    if bx - ax > Sx or by - ay > Sy or Sx > w or Sy > h:
        return None
    def place(a, b, S, P, n):
        m = max(2, int(INK_MARGIN * S))                  # buitenste tekstregel is geen papierrand
        lo, hi = max(0, b + m - S), min(a - m, n - 1 - S)
        if lo > hi:
            return min(max(int(round((a + b - S) / 2)), 0), n - S)
        return max(range(lo, hi + 1), key=lambda t: P[t] + P[t + S])
    return (0.0, angle, place(ax, bx, Sx, Px, w), place(ay, by, Sy, Py, h), Sx, Sy)


def ink_span(profile):
    """Van waar tot waar ligt de tekst langs één as (0,5-99,5% van de inkt)."""
    total = float(profile.sum())
    if total <= 0:
        return None
    cumulative = np.cumsum(profile) / total
    return int(np.searchsorted(cumulative, .005)), int(np.searchsorted(cumulative, .995))


def find_square(Gx, Gy, size, sc=0.2, ratio=1.0, px_mm=None):
    """Het vierkant (hoes) dat het best op de randen past. Voor elke kleine
    draaiing: profiel van horizontale en verticale randen, en het paar randen
    op afstand S (x en y bijna gelijk) met de meeste steun. ratio=2: een
    uitgeklapte binnenkant (twee panelen naast elkaar), lange zijde = 2 x korte.
    Geeft (masker op canvas, hoek, kader na draaien, steun) of None."""
    h, w = Gx.shape
    c = (w / 2, h / 2)
    Smin, Smax = int(0.55 * min(w, h)), min(w, h) - 2
    long_y = h >= w                                   # alleen van belang bij ratio > 1
    gradient = np.hypot(Gx, Gy)
    ink = text_ink(Gx, Gy, gradient)
    def search():
        best = None
        def pairs(P):
            n = len(P); out = np.full(n, -1.0); arg = np.zeros(n, int)
            for S in range(Smin, n - 1):
                v = P[:n - S] + P[S:]; i = int(np.argmax(v)); out[S] = v[i]; arg[S] = i
            return out, arg
        for ang in np.arange(-4, 4.01, 0.1):
            R = cv2.getRotationMatrix2D(c, ang, 1.0)
            Py = cv2.warpAffine(Gy, R, (w, h)).sum(axis=1); Px = cv2.warpAffine(Gx, R, (w, h)).sum(axis=0)
            by_, ay = pairs(Py); bx_, ax = pairs(Px)
            # korte zijde S op de ene as, de andere as rond ratio x S
            short, other = (bx_, by_) if ratio > 1 and long_y else (by_, bx_)
            for S in range(Smin, Smax + 1):
                if short[S] < 0:
                    continue
                lo, hi = int(ratio * S * 0.96), min(int(ratio * S * 1.04) + 1, Smax if ratio == 1 else len(other) - 1)
                if lo > hi:
                    continue
                j = lo + int(np.argmax(other[lo:hi + 1]))
                if other[j] < 0:
                    continue
                sc_ = short[S] + other[j]
                if ratio > 1 and long_y:
                    candidate = (sc_, ang, ax[S], ay[j], S, j)
                else:
                    candidate = (sc_, ang, ax[j], ay[S], j, S)
                if best is None or sc_ > best[0]:
                    best = candidate
        return best
    def outside(candidate):
        """Deel van de tekst (sterke randen) dat buiten het kader valt."""
        _, ang, x0, y0, Sx, Sy = candidate
        rotated = cv2.warpAffine(ink, cv2.getRotationMatrix2D(c, ang, 1.0), (w, h))
        m = max(2, int(.01 * max(Sx, Sy)))
        inside = rotated[max(0, y0 - m):y0 + Sy + m, max(0, x0 - m):x0 + Sx + m].sum()
        return 1 - inside / max(float(rotated.sum()), 1.0)
    best = search()
    if best is None:
        return None
    # A closed outer rectangle is stronger evidence than unrelated profile
    # peaks (scanner lid on one side, printed artwork on another).
    closed=[]
    for threshold in ((8,12,20,30) if ratio == 1 else ()):
        edges=cv2.morphologyEx((gradient>threshold).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour)
            if area<.60*w*h:continue
            polygon=cv2.approxPolyDP(contour,.015*cv2.arcLength(contour,True),True)
            if len(polygon)!=4 or not cv2.isContourConvex(polygon):continue
            rect=cv2.minAreaRect(contour);rw,rh=rect[1]
            if min(rw,rh)<=0 or max(rw,rh)/min(rw,rh)>1.03 or area/(rw*rh)<.96:continue
            angle=rect[2]
            if angle>45:angle-=90
            if abs(angle)>4:continue
            corners=cv2.boxPoints(rect)
            if np.any(corners[:,0]<3) or np.any(corners[:,0]>w-4) or np.any(corners[:,1]<3) or np.any(corners[:,1]>h-4):continue
            rotation=cv2.getRotationMatrix2D(c,angle,1)
            aligned=corners@rotation[:,:2].T+rotation[:,2]
            left,top=aligned.min(axis=0);right,bottom=aligned.max(axis=0)
            closed.append((area,angle,int(round(left)),int(round(top)),int(round(right-left)),int(round(bottom-top))))
    closed_choice=False
    if closed:
        candidate=max(closed,key=lambda item:item[0])
        # Preserve established good crops: replace the profile result only when
        # the closed boundary disagrees materially in position or dimensions.
        discrepancy=max(abs(candidate[i]-best[i]) for i in range(2,6))/max(candidate[4:6])
        if discrepancy>.025:best=candidate;closed_choice=True
    judged = best
    if ink.sum() >= INK_MIN_AREA * w * h and outside(best) > INK_OUTSIDE:
        # Wit vel op het witte deksel, of een gedrukte balk/lijn: de papierrand is
        # zwak en een kader binnen de hoes scoort hoger (Deena & Jim tekstvel: 27%
        # van de tekst erbuiten; Hotel New York-voorkant: de ANOUK-balk). Neem dan
        # per zijde de eerste duidelijke rand buiten de tekst.
        wider = edge_box(Gx, Gy, ink, best[1], c, ratio)
        if wider is not None and outside(wider) <= INK_OUTSIDE:
            print(f'cover: frame left out {outside(best):.0%} of the text; chose a wider frame on the paper edge', flush=True)
            # De steun (hoes of plaat?) blijft die van het sterke vierkant: een zwakke
            # papierrand mag een tekstvel niet tot plaat maken (Deena & Jim).
            judged = best; best = wider; closed_choice = False
        elif px_mm:
            known = known_box(Gx, Gy, ink, best[1], c, ratio, LP_MM * px_mm * sc)
            if known is not None and outside(known) <= INK_OUTSIDE:
                # Wit vel op het witte deksel en geen papierrand te zien (Queen for a Day,
                # tekstvel kant A: kader op de scanrand sneed 26% van de tekst weg).
                # Dan de bekende maat (12") rond de tekst.
                print(f'cover: frame left out {outside(best):.0%} of the text and no paper edge found; used the known 12" size around the text', flush=True)
                judged = best; best = known; closed_choice = False
    _, ang, x0, y0, Sx, Sy = best
    # vierkant terug naar het canvas (volle resolutie)
    R = cv2.getRotationMatrix2D(c, ang, 1.0); Ri = cv2.invertAffineTransform(R)
    cor = np.float32([[x0, y0], [x0 + Sx, y0], [x0 + Sx, y0 + Sy], [x0, y0 + Sy]])
    cor = (cor @ Ri[:, :2].T + Ri[:, 2]) / sc
    m = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(m, [np.round(cor).astype(np.int32)], 1)
    m = cv2.erode(m, np.ones((9, 9), np.uint8))           # schaduwrandje van de hoes eraf
    inset = max(5, int((.010 if closed_choice else .006) * max(Sx,Sy) / sc))
    box = (int(x0 / sc) + inset, int(y0 / sc) + inset, int(Sx / sc) - 2*inset, int(Sy / sc) - 2*inset)
    G = np.hypot(Gx, Gy)
    def perimeter(candidate):
        _, a, px, py, pw, ph = candidate
        Rc = cv2.invertAffineTransform(cv2.getRotationMatrix2D(c, a, 1.0))
        corners = np.float32([[px, py], [px + pw, py], [px + pw, py + ph], [px, py + ph]]) @ Rc[:, :2].T + Rc[:, 2]
        per = []
        for (xa, ya), (xb, yb) in zip(corners, np.roll(corners, -1, axis=0)):
            t = np.linspace(0, 1, 200)[:, None]; per.append(np.hstack([xa + t * (xb - xa), ya + t * (yb - ya)]))
        return np.vstack(per)
    support = ring_support(G, perimeter(judged))      # zoals voorheen: steun van het sterke vierkant
    return m > 0, float(ang), box, support, closed_choice


def flatten_paper(out, inside, sc=0.125):
    """Schaduw in wit papier wegwerken (hoes lag niet helemaal plat, in elke
    scan net anders). Meet de helderheid van het witte papier, maak er een glad
    lichtveld van en licht de donkere plekken op tot het papier overal even wit
    is. Alleen ophelderen, alleen bij hoezen met flink wat wit papier."""
    h, w = out.shape[:2]
    small = cv2.resize(out[:, :, :3], (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    ins = cv2.erode(cv2.resize(inside.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST), np.ones((5, 5), np.uint8)) > 0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV); L = hsv[:, :, 2].astype(np.float32)
    if ins.sum() < 1000:
        return
    top = np.percentile(L[ins], 97)
    paper = ins & (L > top - 30) & (hsv[:, :, 1] < 30)
    paper = cv2.erode(paper.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # niet tegen tekst aan
    if paper.sum() < 0.15 * ins.sum():
        return
    sig = 0.012 * max(small.shape)
    pf = paper.astype(np.float32)
    field = cv2.GaussianBlur(L * pf, (0, 0), sig) / np.maximum(cv2.GaussianBlur(pf, (0, 0), sig), 1e-3)
    far = cv2.GaussianBlur(pf, (0, 0), sig) < 0.02                   # ver van papier: niets doen
    gain = np.clip(top / np.maximum(field, 1), 1.0, 1.25)
    gain[far] = 1.0
    gain = cv2.GaussianBlur(gain, (0, 0), sig / 2)
    g = cv2.resize(gain, (w, h), interpolation=cv2.INTER_CUBIC)
    out[:, :, :3] = np.clip(out[:, :, :3].astype(np.float32) * g[..., None], 0, 255).astype(np.uint8)
    print(f'white paper evened out (max {gain.max():.2f}x brightened)')
    return g


def masked_detail_energy(gray, valid, sigma):
    """Measure image detail without counting the border of the scan as ink.

    A Laplacian needs real neighbours on every side. Padding and the edge of
    an aligned scan cannot provide evidence of sharper printed material.
    """
    interior = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
    energy = cv2.Laplacian(gray, cv2.CV_32F) ** 2
    energy *= interior
    observed = cv2.GaussianBlur(interior, (0, 0), sigma)
    return cv2.GaussianBlur(energy, (0, 0), sigma) / np.maximum(observed, .01)


def vinyl_spot_evidence(image, covered, sigma):
    """Find dark spots and flat, light neutral alternatives in one aligned scan."""
    light = image.mean(axis=2)
    valid = cv2.erode(covered.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    coverage = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
    nearby = cv2.GaussianBlur(light * valid, (0, 0), sigma) / np.maximum(coverage, .01)
    deficit = nearby - light
    neutral = image.max(axis=2) - image.min(axis=2) <= 18
    eligible = valid & (coverage >= .98) & (nearby >= 207) & neutral
    return light, eligible & (np.abs(deficit) <= 2.5), eligible & (deficit >= 9)


def render_full(paths, order, H, Tsh, O, f, out_small_wh, canvas_small, masks, wts, gainm, best, ks, flat, shape, source_scales=None, release_work=None, sources_ready=None, output_allocator=None,before_sources=None,fields=None,support=None,vinyl_px_mm=None,cover_dpi=None,paper_white=None):
    """Het resultaat opnieuw opbouwen op volle scanresolutie, in blokken.
    Alles wat berekend is (uitlijning, kleurcorrectie, gewichten, welke scan
    per plek de details levert, vorm) komt van de kleine schaal; de pixels
    komen rechtstreeks uit de originele scans."""
    s = 1.0 / f
    S = np.diag([s, s, 1.0]); Si = np.diag([f, f, 1.0])
    Wf, Hf = int(round(out_small_wh[0] * s)), int(round(out_small_wh[1] * s))
    print(f'building full resolution: {Wf}x{Hf} px …', flush=True)
    wo, ho = out_small_wh
    O2 = O[:2]
    def to_out(m, interp):                              # kaart canvas -> uitvoer (klein)
        result=cv2.warpAffine(m,O2,(wo,ho),flags=interp,borderMode=cv2.BORDER_CONSTANT)
        drop_array_pages(m)
        return result
    W_o = {k: to_out(wts.pop(k).astype(np.float32,copy=False), cv2.INTER_LINEAR) for k in ks}
    V_o = {k: to_out(masks.pop(k).astype(np.uint8), cv2.INTER_NEAREST) for k in ks}
    G_o = {}
    for k in ks:
        gain=gainm.pop(k,None)
        if isinstance(gain,FieldGain):G_o[k]=(gain.plane,to_out(gain.field.astype(np.float32),cv2.INTER_LINEAR))
        else:G_o[k]=gain if isinstance(gain,GainPlane) else to_out(gain,cv2.INTER_LINEAR) if gain is not None else None
    gain=None
    gc.collect()
    B_o = to_out(best.astype(np.uint8), cv2.INTER_NEAREST)
    Z_o = to_out(support.astype(np.float32), cv2.INTER_LINEAR) if support is not None else None
    F_o = to_out(flat.astype(np.float32), cv2.INTER_LINEAR) if flat is not None else None
    if release_work is not None:release_work()
    gc.collect()
    progress('Opening original scans',0,len(ks))
    with MappedScanStore(paths,ks,load,decode_into=load_into,on_reserved=before_sources,output_bytes=Wf*Hf*4,on_loaded=lambda current,total:progress('Opening original scans',current,total)) as srcs:
        if sources_ready is not None:sources_ready()
        M, polys = {}, {}
        for k in ks:
            hk, wk = srcs[k].shape[:2]
            source_Si = np.diag([source_scales[k],source_scales[k],1.0]) if source_scales is not None else Si
            M[k] = S @ O @ Tsh @ H[k] @ source_Si                    # bron (vol) -> uitvoer (vol)
            e = 6 / source_scales[k] if source_scales is not None else 6*s
            c = np.float64([[e, e, 1], [wk - e, e, 1], [wk - e, hk - e, 1], [e, hk - e, 1]])
            polys[k] = (c @ M[k].T)[:, :2]
        if shape[0] == 'disc':
            cx, cy, r = shape[1]
            cxo, cyo = (O @ [cx, cy, 1])[:2] * s; ro = r * s
        inverse_O=np.linalg.inv(O)
        sig = TONE_SIGMA * s; pad = int(3 * sig) + 8
        tile=render_tile_size()
        print(f'memory-bounded: source images via temporary disk blocks, output blocks {tile} px',flush=True)
        progress('Building image',0,1)
        out = output_allocator((Hf,Wf,4)) if output_allocator is not None else np.zeros((Hf,Wf,4),np.uint8)
        def sample(m, x0, y0, w, h, interp):               # kleine kaart -> blok op volle schaal
            A = np.array([[f, 0, (x0 + 0.5) * f - 0.5], [0, f, (y0 + 0.5) * f - 0.5]])
            return cv2.warpAffine(m, A, (w, h), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        nty = (Hf + tile - 1) // tile; ntx = (Wf + tile - 1) // tile
        replaced_spots = 0
        for ty in range(nty):
            for tx in range(ntx):
                x0, y0 = tx * tile - pad, ty * tile - pad
                w = min(tile, Wf - tx * tile) + 2 * pad; h = min(tile, Hf - ty * tile) + 2 * pad
                T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
                low = np.zeros((h, w, 3), np.float32); high = np.zeros((h, w, 3), np.float32)
                acc = np.zeros((h, w), np.float32); anym = np.zeros((h, w), bool)
                bt = sample(B_o, x0, y0, w, h, cv2.INTER_NEAREST)
                # Require two overlapping clean readings before discarding a darker one.
                check_spots = vinyl_px_mm is not None and len(ks) >= 3 and Z_o is not None
                if check_spots:
                    clean_count = np.zeros((h, w), np.uint8)
                    darker = np.zeros((h, w), bool)
                    clean_low = np.full((h, w), 256, np.float32)
                    clean_high = np.zeros((h, w), np.float32)
                    clean_pixels = np.zeros((h, w, 3), np.uint8)
                    material = sample(Z_o, x0, y0, w, h, cv2.INTER_LINEAR) >= .45
                for n, k in enumerate(ks):
                    pk = polys[k] - [x0, y0]
                    if pk[:, 0].max() < 0 or pk[:, 1].max() < 0 or pk[:, 0].min() > w or pk[:, 1].min() > h:
                        continue
                    moved = bool(fields) and k in fields
                    if moved:
                        # plaatselijke verschuiving (canvas, 1/LOCAL_Q schaal): uitvoerpunt p -> canvas c -> c + d -> p' -> bron
                        # De verbinding uitvoerpunt -> bronpunt is glad (affien plus een veld op 1/LOCAL_Q schaal): op een raster van
                        # 8 pixels uitrekenen en lineair vergroten is voor het veld hetzelfde als zijn eigen bemonstering, en 64x minder werk.
                        g_ = 8; nh_, nw_ = h // g_ + 3, w // g_ + 3
                        yy_, xx_ = np.mgrid[0:nh_, 0:nw_].astype(np.float32); xx_ = xx_ * g_ + x0; yy_ = yy_ * g_ + y0
                        cx_ = ((inverse_O[0, 0] * xx_ * f + inverse_O[0, 1] * yy_ * f + inverse_O[0, 2]) / LOCAL_Q).astype(np.float32)
                        cy_ = ((inverse_O[1, 0] * xx_ * f + inverse_O[1, 1] * yy_ * f + inverse_O[1, 2]) / LOCAL_Q).astype(np.float32)
                        dx_ = cv2.remap(fields[k][..., 0], cx_, cy_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        dy_ = cv2.remap(fields[k][..., 1], cx_, cy_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        px_ = xx_ + s * (O[0, 0] * dx_ + O[0, 1] * dy_); py_ = yy_ + s * (O[1, 0] * dx_ + O[1, 1] * dy_)
                        Mi_ = np.linalg.inv(M[k])
                        up_ = np.float32([[1.0 / g_, 0, 0], [0, 1.0 / g_, 0]])
                        mx_ = cv2.warpAffine((Mi_[0, 0] * px_ + Mi_[0, 1] * py_ + Mi_[0, 2]).astype(np.float32), up_, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                        my_ = cv2.warpAffine((Mi_[1, 0] * px_ + Mi_[1, 1] * py_ + Mi_[1, 2]).astype(np.float32), up_, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                        hk, wk_ = srcs[k].shape[:2]; e_ = 6 / source_scales[k] if source_scales is not None else 6 * s
                        # de omtrek van de scan verschuift mee: geldig is waar de verschoven plek in de scan ligt
                        # (anders bleef aan een scanrand een reepje tot ±0,7 mm leeg)
                        fr = ((mx_ >= e_) & (mx_ <= wk_ - e_) & (my_ >= e_) & (my_ <= hk - e_)).astype(np.uint8)
                        xx_ = yy_ = cx_ = cy_ = dx_ = dy_ = px_ = py_ = None
                    else:
                        fr = np.zeros((h, w), np.uint8)
                        cv2.fillPoly(fr, [np.round(pk * 16).astype(np.int32)], 1, shift=4)
                    if shape[0] == 'square':
                        mk = fr > 0
                    elif shape[0] == 'disc':
                        yy, xx = np.ogrid[0:h, 0:w]
                        mk = (fr > 0) & ((xx + x0 - cxo) ** 2 + (yy + y0 - cyo) ** 2 <= ro * ro)
                    else:
                        mk = (fr > 0) & (sample(V_o[k], x0, y0, w, h, cv2.INTER_NEAREST) > 0)
                    if not mk.any():
                        continue
                    # de scherpe pixels komen alleen uit de gekozen scan; een scan die hier niet gekozen is levert alleen de
                    # grove toon (30 px of meer uitgesmeerd): daar is de goedkope interpolatie genoeg
                    winner = (bt == n) & mk
                    interp_ = cv2.INTER_LANCZOS4 if winner.any() else cv2.INTER_LINEAR
                    if moved:
                        im = cv2.remap(srcs[k], mx_, my_, interp_, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                        mx_ = my_ = None
                    else:
                        im = cv2.warpAffine(srcs[k], (T @ M[k])[:2], (w, h), flags=interp_,
                                            borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                    original = im.copy() if paper_white and k in paper_white else None
                    if isinstance(G_o[k],tuple):
                        plane_k,field_k=G_o[k]
                        if plane_k is not None:im *= plane_k.output_block(inverse_O,f,x0,y0,w,h,out_small_wh)
                        im *= sample(field_k, x0, y0, w, h, cv2.INTER_LINEAR)[..., None]
                    elif G_o[k] is not None:
                        im *= G_o[k].output_block(inverse_O,f,x0,y0,w,h,out_small_wh) if isinstance(G_o[k],GainPlane) else sample(G_o[k], x0, y0, w, h, cv2.INTER_LINEAR)
                    if original is not None:
                        from scan_paper_light import preserve_white
                        im = preserve_white(original, im, paper_white[k])
                        original = None
                    if check_spots:
                        light, clean, dark = vinyl_spot_evidence(im, mk, max(1.2, .34 * vinyl_px_mm))
                        clean &= material
                        darker |= dark & material
                        clean_count += clean
                        chosen = clean & (light >= clean_high)
                        clean_pixels[chosen] = np.clip(im[chosen] + .5, 0, 255).astype(np.uint8)
                        np.minimum(clean_low, light, out=clean_low, where=clean)
                        np.maximum(clean_high, light, out=clean_high, where=clean)
                    mf = mk.astype(np.float32)
                    wk = (np.maximum(sample(W_o[k], x0, y0, w, h, cv2.INTER_LINEAR), 0) + 1e-4) * mf
                    lk = tone_blur(im, mf, None if Z_o is None else sample(Z_o, x0, y0, w, h, cv2.INTER_LINEAR), sig, detail_blur)
                    win = cv2.GaussianBlur(winner.astype(np.float32), (0, 0), 2 * s) if winner.any() else None
                    low += lk * wk[..., None]; acc += wk
                    if win is not None:
                        high += (im - lk) * win[..., None]
                    anym |= mk
                res = low / np.maximum(acc, 1e-6)[..., None] + high
                if check_spots:
                    repair = (clean_count >= 2) & darker & (clean_high - clean_low <= 5)
                    repair &= clean_low - res.mean(axis=2) >= 4
                    replaced_spots += int(repair[pad:h-pad, pad:w-pad].sum())
                    res[repair] = clean_pixels[repair]
                if F_o is not None:
                    res *= sample(F_o, x0, y0, w, h, cv2.INTER_LINEAR)[..., None]
                blk = np.zeros((h, w, 4), np.uint8)
                blk[..., :3] = np.clip(res + 0.5, 0, 255).astype(np.uint8); blk[..., 3] = np.where(anym, 255, 0)
                out[ty * tile:ty * tile + h - 2 * pad, tx * tile:tx * tile + w - 2 * pad] = blk[pad:h - pad, pad:w - pad]
                srcs.drop_pages()
            flush_array_rows(out,ty*tile,min(Hf,(ty+1)*tile))
            srcs.space.committed((min(Hf,(ty+1)*tile)-ty*tile)*Wf*4)
            print(f'  {int(100 * (ty + 1) / nty)}%', flush=True)
        if replaced_spots:
            print(f'NOTE: vinyl spots replaced from overlapping scans: {replaced_spots} pixels', flush=True)
        if cover_dpi and len(ks) > 1:
            from scan_line_source import repair_lines
            progress('Checking interrupted lines against original scans')
            repaired = repair_lines(out, [srcs[k] for k in ks],
                                    [np.linalg.inv(M[k])[:2] for k in ks], cover_dpi)
            if repaired:
                print(f'NOTE: cover lines restored from overlapping scans: {len(repaired)} sections',
                      flush=True)
        return out


def repair_paper_gaps(out,before_fill=None):
    """Continue local paper tones into small blank gaps; never fill with one global colour.
    Large gaps and gaps beside artwork stay transparent and are reported."""
    h,w = out.shape[:2]
    scale = min(1.0, 1600/max(h,w))
    small = cv2.resize(out, (round(w*scale),round(h*scale)), interpolation=cv2.INTER_AREA)
    hole = (small[...,3] < 128).astype(np.uint8)
    if not hole.any():
        return out
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hole)
    accepted = np.zeros_like(hole)
    for k in range(1,n):
        m = (labels == k).astype(np.uint8)
        ring = (cv2.dilate(m,np.ones((11,11),np.uint8)) > 0) & (hole == 0)
        pixels = small[ring,:3]
        if len(pixels) < 30:
            continue
        # Paper only: little chroma and no printed detail adjacent to the gap.
        hsv = cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2HSV).reshape(-1,3)
        plain = np.percentile(hsv[:,1],95) < 65 and np.percentile(hsv[:,2],10) > 140
        if stats[k,cv2.CC_STAT_AREA] < .08*hole.size and plain:
            accepted[m > 0] = 1
    if accepted.any():
        if before_fill is not None:before_fill()
        # Exclude the shadow and interpolation fringe at the edges of missing paper.
        accepted = cv2.dilate(accepted,np.ones((31,31),np.uint8))
        # Harmonic interpolation continues the surrounding tones smoothly.
        # Telea inpainting produces streaks in large areas of blank paper.
        fixed = None
        valid = (1-accepted).astype(np.float32)
        kernel = np.array([[0,.25,0],[.25,0,.25],[0,.25,0]],np.float32)
        longest = max(small.shape[:2])
        for side in (50,100,200,400,800,1600):
            side = min(side,longest)
            wh = (max(1,round(small.shape[1]*side/longest)),max(1,round(small.shape[0]*side/longest)))
            weights = cv2.resize(valid,wh,interpolation=cv2.INTER_AREA)
            known = cv2.resize(small[:,:,:3].astype(np.float32)*valid[...,None],wh,interpolation=cv2.INTER_AREA)/np.maximum(weights[...,None],1e-6)
            missing = cv2.resize(accepted,wh,interpolation=cv2.INTER_NEAREST)>0
            field = np.full_like(known,np.median(small[valid>0,:3],axis=0)) if fixed is None else cv2.resize(fixed,wh,interpolation=cv2.INTER_LINEAR)
            field[~missing] = known[~missing]
            for _ in range(300):
                mean = cv2.filter2D(field,-1,kernel,borderType=cv2.BORDER_REFLECT)
                field[missing] = mean[missing]
            fixed = field
            if side == longest:
                break
        fixed = np.clip(cv2.resize(fixed,(small.shape[1],small.shape[0])),0,255).astype(np.uint8)
        fill = cv2.resize(fixed,(w,h),interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(accepted,(w,h),interpolation=cv2.INTER_NEAREST)>0
        out[mask,:3] = fill[mask]; out[mask,3] = 255
        print(f'small empty paper areas filled in locally: {100*mask.mean():.2f}%')
    remaining = (out[:,:,3]<128).mean()
    if remaining > .001:
        print(f'NOTE: {100*remaining:.2f}% of the cover lacks scan data; left transparent.')
    return out


def upright_preview(out):
    """Same RGB thumbnail as re-opening a PNG; share full-size bytes instead."""
    from PIL import Image
    h,w=out.shape[:2]
    # RGBX prevents alpha premultiplication, matching the established RGB conversion.
    image=Image.frombuffer('RGBX',(w,h),np.ascontiguousarray(out),'raw','RGBX',0,1)
    image.thumbnail((1600,1600))
    bgr=np.asarray(image.convert('RGB'))
    return Image.fromarray(bgr[:,:,::-1].copy())


def upright_output(out,work):
    import scan_cutout as CU
    progress('Checking cover orientation')
    q=CU.reading_quadrant(upright_preview(out))
    print('orientation: '+('unknown' if q is None else str(q)),flush=True)
    if not q:return out
    view=np.rot90(out,-int(q//90))
    rotated=work.empty(('upright-output',0),view.shape,np.uint8)
    for y0 in range(0,view.shape[0],256):
        y1=min(view.shape[0],y0+256)
        rotated[y0:y1]=view[y0:y1]
        flush_array_rows(rotated,y0,y1)
        drop_array_pages(out)
    return rotated


def main():
    with contextlib.ExitStack() as resources:
        main_with_resources(resources)

def main_with_resources(resources,lease=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('uitvoer'); p.add_argument('delen', nargs='+')
    p.add_argument('--geen-kleur', action='store_true')
    p.add_argument('--niet-op-vinyl', default='', help='deze scan (bestandsnaam) alleen gebruiken waar geen andere scan ligt (Sleeve Transparant: labelscan)')
    p.add_argument('--belichting-alleen', default='', help='met --geen-kleur: alleen deze scan (bestandsnaam) gelijktrekken met de andere (Sleeve Transparant)')
    p.add_argument('--lichtkolom', default=None, help='transparant vinyl: lichtcorrectie per scannerkolom uit dit profiel (scan_lichtkolom.py)')
    p.add_argument('--rechtop',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--snelle-tussenbestanden',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--referentie', type=int)
    p.add_argument('--plaat-onderdeel', action='store_true', help=argparse.SUPPRESS)   # feitenroute: dit is een plaat (glans is hier geen schaduw)
    p.add_argument('--toon-onderdeel', default=None, help=argparse.SUPPRESS)   # feitenroute: welk onderdeel dit is (voor het zoeken van de rand van de toon)
    p.add_argument('--toon-format', default=None, help=argparse.SUPPRESS)
    p.add_argument('--klein', action='store_true', help='output at ~300 dpi (fast) instead of full scan resolution')
    p.add_argument('--vorm', choices=['auto', 'hoes', 'plaat', 'geen', 'ruw'], default='auto', help='hoes = square, plaat = round, geen = by color only, ruw = whole scans, no cropping (default: decide automatically)')
    a = p.parse_args()
    if a.lichtkolom:
        import scan_lichtkolom
        global LICHTKOLOM
        LICHTKOLOM=scan_lichtkolom.load_profile(a.lichtkolom)
        print(f'lichtkolom: profiel {a.lichtkolom} ({len(LICHTKOLOM["columns"])} kolommen)',flush=True)
    cv2.setNumThreads(2)
    if len(a.delen) < 2:
        sys.exit('at least two partial scans needed')

    # grote scans (1200 dpi) eerst verkleinen: een Mac met 8 GB loopt anders vast.
    # Eén factor voor alle delen, anders passen ze niet meer op elkaar.
    from PIL import Image as _I
    _I.MAX_IMAGE_PIXELS = None
    metadata = [_I.open(path) for path in a.delen]
    max_source_pixels=max(im.width*im.height for im in metadata)
    # Reserve the larger of preparation and full-source decoding before allocating.
    # Admission only shrinks later, so concurrent workers cannot deadlock upgrading.
    reserve=max(29+max(0,len(a.delen)-4)*3, int(np.ceil((max_source_pixels*8+512*1024**2)/(64*1024**2))))
    if a.toon_format == 'vinyl' and reserve > int(os.environ.get('SLEEVE_MEMORY_SLOTS', '72')):
        for im in metadata: im.close()
        raise ValueError('These scans cannot be prepared safely with the available memory; '
                         'process fewer scans together. No lower-resolution export was made.')
    weight = reserve if a.toon_format == 'vinyl' else min(70, reserve)
    lease=resources.enter_context(MemoryLease(weight,phase='Opening partial scans'))
    import scan_dpi
    dpis = [scan_dpi.read(path) for path in a.delen]          # feit: EXIF/JFIF goed gelezen (scan_dpi)
    ref_dpi = max(dpis) if all(50<d<5000 for d in dpis) else None
    ratios = [ref_dpi/d for d in dpis] if ref_dpi else [1.]*len(dpis)
    longest = max(max(im.size)*ratio for im,ratio in zip(metadata,ratios))
    for im in metadata: im.close()
    if max(ratios)-min(ratios)>.01:
        print('Different scan resolutions normalized: '+', '.join(str(round(d))+' dpi' for d in dpis),flush=True)
    f = min(1.0, MAX_SIDE / longest)
    if f < 1:
        print(f'computing at factor {f:.3f} ({int(longest * f)} px); output at ' + ('that scale' if a.klein else f'full resolution ({longest} px)'))
    if f < 1 and not a.klein:
        # Check scratch requirements before expensive alignment/blending can trigger swap.
        MappedScanStore(a.delen,range(len(a.delen)),load)
    source_scales = [f*ratio for ratio in ratios]
    px_mm = ref_dpi * f / 25.4 if ref_dpi else None      # canvaspixels per mm (rekenschaal)
    imgs=[];progress('Opening partial scans',0,len(a.delen))
    for index,(fn,scale) in enumerate(zip(a.delen,source_scales)):
        imgs.append(load(fn,scale,consume_cache=True));progress('Opening partial scans',index+1,len(a.delen))
    progress('Detecting edges and background')
    if a.vorm == 'ruw':
        # Feitenaanpak: geen gok over wat deksel is; overeenkomende punten op de hele
        # scan (een egaal deksel levert toch geen punten op), alleen de scanrand zelf niet.
        def whole(im):
            m = np.ones(im.shape[:2], bool); m[:6] = m[-6:] = False; m[:, :6] = m[:, -6:] = False
            return m
        valid = [whole(im) for im in imgs]
    else:
        valid = [~background_mask(im) for im in imgs]
    # kenmerken alleen op de hoes; maar is bijna alles als 'deksel' gezien (witte/crème
    # hoes op een wit deksel), zoek dan op de hele scan, anders mist hij de tekst
    def fmask(im, v):
        if v.mean() >= 0.25:
            return v
        m = np.ones(im.shape[:2], bool); m[:6] = m[-6:] = False; m[:, :6] = m[:, -6:] = False
        return m
    feats = [features(im, fmask(im, v)) for im, v in zip(imgs, valid)]
    progress('Aligning partial scans precisely')
    H, ref, W, missing = place(len(imgs), feats)
    if missing:
        # Een deelscan past nergens op. Eerst: staat er flauw drukwerk dat pas met meer contrast te zien is?
        feats2 = [features(faint_print(im), fmask(im, v)) for im, v in zip(imgs, valid)]
        H2, ref2, W2, missing2 = place(len(imgs), feats2)
        if len(missing2) < len(missing):
            print('NOTE: aligned on faint print (contrast raised for matching only): ' + ', '.join(os.path.basename(a.delen[k]) for k in missing if k not in missing2), flush=True)
            H, ref, W, missing = H2, ref2, W2, missing2
        feats2 = None
    if missing and a.vorm == 'ruw' and len(H) >= 2:
        # Dan: geen drukwerk gemeen, maar wel de papierranden (scan_edges): die lopen recht door over de deelscans.
        import scan_edges
        try:
            on_edges = scan_edges.place_on_edges(imgs, H, missing, px_mm, log=lambda text: None)
        except Exception as error:                         # geen feit gevonden: de deelscan blijft weg, met de melding hieronder
            on_edges = {}; print(f'edge placement not possible: {error}', flush=True)
        for k, Hk in on_edges.items():
            H[k] = Hk
            print('NOTE: placed on its paper edges (no print in common with the other scans): ' + os.path.basename(a.delen[k]), flush=True)
        missing = [k for k in missing if k not in H]
    if missing:
        unresolved = []
        for k in missing:
            gray = cv2.cvtColor(imgs[k],cv2.COLOR_BGR2GRAY).astype(np.float32)
            h,w = gray.shape; gray = gray[int(.15*h):int(.85*h),int(.15*w):int(.85*w)]
            detail = gray-cv2.GaussianBlur(gray,(0,0),3)
            if (np.abs(detail)>8).mean() < .001:
                print('NOTE: left out, no print in common and no paper edge to place it on: '+os.path.basename(a.delen[k]))
            else:
                unresolved.append(k)
        if unresolved or len(H)<2:
            raise ValueError('Not all scans fit together reliably: ' + ', '.join(a.delen[k] for k in missing) + '. Select only parts of the same side.')
    order = [k for k in range(len(imgs)) if k in H]
    if a.referentie:
        r = a.referentie - 1
        Hr = np.linalg.inv(H[r]); H = {k: Hr @ v for k, v in H.items()}; ref = r

    # oriëntatie altijd van het eerste deel (kleurreferentie kan een ander deel zijn)
    if order and order[0] == 0 and 0 in H:
        H0 = np.linalg.inv(H[0]); H = {k: H0 @ v for k, v in H.items()}

    # canvas
    corners = []
    for k in order:
        h, w = imgs[k].shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, H[k]))
    allc = np.concatenate(corners)
    x0, y0 = np.floor(allc.min(axis=(0, 1))).astype(int); x1, y1 = np.ceil(allc.max(axis=(0, 1))).astype(int)
    Tsh = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float); size = (x1 - x0, y1 - y0)

    warped, masks, frames = {}, {}, {}
    for k in order:
        M = (Tsh @ H[k])[:2]
        warped[k] = cv2.warpAffine(imgs[k], M, size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        masks[k] = cv2.warpAffine(valid[k].astype(np.uint8), M, size, flags=cv2.INTER_NEAREST) > 0
        fr = np.zeros(imgs[k].shape[:2], np.uint8); fr[6:-6, 6:-6] = 1
        frames[k] = cv2.warpAffine(fr, M, size, flags=cv2.INTER_NEAREST) > 0

    fields = {}; support = None
    # Isolated research switch: never enabled by the app or ordinary pipeline.
    groove_probe = a.plaat_onderdeel and os.environ.get('SLEEVE_GROOVE_FLOW_PROBE') == '1'
    if LOCAL_ALIGN and a.vorm == 'ruw' and (not a.plaat_onderdeel or groove_probe) and len(order) > 1:
        progress('Laying partial scans on each other locally')
        alignment_images = {k: faint_print(warped[k]) for k in order} if groove_probe else warped
        fields = local_fields(alignment_images, frames, order, px_mm, log=lambda t: print(t, flush=True))
        if groove_probe:
            del alignment_images
        for k, d in fields.items():
            warped[k] = apply_field(warped[k], d, LOCAL_Q, cv2.INTER_LINEAR)
            frames[k] = apply_field(frames[k].astype(np.uint8), d, LOCAL_Q, cv2.INTER_NEAREST) > 0
            masks[k] = apply_field(masks[k].astype(np.uint8), d, LOCAL_Q, cv2.INTER_NEAREST) > 0
    a_sizes = {k: (imgs[k].shape[1], imgs[k].shape[0]) for k in order}
    imgs=valid=feats=None
    gc.collect()
    # 0. hoes of plaat? Een plaat (rond) gaat prima op kleur: het deksel eromheen
    #    wordt weggesneden. Een hoes is altijd vierkant: zoek het vierkant op de
    #    hoesranden; alles erbuiten is deksel, alles erbinnen hoes, welke kleur
    #    de hoes ook heeft (wit en grijs zijn op kleur niet van het deksel te
    #    onderscheiden, en het deksel is grijzer naarmate de hoes dikker is).
    square = None; disc_c = None
    if a.vorm == 'ruw':
        # Feitenaanpak: niets wegsnijden, de hele scans samenvoegen; het uitsnijden
        # gebeurt daarna op feiten (scan_feiten.py).
        for k in order:
            masks[k] = frames[k].copy()
    if a.vorm not in ('geen', 'ruw'):
        Gx, Gy, gmean, cover = edge_maps(warped, frames, order, size)
        G = np.hypot(Gx, Gy)
        # Uitgeklapte binnenkant: het samengevoegde vlak is ongeveer twee keer zo
        # lang als breed (Deena & Jim: 6 deelscans, ~61 x 30 cm). Dan een
        # rechthoek van 2:1 zoeken; Sleeve splitst die daarna in twee panelen.
        spread = max(size) / min(size) >= SPREAD_ASPECT
        sq = find_square(Gx, Gy, size, ratio=2.0 if spread else 1.0, px_mm=px_mm) if a.vorm in ('auto', 'hoes') else None
        ci = find_circle(gmean, cover, G) if a.vorm == 'plaat' or (a.vorm == 'auto' and not spread) else None
        if os.environ.get('STITCH_DEBUG'): print('DEBUG square support', None if sq is None else round(sq[3], 1), 'cirkel', None if ci is None else round(ci[0], 1))
        # Twijfel (cirkel bijna even sterk als het vierkant, bv. doorzichtig vinyl
        # op het witte deksel, To Get Her Together): het asgat beslist.
        holed = ci is not None and sq is not None and not sq[4] and ci[0] > .85 * sq[3] and centre_hole(gmean, cover, ci[1])
        is_disc = ci is not None and (sq is None or a.vorm == 'plaat' or (not sq[4] and (ci[0] > 1.15 * sq[3] or holed)))
        if is_disc:
            cx, cy, r = (v / 0.2 for v in ci[1])
            disc = np.zeros((size[1], size[0]), np.uint8)
            cv2.circle(disc, (int(round(cx)), int(round(cy))), int(r * 1.01), 1, -1)
            disc_c = (cx, cy, r * 1.01)
            for k in order:
                masks[k] = frames[k] & (disc > 0)
            print(f'record: circle found (diameter {2 * r:.0f} px)')
        elif sq is not None:
            square = sq[:3]
            sq_mask, sq_angle, sq_box = square
            for k in order:
                masks[k] = frames[k] & sq_mask
            print(f'cover: square found ({sq_box[2]}x{sq_box[3]} px), skew {sq_angle:+.2f} degrees')

    Gx=Gy=gmean=G=cover=frames=None
    gc.collect()
    # 1. kleur/licht gelijktrekken: elk deel krijgt per kanaal een zacht
    #    verlopende correctie (vlak in log-ruimte: factor + lichtverloop van de
    #    scannerlamp), gemeten in de overlap met wat al samengevoegd is
    ks = list(order)
    height,width=size[1],size[0]
    print(f'working canvas: {width}x{height} px; work buffers on temporary disk blocks',flush=True)
    estimate=height*width*(4*len(ks)+32)
    progress('Preparing work files for merging')
    work=resources.enter_context(MappedArrayStore(estimate))
    img={k:CorrectedImage(warped[k]) for k in ks}
    wts={k:work.put(('weight',k),feather_weight(masks[k])) for k in ks}
    gainm={}
    paper_white={}
    if a.toon_format == 'vinyl' and a.toon_onderdeel == 'insert' and len(ks)>1:
        from scan_paper_light import agreement,paper_samples
        qw,qh=max(1,width//4),max(1,height//4)
        previews=[];preview_masks=[]
        for k in ks:
            pixels,covered=paper_samples(img[k].raw,masks[k],width,height)
            previews.append(pixels);preview_masks.append(covered)
            drop_array_pages(img[k].raw)
        paper_diagnostics={}
        paper=agreement(previews,preview_masks,(px_mm or 1)/4,paper_diagnostics)
        print('PAPER_WHITE_CHECK: '+__import__('json').dumps(paper_diagnostics),flush=True)
        if paper:
            for k,anchor in zip(ks,paper['white_anchors']):
                img[k].white_anchor=anchor
                paper_white[k]=anchor
            print('PAPER_WHITE: '+__import__('json').dumps(paper),flush=True)
        previews=preview_masks=None
    progress('Matching exposure between partial scans',0,len(ks))
    glass_path=os.path.join(os.path.dirname(os.path.abspath(a.delen[0])),GLASS_FILE)
    if not a.geen_kleur and len(ks)>1 and not a.plaat_onderdeel:
        # eerst het lichtpatroon van het glas uit de ruwe scans (anders trekt het lichtvlak per scan dat
        # patroon als een schuin verloop door over de hele scan: Urban Solitude backs, links 99,6 / rechts 95);
        # bewaard bij de scans, voor de platen van dezelfde release
        progress('Measuring the light of the scanner glass')
        glass_fields(img,masks,ks,H,Tsh,a_sizes,height,width,px_mm,log=lambda t:print(t,flush=True),save=glass_path)
    elif GLASS_FOR_RECORDS and not a.geen_kleur and len(ks)>1 and a.plaat_onderdeel:
        # plaat: niet op het vinyl meten (de groeven glimmen per richting anders), wel het patroon van een
        # hoes uit dezelfde map gebruiken
        stored=glass_load(glass_path,a_sizes,ks)
        if stored:
            glass_apply(img,ks,H,Tsh,a_sizes,height,width,stored['pattern'])
            pat=np.asarray(stored['pattern'])
            print(f'scanner light pattern from the sleeve scans in this folder: {100*(pat.min()-1):+.1f}% to {100*(pat.max()-1):+.1f}%',flush=True)
    only=[k for k in ks if a.belichting_alleen and os.path.basename(a.delen[k])==a.belichting_alleen] if a.belichting_alleen else []
    if not a.geen_kleur or only:
        done=[ref];todo=[k for k in ks if k!=ref]
        if only:
            # Sleeve Transparant: alleen deze scan (de labelscan, vaak uit een andere sessie) gelijktrekken met de rest;
            # de andere deelscans houden hun belichting (bij doorzichtig vinyl trok het gelijktrekken een hele scan donkerder)
            done=[k for k in ks if k not in only];todo=list(only)
        while todo:
            acc=sum(wts[d] for d in done);cm=acc>0;work.drop_pages()
            comp=None
            comp=work.empty(('composite',0),(height,width,3))
            for y0 in range(0,height,256):
                y1=min(height,y0+256)
                comp[y0:y1]=sum(img[d][y0:y1]*wts[d][y0:y1,...,None] for d in done)/np.maximum(acc[y0:y1],1e-6)[...,None]
                work.drop_pages()
            k=max(todo,key=lambda t:int((masks[t]&cm).sum()))
            if (masks[k]&cm).sum()<5000:break
            field=img[k].gain if isinstance(img[k].gain,FieldGain) else None
            src=img[k].raw if field is None else np.concatenate([img[k][y0:min(height,y0+512)] for y0 in range(0,height,512)],0)
            gain=gain_correct(src,masks[k],comp,cm,release=work.drop_pages,fit_only=True);src=None
            img[k].gain=FieldGain(gain,field.field) if field is not None else gain
            gainm[k]=img[k].gain;gain=None
            done.append(k);todo.remove(k);work.drop_pages()
            progress('Matching exposure between partial scans',len(done),len(ks))
        comp=acc=cm=None;work.discard(('composite',0))
        for k in only:                                     # Sleeve Transparant: welke belichting de labelscan kreeg (logboek)
            g_=img[k].gain
            if isinstance(g_,FieldGain):g_=g_.plane
            if isinstance(g_,GainPlane):print('label scan exposure: %s x%s'%(os.path.basename(a.delen[k]),'/'.join('%.3f'%float(np.exp(c[0])) for c in g_.coefficients[::-1])),flush=True)
    if not a.geen_kleur and len(ks)>1:
        for k in ks:
            if isinstance(img[k].gain,FieldGain):gainm[k]=img[k].gain
    # Welke deelscan per plek. Feit: scherpte zegt alleen iets waar detail is (tekst, drukwerk); daar
    # wint de scherpste (een opgetilde of wazige scan verliest). Op een egaal vlak (glad vinyl, effen
    # papier) is 'scherpte' alleen ruis en glans, en gaf dat een lappendeken van deelscans (Urban
    # Solitude blauw, 12 nov). Daar wint de deelscan waarin die plek het verst van de glasrand ligt:
    # daar ligt het onderdeel het vlakst en is het licht het gelijkmatigst.
    progress('Comparing details between partial scans',0,height)
    best=np.zeros((height,width),np.uint8)
    q=4                                                  # diepte in de scan op 1/4 rekenen (genoeg voor een gladde grens)
    depth=[]
    for k in ks:
        mk=cv2.resize(masks[k].astype(np.uint8),(max(1,width//q),max(1,height//q)),interpolation=cv2.INTER_NEAREST)
        # één rij extra onderaan (kopie van de laatste): bij een hoogte net boven een veelvoud van 768
        # viel de laatste band anders buiten de verkleinde kaart en stopte het samenvoegen (0.5.29)
        dk=cv2.distanceTransform(mk,cv2.DIST_L2,5);depth.append(np.concatenate([dk,dk[-1:]],0))
    if a.plaat_onderdeel:
        # Plaat: glans in de groeven hangt af van de richting, niet van de scherpte. Alleen sterk detail (label,
        # rand) telt; deze grove drempel (op 1/4 gemeten) gaf schoon vinyl. Met de fijne drempel van papier
        # kwam de lappendeken terug (Urban Solitude blauw kant B, 12 nov).
        tops=None
        for n,k in enumerate(ks):
            gs=np.concatenate([cv2.resize(cv2.cvtColor(np.clip(img[k][y0:min(height,y0+768)],0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY),
                                          (max(1,width//q),max(1,(min(height,y0+768)-y0)//q)),interpolation=cv2.INTER_AREA) for y0 in range(0,height,768)],0).astype(np.float32)
            shq=cv2.GaussianBlur(cv2.Laplacian(gs,cv2.CV_32F)**2,(0,0),max(1.0,(SHARP_SIGMA_MM*px_mm if px_mm else 25)/q))*q**4     # zelfde maat als op volle rekenschaal (ruwweg)
            shq[cv2.resize(masks[k].astype(np.uint8),(shq.shape[1],shq.shape[0]),interpolation=cv2.INTER_NEAREST)==0]=0
            tops=shq if tops is None else np.maximum(tops,shq)
            work.drop_pages()
        floor=float(np.percentile(tops[tops>0],40))*3.0 if (tops>0).any() else 0.0
        tops=None
    else:
        # Wat telt als detail: ruim boven de 'scherpte' van een egaal vlak, gemeten over het hele beeld, op dezelfde
        # schaal als de keuze hieronder (elke 4e pixel). Eerst op 1/4 verkleind gemeten: dat gaf een drempel ±100x te
        # hoog, kleine grijze tekst telde als leeg papier en de wazige scan won (Urban Solitude back, 12 nov).
        tops=[]
        for y0 in range(0,height,768):
            y1=min(height,y0+768);top=max(0,y0-102);bottom=min(height,y1+102);m=None
            for k in ks:
                g=cv2.cvtColor(np.clip(img[k][top:bottom],0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
                sigma = SHARP_SIGMA_MM*px_mm if px_mm else 25
                sh = (masked_detail_energy(g, masks[k][top:bottom], sigma)
                      if a.toon_format == 'vinyl' else cv2.GaussianBlur(cv2.Laplacian(g,cv2.CV_32F)**2,(0,0),sigma))[y0-top:y1-top:4,::4]
                sh[~masks[k][y0:y1:4,::4]]=0
                m=sh if m is None else np.maximum(m,sh)
                work.drop_pages()
            tops.append(m[m>0])
        tops=np.concatenate(tops) if tops else np.zeros(0,np.float32)
        floor=float(np.percentile(tops,40))*3.0 if tops.size else 0.0
        tops=None
    # Sleeve Transparant (--niet-op-vinyl): deze scan (de labelscan) levert alleen nog waar geen andere scan ligt; het label
    # zelf komt er later in (scan_feitenroute.paste_label). Een labelscan uit een andere sessie lag 180 graden gedraaid; daar
    # keert de schaduw van de groeven om en gaf hij groefsprongen in het vinyl (Together Alone geel A, 20 nov).
    excl={k for k in ks if a.niet_op_vinyl and os.path.basename(a.delen[k])==a.niet_op_vinyl and len(ks)>1}
    if excl:
        # alleen als die scan anders ligt dan de rest (meer dan 20 graden gedraaid): dan klopt de groefschaduw niet
        ang=lambda k:math.degrees(math.atan2(float(H[k][1][0]),float(H[k][0][0])))
        rest=[ang(j) for j in ks if j not in excl]
        mid=math.degrees(math.atan2(np.mean([math.sin(math.radians(v)) for v in rest]),np.mean([math.cos(math.radians(v)) for v in rest])))
        turned={k:abs((ang(k)-mid+180)%360-180) for k in excl}
        excl={k for k,v in turned.items() if v>20}
        print(('label scan not used on the vinyl where other scans lie: %s (turned %.0f degrees against the other scans)' if excl else
               'label scan lies like the other scans (%s, %.0f degrees): also used on the vinyl')%(a.niet_op_vinyl,max(turned.values())),flush=True)
    # Keep the same 102-row convolution halo; wider bands avoid recomputing it.
    for y0 in range(0,height,768):
        y1=min(height,y0+768);top=max(0,y0-102);bottom=min(height,y1+102)
        shs=[];deps=[]
        others={k:np.logical_or.reduce([masks[j][y0:y1] for j in ks if j!=k]) for k in excl}
        for n,k in enumerate(ks):
            g=cv2.cvtColor(np.clip(img[k][top:bottom],0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
            # scherpte over ±0,7 mm: breder telde de schaduwlijn langs de rand van een hoes mee als 'scherp' (de wazige
            # scan won daar de tekst, Urban Solitude back)
            sigma = SHARP_SIGMA_MM*px_mm if px_mm else 25
            paper_detail = not a.plaat_onderdeel and a.toon_format == 'vinyl'
            sh = (masked_detail_energy(g, masks[k][top:bottom], sigma)
                  if paper_detail else cv2.GaussianBlur(cv2.Laplacian(g,cv2.CV_32F)**2,(0,0),sigma))[y0-top:y1-top]
            sh[~masks[k][y0:y1]]=-1.0
            shs.append(sh)
            r0=y0//q;r1=min(depth[n].shape[0],(y1+q-1)//q+1)
            d=cv2.resize(depth[n][r0:r1],(width,(r1-r0)*q),interpolation=cv2.INTER_LINEAR)[y0-r0*q:y0-r0*q+(y1-y0)]
            if d.shape[0]<y1-y0:d=np.pad(d,((0,y1-y0-d.shape[0]),(0,0)),mode='edge')
            d[~masks[k][y0:y1]]=-1.0
            if not a.plaat_onderdeel:
                # hoes: buiten tekst de keuze van 0.5.28 (scherpte over ±25 px, gewogen naar de afstand tot de
                # scanrand). De diepste scan kiezen verlegde de gevonden hoesrand 1-1,4 mm waar een hoes niet
                # vlak lag (Deena front, Queen-insert, vaste toets 13 nov).
                s25 = (masked_detail_energy(g, masks[k][top:bottom], 25)
                       if paper_detail else cv2.GaussianBlur(cv2.Laplacian(g,cv2.CV_32F)**2,(0,0),25))[y0-top:y1-top]
                d=wts[k][y0:y1]*np.sqrt(s25+1.0);d[~masks[k][y0:y1]]=-1.0;s25=None
            if k in excl:
                sh[others[k]&masks[k][y0:y1]]=-.5;d[others[k]&masks[k][y0:y1]]=-.5   # alleen als laatste keus (-1 = niet gescand)
            deps.append(d)
            work.drop_pages()
        others=None
        S_=np.stack(shs);D_=np.stack(deps)
        order_=np.argsort(S_,0)
        first=np.take_along_axis(S_,order_[-1:],0)[0];second=np.take_along_axis(S_,order_[-2:-1],0)[0] if len(ks)>1 else np.zeros_like(first)
        # echt detail (tekst, drukwerk, de rand van een plaat): de scherpste. Ook kleine, lichte tekst: daar is
        # de ene scan duidelijk scherper dan de andere (een wazige scan gaf dubbele letters, Urban Solitude back).
        detail=(first>floor)|((first>DETAIL_RATIO*np.maximum(second,1e-6))&(first>.3*floor))
        if not a.plaat_onderdeel:
            # hoes: alleen waar één scan duidelijk scherper is (kleine tekst die in een andere scan wazig is). Losse
            # lettertjes werden daarna door het samenhangend maken weer opgeslokt door de scan eromheen; daarom het
            # hele tekstblok (±1,5 mm rond de letters) uit de scan die daar gemiddeld het scherpst is.
            det=((first>DETAIL_RATIO*np.maximum(second,1e-6))&(first>.3*floor)).astype(np.uint8)
            rr_=max(1,int(1.5*px_mm)) if px_mm else 30
            detail=cv2.dilate(det,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rr_+1,2*rr_+1)))>0;det=None
            sig=max(1.0,(1.0*px_mm) if px_mm else 20)
            Sb=np.stack([np.where(m_>=0,cv2.GaussianBlur(np.maximum(m_,0),(0,0),sig),-1.0) for m_ in S_])
            order_=np.argsort(Sb,0)[-1:];Sb=None
            best[y0:y1]=np.where(detail,order_[0],np.argmax(D_,0)).astype(np.uint8)
            progress('Comparing details between partial scans',y1,height)
            continue
        if os.environ.get('STITCH_SHARP') and y0==0:print('SHARP floor',floor,flush=True)
        best[y0:y1]=np.where(detail,order_[-1],np.argmax(D_,0)).astype(np.uint8)
        progress('Comparing details between partial scans',y1,height)
    S_=D_=shs=deps=depth=None
    value=winner=g=sh=score=better=None
    # Eén deelscan per plek: maak de keuzekaart samenhangend en neem ook de
    # grove laag vooral uit die scan. Overal mengen gaf dubbele, vage letters
    # waar een hoes niet vlak lag en de deelscans een paar pixels verschilden.
    progress('Choosing seams between partial scans')
    value=np.full((height,width),-1.0,np.float32);coherent=np.zeros((height,width),np.uint8)
    for n,k in enumerate(ks):
        share=cv2.GaussianBlur(((best==n)&masks[k]).astype(np.float32),(0,0),WINNER_SMOOTH)
        share[~masks[k]]=-1.0
        better=share>value;value[better]=share[better];coherent[better]=n
    best[:]=coherent
    if len(ks)>1:
        # Welke deelscan per plek: de scherpste (zoals 0.4.11). De naadzoeker
        # mag alleen de grens verleggen, binnen een smalle strook, zodat die
        # tussen woorden en regels loopt. Zo kiest hij nooit een wazige rand of
        # een deelscan die op die plek niet goed op de andere ligt.
        corridor=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*SEAM_CORRIDOR+1,2*SEAM_CORRIDOR+1))
        allowed={k:(cv2.dilate((best==n).astype(np.uint8),corridor)>0)&masks[k] for n,k in enumerate(ks)}
        seams=seam_regions(img,allowed,ks,height,width)
        allowed=None
        if seams is not None:best[:]=np.where(seams<255,seams,best)
        if os.environ.get('STITCH_BEST'):cv2.imwrite(os.environ['STITCH_BEST'],(best.astype(np.float32)*(255/max(1,len(ks)-1))).astype(np.uint8))   # hulpbeeld: welke deelscan per plek
        if os.environ.get('STITCH_CHOICE_ONLY'):
            # voortoets (Tests/feiten/snel.py): alleen de keuze welke deelscan waar, op rekenschaal; geen eindbeeld
            lab_=np.full(best.shape,255,np.uint8)
            for n_,k_ in enumerate(ks):lab_[(best==n_)&masks[k_]]=k_   # nummer van de scan in de invoer: gelijk tussen versies
            cv2.imwrite(os.environ['STITCH_CHOICE_ONLY'],lab_);sys.exit(0)
        if not a.geen_kleur:
            progress('Levelling tone across the seams')
            if SEAM_LEVEL and seam_level(img,masks,ks,best,height,width,px_mm,log=lambda t:print(t,flush=True)):
                for k in ks:gainm[k]=img[k].gain
    value=coherent=share=better=None
    # Accumulate one band across all sources, then write the byte output directly.
    # Per-pixel source order and float32 arithmetic match the mapped-buffer path.
    pad=int(4*TONE_SIGMA)+2
    gauss=lambda v,sg:cv2.GaussianBlur(v,(0,0),sg)
    def gauss_fast(v,sg,q=4):
        # de grove toon is zo glad dat een kwart van de resolutie genoeg is (alleen waar het beeld niet het eindresultaat is)
        h_,w_=v.shape[:2]
        if min(h_,w_)<8*q:return cv2.GaussianBlur(v,(0,0),sg)
        sm=cv2.resize(v,(w_//q,h_//q),interpolation=cv2.INTER_AREA)
        return cv2.resize(cv2.GaussianBlur(sm,(0,0),sg/q),(w_,h_),interpolation=cv2.INTER_LINEAR)
    def mix_small(sup,label,fast=False):
        blur=gauss_fast if fast else gauss
        out=np.zeros((height,width,4),np.uint8);anymask=np.zeros((height,width),bool);wsum={}
        progress(label,0,len(ks)*height)
        for y0 in range(0,height,256):
            y1=min(height,y0+256);top=max(0,y0-pad);bottom=min(height,y1+pad)
            low=np.zeros((y1-y0,width,3),np.float32);high=np.zeros_like(low)
            for n,k in enumerate(ks):
                mk=masks[k][top:bottom].astype(np.float32)
                lk=tone_blur(img[k][top:bottom],mk,None if sup is None else sup[top:bottom],TONE_SIGMA,blur)[y0-top:y1-top]
                win=cv2.GaussianBlur(((best[top:bottom]==n)&masks[k][top:bottom]).astype(np.float32),(0,0),2)[y0-top:y1-top]
                wk=wts[k][y0:y1]
                low+=lk*wk[...,None];wsum[n]=wk
                high+=(img[k][y0:y1]-lk)*win[...,None]
                work.drop_pages()
            acc=sum(wsum.values());wsum.clear()
            res=low/np.maximum(acc,1e-6)[...,None]+high
            out[y0:y1,:,:3]=np.clip(res+.5,0,255).astype(np.uint8)
            for k in ks:anymask[y0:y1]|=masks[k][y0:y1]
            out[y0:y1,:,3]=np.where(anymask[y0:y1],255,0)
            work.drop_pages()
            progress(label,len(ks)*y1,len(ks)*height)
        return out,anymask
    out=anymask=None
    if a.toon_onderdeel and a.vorm=='ruw' and len(ks)>1 and px_mm:
        # feit: de grove toon alleen op het onderdeel zelf (zie tone_support); eerst de gewone menging voor het zoeken van de rand
        import time as _t; t_0=_t.time()
        progress('Finding the edge for the tone')
        out0,am0=mix_small(None,'Finding the edge for the tone',fast=True)
        t_1=_t.time()
        ys0,xs0=np.where(am0)
        if ys0.size:
            ox0,oy0=int(xs0.min()),int(ys0.min())
            O0=np.array([[1,0,-ox0],[0,1,-oy0],[0,0,1]],float)
            support=tone_support(out0[ys0.min():ys0.max()+1,xs0.min():xs0.max()+1],ox0,oy0,
                                 scans_meta(order,a_sizes,Tsh,H,O0,a,f,ref_dpi,small=True),a.toon_onderdeel,a.toon_format,px_mm,height,width,a.uitvoer)
        out0=am0=None
        print(f'tone: plain mix {t_1-t_0:.0f}s, edge search {_t.time()-t_1:.0f}s',flush=True)
        if support is None:print('tone: no edge found for the '+a.toon_onderdeel+'; measured over everything each scan sees',flush=True)
    seam_locations = {}
    seam_evidence = {}
    if len(ks) > 1:
        audit_lines = a.toon_format == 'vinyl' and not a.plaat_onderdeel
        quality = stitch_quality(img, masks, ks, best, height, width, 1.0 / f,
                                 support if audit_lines else None, px_mm, seam_locations, seam_evidence)
        print('SLEEVE_QUALITY ' + __import__('json').dumps(quality), flush=True)
    if f<1 and not a.klein and a.vorm=='ruw':
        # bij volle resolutie vervangt render_full dit kleine beeld; alleen de dekking (bijsnijden en maat) blijft nodig: niet mengen
        anymask=np.zeros((height,width),bool)
        for k in ks:anymask|=masks[k]
        out=np.zeros((height,width,4),np.uint8);out[...,3]=np.where(anymask,255,0)
    else:
        out,anymask=mix_small(support,'Finishing transitions between partial scans')
    imgs=valid=feats=warped=frames=img=low=high=score=res=acc=g=sh=mk=lk=win=None
    work.keep_only({('weight',k) for k in ks}|{('gain',k) for k in ks})
    gc.collect()
    flat = None
    if square is not None and not a.geen_kleur:
        progress('Evening out cover lighting')
        flat = flatten_paper(out, square[0])
    progress('Straightening shape and checking edges')
    O = np.eye(3)                                         # canvas -> uitvoer (kleine schaal)
    if square is not None:
        # recht op de hoesranden en precies op het vierkant bijsnijden
        sq_mask, sq_angle, (bx, by, bw, bh) = square
        R = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), sq_angle, 1.0)
        out = cv2.warpAffine(out, R, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        out = out[by:by + bh, bx:bx + bw]
        anymask = out[:, :, 3] > 128
        O = np.array([[1, 0, -bx], [0, 1, -by], [0, 0, 1]], float) @ np.vstack([R, [0, 0, 1]])
    nz = anymask
    ys, xs = np.where(nz)
    out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    O = np.array([[1, 0, -xs.min()], [0, 1, -ys.min()], [0, 0, 1]], float) @ O
    if a.vorm == 'ruw':
        import json
        json.dump(scans_meta(order, a_sizes, Tsh, H, O, a, f, ref_dpi), open(a.uitvoer + '.scans.json', 'w'), indent=1)
    if f < 1 and not a.klein:
        shape = ('square', None) if square is not None else (('disc', disc_c) if disc_c is not None else (None, None))
        out_wh=(out.shape[1],out.shape[0])
        imgs=valid=feats=warped=frames=img=score=low=high=res=acc=msU=nz=anymask=None
        Gx=Gy=gmean=G=cover=g=sh=mk=win=out=comp=cm=ys=xs=None
        best=best.astype(np.uint8)
        gc.collect()
        field_bytes=out_wh[0]*out_wh[1]*(5*len(ks)+1+(4 if flat is not None else 0))
        field_units=max(1,int(np.ceil((field_bytes+192*1024**2)/(64*1024**2))))
        decoder_units=max(field_units,int(np.ceil((max_source_pixels*8+field_bytes+192*1024**2)/(64*1024**2))))
        if lease is not None and decoder_units>lease.weight:
            # Reject a group whose decoder would exceed its admitted maximum;
            # never wait for an upgrade beyond the preparation reservation.
            raise ValueError('This part needs more memory than was available beforehand. Process fewer parts at once.')
        def release_preparation():
            work.keep_only(set())
            if lease is not None:lease.resize(field_units)
        out = render_full(a.delen, order, H, Tsh, O, f, out_wh, size,
                          masks, wts, gainm, best, ks, flat, shape, source_scales, release_work=release_preparation,sources_ready=lambda:lease.resize(min(lease.weight,12+max(0,len(ks)-4))) if lease is not None else None,output_allocator=lambda shape:work.empty(('full-output',0),shape,np.uint8),before_sources=lambda:lease.grow(decoder_units,phase='Opening original scans') if lease is not None else None,fields=fields,support=support,
                          vinyl_px_mm=ref_dpi / 25.4 if a.plaat_onderdeel and a.toon_format == 'vinyl' and ref_dpi else None,
                          cover_dpi=ref_dpi if a.toon_onderdeel == 'front' and a.toon_format == 'vinyl'
                          and os.environ.get('SLEEVE_COVER_LINE_SOURCE', '1') == '1' else None,
                          paper_white=paper_white)
        masks=wts=gainm=best=flat=None
        gc.collect()
    # Full source maps and tile buffers have gone. Flushed output bytes can wait
    # on disk while the independent save/repair stage acquires its own budget.
    drop_array_pages(out)
    gc.collect()
    encode_units=min(70,4+int(np.ceil(out.nbytes/(64*1024**2))))
    if lease is not None:lease.release()
    final_lease=resources.enter_context(MemoryLease(encode_units,phase='Saving merged image'))
    repair_expanded=False
    def before_paper_fill():
        nonlocal final_lease,repair_expanded
        # Most covers need only the small inspection. Reserve the expensive
        # full-size local fill only when the inspected image actually needs it.
        drop_array_pages(out)
        gc.collect()
        final_lease.release()
        final_lease=resources.enter_context(MemoryLease(48,phase='Filling small open edges'))
        repair_expanded=True
    if square is not None:
        progress('Checking small open edges')
        out=repair_paper_gaps(out,before_fill=before_paper_fill)
        drop_array_pages(out)
        if repair_expanded:
            gc.collect()
            final_lease.release()
            final_lease=resources.enter_context(MemoryLease(encode_units,phase='Saving merged image'))
    if seam_locations:
        # O maps the analysis canvas to the cropped merge. The full renderer
        # uses that same mapping at native scan resolution.
        transform = O[:2] / (f if f < 1 and not a.klein else 1)
        diagnostics_dir = os.environ.get('SLEEVE_SEAM_DIAGNOSTICS_DIR')
        diagnostics = [] if diagnostics_dir else None
        for (k, other), points in seam_locations.items():
            unresolved = [] if diagnostics_dir else None
            ev_ = seam_evidence.get((k, other)); ev2 = seam_evidence.get(('other', k, other))
            if ev_ is not None and ev2 is not None and len(ev2) == len(ev_):
                ev_ = [(p, q) for p, q in zip(ev_, ev2)]
            count = unresolved_seam_points(out, points, transform, px_mm, unresolved, ev_)
            if unresolved:
                for x, y, reason in unresolved:
                    center = transform @ [x, y, 1]
                    diagnostics.append((center, reason, os.path.basename(a.delen[k]),
                                        os.path.basename(a.delen[other])))
            names = f'{os.path.basename(a.delen[k])} / {os.path.basename(a.delen[other])}'
            if len(points) > count:
                print(f'SEAM_CHECK: {names} ({len(points) - count} source differences resolved in merged image)', flush=True)
            if count:
                print(f'NOTE: cover line changes at a seam: {names} ({count} unconfirmed locations)', flush=True)
        if diagnostics:
            from scan_seam_probe import write_seam_diagnostics
            folder = write_seam_diagnostics(diagnostics_dir, a.toon_onderdeel, out, diagnostics, ref_dpi)
            from scan_seam_probe import write_source_evidence
            write_source_evidence(folder, seam_locations, seam_evidence, transform, px_mm,
                                  {k: os.path.basename(a.delen[k]) for k in ks})
            print(f'SEAM_DETAILS: {folder}', flush=True)
    if a.rechtop and square is not None:
        out=upright_output(out,work)
    progress('Saving merged image')
    if a.snelle_tussenbestanden and disc_c is not None:
        import json
        a.uitvoer += '.tif'
        if not cv2.imwrite(a.uitvoer,out,[cv2.IMWRITE_TIFF_COMPRESSION,1]):
            raise OSError('Could not write lossless intermediate')
        print('SLEEVE_INTERMEDIATE '+json.dumps({'path':a.uitvoer}),flush=True)
    elif not cv2.imwrite(a.uitvoer, out, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError('Could not save merged image')
    print(f'done: {a.uitvoer}  ({out.shape[1]}x{out.shape[0]}, {len(order)} parts, reference: {a.delen[ref]})')


if __name__ == '__main__':
    main()
