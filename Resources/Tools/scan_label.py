"""Extract a label from an explicitly identified original label scan."""
import argparse
import json

import cv2
import numpy as np
from PIL import Image

import scan_dpi
from scan_feiten import find_hole, hole_outline
from scan_stitch import open_srgb


def fit_circle(points):
    x, y = points.T
    d, e, f = np.linalg.lstsq(np.c_[x, y, np.ones(len(x))], -(x*x + y*y), rcond=None)[0]
    return -d/2, -e/2, np.sqrt(max(0, (d*d + e*e)/4 - f))


def label_boundary(image, px_mm):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p = px_mm
    guesses = cv2.HoughCircles(cv2.GaussianBlur(grey, (5, 5), 1), cv2.HOUGH_GRADIENT,
                              1.1, 14*p, param1=85, param2=30,
                              minRadius=round(33*p), maxRadius=round(62*p))
    if guesses is None:
        return None
    theta = np.arange(360, dtype=np.float32)[:, None] * (2*np.pi/360)
    best = None
    h, w = grey.shape
    for cx, cy, r in guesses[0]:
        reach = 2*p
        if min(cx, cy, w-1-cx, h-1-cy) < r + reach:
            continue
        radius = r + np.linspace(-reach, reach, 31)[None, :]
        u = (cx + np.cos(theta)*radius).astype(np.float32)
        v = (cy + np.sin(theta)*radius).astype(np.float32)
        samples = cv2.remap(image, u, v, cv2.INTER_LINEAR).astype(np.float32)
        smooth = cv2.blur(samples, (3, 5))
        delta = np.linalg.norm(smooth[:, 4:] - smooth[:, :-4], axis=2)
        at = delta.argmax(axis=1)
        response = delta[np.arange(360), at]
        x = u[np.arange(360), at+2]
        y = v[np.arange(360), at+2]
        points = np.c_[x, y]
        valid = response >= 14
        if valid.sum() < 190:
            continue
        for _ in range(4):
            xx, yy, rr = fit_circle(points[valid])
            new = (response >= 14) & (np.abs(np.hypot(x-xx, y-yy)-rr) <= .4*p)
            if new.sum() < 190 or np.array_equal(valid, new):
                break
            valid = new
        coverage = valid.reshape(12, 30).mean(axis=1)
        if coverage.min() < .25:
            continue
        strength = float(np.median(response[valid]))
        score = float(valid.mean())*min(strength, 55)
        if best is None or score > best[0]:
            best = (score, xx, yy, rr, float(valid.mean()))
    return best


def outer_material_boundary(image, cx, cy, radius, px_mm):
    """A print line returns to paper; a physical edge changes the material."""
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)[:, None]
    reach = min(radius + 8*px_mm, min(cx, cy, image.shape[1]-1-cx, image.shape[0]-1-cy))
    distances = np.arange(radius - .5*px_mm, reach, max(.5, .12*px_mm))
    if len(distances) < 25:
        return radius
    x = (cx + np.cos(angles)*distances).astype(np.float32)
    y = (cy + np.sin(angles)*distances).astype(np.float32)
    samples = cv2.remap(image, x, y, cv2.INTER_LINEAR).astype(np.float32)
    span = max(3, round(.85*px_mm / (distances[1]-distances[0])))
    best = (0., radius)
    passing, loose = [], []
    for i in range(span, len(distances)-2*span):
        if distances[i] < radius + .6*px_mm:
            continue
        inside = np.median(samples[:, i-span:i], axis=1)
        outside = np.median(samples[:, i:i+span], axis=1)
        beyond = np.median(samples[:, i+span:i+2*span], axis=1)
        change = np.linalg.norm(outside-inside, axis=1)
        stable = np.linalg.norm(beyond-outside, axis=1)
        confirmed = (change >= 18) & (stable <= .65*change)
        coverage = confirmed.reshape(12,30).mean(axis=1)
        if confirmed.mean() < .72 or np.count_nonzero(coverage >= .4) < 10:
            if (confirmed.mean() >= .45 and np.count_nonzero(coverage >= .4) >= 6
                    and 97 <= 2 * distances[i] / px_mm <= 103):
                loose.append((float(np.median(change[confirmed]) * confirmed.mean()), float(distances[i]), i))
            continue
        score = float(np.median(change[confirmed]) * confirmed.mean())
        passing.append((score, float(distances[i])))
        if score > best[0]:
            best = score, float(distances[i])
    # Sleeve Transparant: een 12"-label is ±100 mm (gemeten 99,2-99,8). Een gedrukte band binnen het label (GF magenta A:
    # randtekst op een andere tint, 93,5 mm) kan sterker zijn dan de echte rand (100,2 mm). Ligt de gekozen rand onder
    # 97 mm en is er tussen 97 en 103 mm ook een bevestigde materiaalrand, dan is dat de rand van het label.
    if 2 * best[1] / px_mm < 97:
        outer = [c for c in passing if 97 <= 2 * c[1] / px_mm <= 103]
        if outer:
            best = max(outer)
        elif loose:
            # Helder vinyl (TA clear A: 93 -> 100 mm): achter de echte rand is het vinyl onrustig, de rand haalt de strenge
            # dekking niet. Telt alleen als de band tot die rand duidelijk ander materiaal is dan wat erbuiten ligt (dus
            # nog label, geen strook vinyl voor een ring in het vinyl).
            score, dist, i = max(loose)
            first = int(np.searchsorted(distances, radius + .6*px_mm)); last = i - span
            band = np.median(samples[:, first:last].reshape(-1, 3), axis=0) if last > first else None
            beyond_ = np.median(samples[:, i + span:i + 3*span].reshape(-1, 3), axis=0) if i + 3*span <= samples.shape[1] else None
            if band is not None and beyond_ is not None and np.linalg.norm(band - beyond_) >= 60:
                best = score, dist
    if 2 * best[1] / px_mm < 97:
        # Sleeve Transparant: wit label op wit/licht vinyl (Trails of Fails 2022, Deena & Jim wit: label L* 216, vinyl 239):
        # de materiaalrand is zwak (verschil < 18) en de gekozen rand is een gedrukte ring (92 mm). Feit: tussen 97 en 103 mm
        # een rondom doorlopende, blijvende overgang (zelfde eisen, drempel 8) en de band tot daar is nog label (lijkt op
        # wat binnen de ring ligt, niet op wat erbuiten ligt).
        # (de scan kan net na de labelrand ophouden: Trails of Fails Scan 13 reikt 50,7 mm; daar de rand van de scan herhalen)
        distances = np.arange(radius - .5*px_mm, 53.5 * px_mm, max(.5, .12*px_mm))
        x = (cx + np.cos(angles)*distances).astype(np.float32); y = (cy + np.sin(angles)*distances).astype(np.float32)
        samples = cv2.remap(image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        lo_i = int(np.searchsorted(distances, 48.5 * px_mm)); hi_i = int(np.searchsorted(distances, 51.5 * px_mm))
        weak = []
        for i in range(max(span, lo_i), min(len(distances) - 2*span, hi_i)):
            inside = np.median(samples[:, i-span:i], axis=1); outside = np.median(samples[:, i:i+span], axis=1)
            beyond = np.median(samples[:, i+span:i+2*span], axis=1)
            change = np.linalg.norm(outside-inside, axis=1); stable = np.linalg.norm(beyond-outside, axis=1)
            confirmed = (change >= 8) & (stable <= .65*change)
            if confirmed.mean() >= .7 and np.count_nonzero(confirmed.reshape(12, 30).mean(axis=1) >= .4) >= 10:
                weak.append((float(np.median(change[confirmed]) * confirmed.mean()), float(distances[i]), i))
        if weak:
            score, dist, i = max(weak)
            ring_in = np.median(samples[:, :max(1, span)].reshape(-1, 3), axis=0)          # net binnen de gedrukte ring
            first = int(np.searchsorted(distances, radius + 1.0*px_mm))
            band = np.median(samples[:, first:max(first + 1, i - span)].reshape(-1, 3), axis=0)
            out_ = np.median(samples[:, i + span:i + 2*span].reshape(-1, 3), axis=0)
            if np.linalg.norm(band - out_) > 1.2 * np.linalg.norm(band - ring_in) or np.linalg.norm(band - out_) >= 12:
                best = score, dist
    return best[1]


def median_edge(image, cx, cy, px_mm, lo=45., hi=53.):
    """Sleeve Transparant: labelrand als de mediaankleur per straal (720 richtingen) tussen lo en hi mm het sterkst
    verspringt. De mediaan negeert opdruk die over de rand loopt (Trails of Fails 2022 kant B: een foto over plaat en label,
    het label is alleen een iets andere tint). Geeft de straal in px, of None als de sprong te klein is."""
    ang = np.linspace(0, 2*np.pi, 720, endpoint=False)[:, None]
    rr = np.arange(lo, hi, .25)[None, :] * px_mm
    x = (cx + np.cos(ang) * rr).astype(np.float32); y = (cy + np.sin(ang) * rr).astype(np.float32)
    lab = cv2.cvtColor(cv2.GaussianBlur(image, (0, 0), max(1., .3 * px_mm)), cv2.COLOR_BGR2LAB).astype(np.float32)
    smp = cv2.remap(lab, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    med = np.median(smp, axis=0)                                   # per straal
    k = 4                                                           # 1 mm
    step = np.array([np.linalg.norm(med[i + k] - med[i - k]) for i in range(k, len(med) - k)])
    if not len(step):
        return None
    i = int(np.argmax(step))
    if step[i] < 5:
        return None
    return float(rr[0, i + k] - .25 * px_mm)


def hole_in_scan(image, px_mm):
    """Sleeve Transparant: het middengat (6,5-8,5 mm rond) op een scan, waar ook: kandidaten met Hough, de sterkste
    volgens find_hole. Geeft (x, y) in px of None."""
    f = 4.0 / px_mm
    g = cv2.medianBlur(cv2.cvtColor(cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=40, param1=80, param2=15, minRadius=12, maxRadius=17)
    best = None
    for x, y, _ in (c[0][:12] if c is not None else []):
        X, Y = x / f, y / f; k = int(10 * px_mm)
        x0, y0 = int(max(0, X - k)), int(max(0, Y - k))
        sub = image[y0:int(Y + k), x0:int(X + k)]
        if sub.shape[0] < k or sub.shape[1] < k:
            continue
        hx, hy, st = find_hole(sub, 3.6 * px_mm, 3 * px_mm)
        if best is None or st > best[0]:
            best = (st, x0 + hx, y0 + hy)
    return best[1:] if best and best[0] >= .5 else None


def persistent_material_boundary(image, cx, cy, radius, px_mm):
    # Keep already-confirmed masks exactly unchanged. If the first persistent
    # colour departure follows printing, try the measured edge near its robust
    # geometric consensus, with the same coverage and residual requirements.
    result = _persistent_material_boundary(image, cx, cy, radius, px_mm)
    if result is None:
        result = _persistent_material_boundary(image, cx, cy, radius, px_mm, refine=True)
    return result


def _persistent_material_boundary(image, cx, cy, radius, px_mm, refine=False):
    """Locate a lasting departure from local paper, not a line returning to it.

    Search relative to the detected ring; never assume a particular LP-label
    diameter. Require evidence beyond the change and a coherent closed ellipse.
    Used only by the explicit enlargement retouch; ordinary cuts stay unchanged.
    """
    p = float(px_mm); step = max(.5, .05*p)
    angles = np.linspace(0, 2*np.pi, 720, endpoint=False)[:, None]
    distances = radius + np.arange(-2*p, 8*p, step)
    x = (cx+np.cos(angles)*distances[None]).astype(np.float32)
    y = (cy+np.sin(angles)*distances[None]).astype(np.float32)
    valid = (x >= 1) & (y >= 1) & (x < image.shape[1]-2) & (y < image.shape[0]-2)
    samples = cv2.remap(image, x, y, cv2.INTER_LINEAR).astype(np.float32)
    samples = cv2.GaussianBlur(samples, (0, 0), max(1., .12*p/step), .6)
    inside = distances < radius-1.1*p
    paper = np.median(samples[:, inside], axis=1)
    noise = np.median(np.linalg.norm(samples[:, inside]-paper[:, None], axis=2), axis=1)
    threshold = np.maximum(8., 3*noise)
    unlike = (np.linalg.norm(samples-paper[:, None], axis=2) > threshold[:, None]) & valid
    span = max(3, int(round(.8*p/step)))
    counts = np.pad(np.cumsum(unlike.astype(np.int32), axis=1), ((0,0),(1,0)))
    ok = np.zeros(unlike.shape, bool)
    tail_unlike = np.cumsum(unlike[:, ::-1].astype(np.int32), axis=1)[:, ::-1]
    tail_valid = np.cumsum(valid[:, ::-1].astype(np.int32), axis=1)[:, ::-1]
    stop = unlike.shape[1]-2*span
    for i in range(stop):
        # Ink rings return to paper; a material change persists over BOTH bands.
        ok[:, i] = ((counts[:,i+span]-counts[:,i]) >= .95*span) & ((counts[:,i+2*span]-counts[:,i+span]) >= .95*span)
        ok[:, i] &= valid[:, i:i+2*span].all(1)
        # Even a broad printed band returns to paper farther out.
        ok[:, i] &= tail_unlike[:, i] >= .95*np.maximum(tail_valid[:, i], 1)
    supported = ok.any(1)
    if supported.mean() < .8:
        return None
    index = ok.argmax(1)
    points = np.c_[x[np.arange(720),index],y[np.arange(720),index]].astype(np.float32)
    keep = supported.copy()
    for _ in range(5):
        (ex,ey),(da,db),deg = cv2.fitEllipse(points[keep])
        theta = np.deg2rad(deg); d = points-[ex,ey]
        error = np.abs(np.hypot((d[:,0]*np.cos(theta)+d[:,1]*np.sin(theta))/(da/2),
                               (-d[:,0]*np.sin(theta)+d[:,1]*np.cos(theta))/(db/2))-1)*min(da,db)/2
        new = supported & (error < .35*p)
        if new.mean() < .8:
            return None
        if np.array_equal(new,keep):
            break
        keep = new
    if refine:
        # The first persistent departure can be ink close to the paper edge.
        # Locate an actual colour step beside the consensus ellipse instead.
        # A candidate must still remain unlike the inner material throughout
        # the measured outside, so a printed ring returning to paper is refused.
        jump = max(1, round(.10*p/step))
        gradient = np.zeros(samples.shape[:2], np.float32)
        gradient[:, jump:-jump] = np.linalg.norm(samples[:, 2*jump:]-samples[:, :-2*jump], axis=2)
        # A real boundary moves towards the observed outside material. An ink
        # stroke ending inside paper can remain unlike the inner printed colour,
        # but moves away from that outside material and must not seed the mask.
        outside = np.median(samples[:, -span:], axis=1)
        approach = np.zeros(samples.shape[:2], np.float32)
        approach[:, jump:-jump] = (np.linalg.norm(samples[:, :-2*jump]-outside[:, None], axis=2)
                                  - np.linalg.norm(samples[:, 2*jump:]-outside[:, None], axis=2))
        persistent = np.zeros_like(ok)
        for shift in range(jump+1):
            persistent[:, :ok.shape[1]-shift] |= ok[:, shift:]
        ids = np.arange(len(angles))
        for _ in range(5):
            theta = np.deg2rad(deg)
            dx, dy = x-ex, y-ey
            residual = np.abs(np.hypot((dx*np.cos(theta)+dy*np.sin(theta))/(da/2),
                                      (-dx*np.sin(theta)+dy*np.cos(theta))/(db/2))-1)*min(da,db)/2
            candidates = persistent & (residual <= 2*.35*p) & (gradient >= 14) & (approach >= 8)
            index = np.where(candidates, gradient, -1).argmax(1)
            available = candidates.any(1)
            if available.mean() < .8:
                return None
            points = np.c_[x[ids,index], y[ids,index]].astype(np.float32)
            (ex,ey),(da,db),deg = cv2.fitEllipse(points[available])
            theta = np.deg2rad(deg); diff = points-[ex,ey]
            error = np.abs(np.hypot((diff[:,0]*np.cos(theta)+diff[:,1]*np.sin(theta))/(da/2),
                                   (-diff[:,0]*np.sin(theta)+diff[:,1]*np.cos(theta))/(db/2))-1)*min(da,db)/2
            keep = available & (error < .35*p)
        if keep.mean() < .8:
            return None
    sectors = keep.reshape(12,60).mean(1)
    if sectors.min() < .5 or max(da,db)/min(da,db) > 1.03:
        return None
    return float(ex),float(ey),float((da+db)/4),float(keep.mean())


def midpoint_edge_check(image, lx, ly, da, db, q, px_mm):
    """Sleeve Transparant: tweede, onafhankelijke meting van de papierrand. De rand per richting als het midden van de
    overgang (eerste punt van binnen dat dichter bij het vinyl ligt dan bij het label), niet als de sterkste gradient.
    Feit (Together Alone goud B 0010, Graduated Fool zilver B 0015): op ondoorzichtig metallic vinyl toont de rand per
    richting een ander kenmerk (lichte papiervezel, onscherpe rand, dunne schaduwlijn); de sterkste gradient springt
    daartussen tot 0,3 mm en de ellips haalde maar 74-79% van de richtingen, terwijl de rand rondom zichtbaar is.
    Geeft (dekking, (x, y, da, db)) of None."""
    r = (da + db) / 4
    angles = np.linspace(0, 2*np.pi, 720, endpoint=False)[:, None]
    off = np.arange(-1.25*px_mm, 1.25*px_mm, .5)
    x = (lx + np.cos(angles)*(r + off[None, :])).astype(np.float32)
    y = (ly + np.sin(angles)*(r + off[None, :])).astype(np.float32)
    colours = cv2.blur(cv2.remap(image, x, y, cv2.INTER_LINEAR).astype(np.float32), (5, 3))
    n = len(off)
    inner = np.median(colours[:, :n//4], axis=1); outer = np.median(colours[:, 3*n//4:], axis=1)
    closer = np.linalg.norm(colours - outer[:, None], axis=2) < np.linalg.norm(colours - inner[:, None], axis=2)
    j = closer.argmax(axis=1); ids = np.arange(len(j))
    ok = (np.linalg.norm(outer - inner, axis=1) >= 25) & closer.any(axis=1)
    if ok.sum() < 360:
        return None
    border = np.c_[x[ids, j], y[ids, j]].astype(np.float32); keep = ok
    for _ in range(3):
        (mx, my), (ma, mb), angle = cv2.fitEllipse(border[keep])
        t = np.deg2rad(angle); d = border - [mx, my]
        dist = np.hypot((d[:, 0]*np.cos(t)+d[:, 1]*np.sin(t))/(ma/2), (-d[:, 0]*np.sin(t)+d[:, 1]*np.cos(t))/(mb/2))
        new = ok & (np.abs(dist-1)*min(ma, mb)/2 <= .18*px_mm)
        if new.sum() < 360 or np.array_equal(new, keep):
            break
        keep = new
    return float(keep.mean()), (mx, my, ma, mb)


def extract(path, dpi, material_only=False):
    with open_srgb(path) as a:
        im = cv2.cvtColor(np.asarray(a), cv2.COLOR_RGB2BGR)
    preview_scale = min(1., 1800/max(im.shape[:2]))
    px_mm = dpi/25.4
    preview = cv2.resize(im, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
    result = label_boundary(preview, px_mm*preview_scale)
    if result is None:
        result = closeup_label(preview, px_mm*preview_scale)   # Sleeve Transparant: losse labelscan (Fake It 2021, Scan 9)
    if result is None:
        # Sleeve Transparant: label in het midden van een samengevoegde kant (STTOF 2025 kant B, zwart label op rood vinyl):
        # de labelcirkel (80-106 mm) met Hough, het middelpunt binnen 15 mm van het midden van het beeld
        alt = closeup_label(preview, px_mm*preview_scale, any_size=True)
        ph, pw = preview.shape[:2]
        if alt is not None and np.hypot(alt[1] - pw/2, alt[2] - ph/2) <= 15*px_mm*preview_scale:
            result = alt
    elif 2*result[3]/(px_mm*preview_scale) < 85:
        # Sleeve Transparant: een lp-label is 80-106 mm; een kleinere cirkel is een ring in de opdruk (Fake It 2021, Scan 11:
        # 71 mm, het label onderaan afgesneden). Dan de labelcirkel zelf (80-106 mm) als die er is.
        alt = closeup_label(preview, px_mm*preview_scale, any_size=True)
        if alt is not None:
            result = alt
    if result is None:
        return None, {'verdict': 'TWIJFEL', 'why': ['the printed label edge could not be measured around the scan']}
    _, cx, cy, r, coverage = result
    cx, cy, r = (v/preview_scale for v in (cx, cy, r))
    # Sleeve Transparant: het label ligt rond het middengat. Is er geen gat in het midden van de gevonden cirkel, maar wel
    # elders op de scan, dan was de cirkel iets anders (een rond stuk opdruk: Trails of Fails 2022 kant B, 94 mm tussen
    # takken). Dan het label rond het gat meten.
    k_ = int(6 * px_mm)
    sub_ = im[int(max(0, cy - k_)):int(cy + k_), int(max(0, cx - k_)):int(cx + k_)]
    st_ = find_hole(sub_, 3.6*px_mm, 1.7*px_mm)[2] if sub_.size else 0
    hole_moved = False
    if st_ < .3 or 2 * r / px_mm < 97:                     # geen gat in het midden, of geen lp-labelmaat
        hole_ = hole_in_scan(im, px_mm)
        if hole_ is not None and np.hypot(hole_[0] - cx, hole_[1] - cy) > 3 * px_mm:
            e_ = median_edge(im, hole_[0], hole_[1], px_mm)
            if e_ is not None and 97 <= 2 * e_ / px_mm <= 103:
                cx, cy, r = hole_[0], hole_[1], e_
                hole_moved = True
    detected_radius = r
    if not hole_moved:
        r = outer_material_boundary(im, cx, cy, r, px_mm)
    if 2 * r / px_mm < 97:
        e_ = median_edge(im, cx, cy, px_mm)                  # laatste kans: de mediaantint per straal
        if e_ is not None and 97 <= 2 * e_ / px_mm <= 103:
            r = e_
    if material_only:
        material = persistent_material_boundary(im, cx, cy, r, px_mm)
        if material is None and abs(r-detected_radius) > .5*px_mm:
            material = persistent_material_boundary(im, cx, cy, detected_radius, px_mm)
        if material is None:
            return None, {'verdict': 'TWIJFEL', 'why': ['native paper boundary not independently confirmed outside the print ring']}
        cx, cy, r, coverage = material
        detected_radius = min(detected_radius, r-.01*px_mm)
    left, top, right, bottom = (int(np.floor(cx-r)), int(np.floor(cy-r)),
                                int(np.ceil(cx+r+1)), int(np.ceil(cy+r+1)))
    crop = im[top:bottom, left:right].copy()
    centre_radius = 3.6*px_mm
    hx, hy, strength = find_hole(crop, centre_radius, 1.7*px_mm)
    hole_note = None
    given = transparent_hole(path, left, top, crop.shape, px_mm)
    if given is not None:
        # Sleeve Transparant: het gat is in de bron al uitgesneden (los label uit de samengevoegde kant, Hotel New York
        # 2022): dat gat, niet opnieuw zoeken (de zoeker kwam er een paar mm naast)
        hx, hy, given_r = given; strength = 1.0
        centre_radius = max(centre_radius, given_r + .5)          # het hele uitgesneden gat (geen wit randje)
        points = np.zeros((0, 2))
    elif strength < .3:
        # Sleeve Transparant: gat niet te meten (doorzichtig vinyl achter het gat, Fake It 2021 Scan 9-10): het
        # standaardgat (7,2 mm) in het midden van het label, met twijfel - zoals bij de kanten
        hx, hy = crop.shape[1] / 2, crop.shape[0] / 2
        hole_note = 'centre hole: edge not clearly measured; cut at the standard size in the centre of the label'
        points = np.zeros((0, 2))
    if given is None and strength >= .3:
        # Keep the visible flap on an irregular hole: fit the supported opening,
        # rejecting points which depart from its circular portion.
        points = hole_outline(crop, hx, hy, centre_radius, top=1.3)
    circle = (hx, hy, centre_radius)
    for _ in range(4 if len(points) else 0):
        x, y, r0 = circle
        dist = np.abs(np.hypot(points[:, 0]-x, points[:, 1]-y)-r0)
        mask = dist < .32*px_mm
        if mask.sum() < len(points)*.5:
            break
        circle = fit_circle(points[mask])
    hx, hy, hole = circle
    # Re-measure the edge on the original pixels. Scanner stretch can make
    # a circle a slight ellipse; a preview-sized circle can include a light fringe.
    angles = np.linspace(0, 2*np.pi, 720, endpoint=False)[:, None]
    reach = (1.25 if r > detected_radius else .65) * px_mm
    radii = r + np.arange(-reach, reach, .5)[None, :]
    x = (cx + np.cos(angles)*radii).astype(np.float32)
    y = (cy + np.sin(angles)*radii).astype(np.float32)
    colours = cv2.remap(im, x, y, cv2.INTER_LINEAR).astype(np.float32)
    blurred = cv2.blur(colours, (5, 3))
    gradient = np.linalg.norm(blurred[:, 5:] - blurred[:, :-5], axis=2)
    j = gradient.argmax(axis=1)
    ids = np.arange(len(j))
    border = np.c_[x[ids, j+2], y[ids, j+2]].astype(np.float32)
    keep = gradient[ids, j] >= 14
    if keep.sum() >= 360:
        for _ in range(3):
            (lx, ly), (da, db), angle = cv2.fitEllipse(border[keep])
            q = np.deg2rad(angle)
            diff = border - [lx, ly]
            dist = np.hypot((diff[:, 0]*np.cos(q)+diff[:, 1]*np.sin(q))/(da/2),
                            (-diff[:, 0]*np.sin(q)+diff[:, 1]*np.cos(q))/(db/2))
            fitted = (gradient[ids, j] >= 14) & (np.abs(dist-1)*min(da, db)/2 <= .18*px_mm)
            if fitted.sum() < 360 or np.array_equal(fitted, keep):
                break
            keep = fitted
    else:
        (lx, ly), (da, db), q = (cx, cy), (2*r, 2*r), 0.
    if r > detected_radius and keep.sum() >= 360:
        # Sleeve Transparant: de rand ligt buiten de eerste (gedrukte) ring; hoe goed de rand rondom te zien is, meet de
        # echte rand (deel van de 720 richtingen dat op de gefitte ellips ligt), niet die eerste ring.
        coverage = float(keep.mean())
    confirmed_by = None
    if material_only and coverage < .8 and keep.sum() >= 360:
        # rand niet rondom op de sterkste gradient: een tweede meting (midden van de overgang) moet hem rondom zien
        # én dezelfde ellips geven (diameter op 0,2 mm, middelpunt op 0,15 mm). De vorm blijft die van de eerste meting.
        second = midpoint_edge_check(im, lx, ly, da, db, q, px_mm)
        if second is not None:
            cov2, (mx, my, ma, mb) = second
            if (cov2 >= .8 and abs((ma+mb)-(da+db))/2 <= .2*px_mm and np.hypot(mx-lx, my-ly) <= .15*px_mm):
                coverage = cov2; confirmed_by = f'midpoint of the transition ({cov2:.2f} of the directions; gradient edge {float(keep.mean()):.2f})'
    if r > detected_radius:
        # Use the full bounds of the refined ellipse. An initial circle can
        # cut off pixels before the native-resolution mask is even applied.
        old_left, old_top = left, top
        ex = np.hypot(da/2*np.cos(q), db/2*np.sin(q))
        ey = np.hypot(da/2*np.sin(q), db/2*np.cos(q))
        left, right = max(0, int(np.floor(lx-ex-1))), min(im.shape[1], int(np.ceil(lx+ex+2)))
        top, bottom = max(0, int(np.floor(ly-ey-1))), min(im.shape[0], int(np.ceil(ly+ey+2)))
        crop = im[top:bottom, left:right].copy()
        hx += old_left-left
        hy += old_top-top
    yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
    dx, dy = xx-(lx-left), yy-(ly-top)
    edge = np.hypot((dx*np.cos(q)+dy*np.sin(q))/(da/2),
                    (-dx*np.sin(q)+dy*np.cos(q))/(db/2))
    outer = np.clip((1-edge)*min(da, db)/2 - .65, 0, 1)
    inside = np.clip(np.hypot(xx-hx, yy-hy)-hole + .5, 0, 1)
    alpha = np.rint(255*np.minimum(outer, inside)).astype(np.uint8)
    out = np.dstack((crop, alpha))
    out[alpha == 0, :3] = 255
    return out, {'verdict': 'GOED' if coverage >= .8 and not hole_note else 'TWIJFEL',
                 'why': ([] if coverage >= .8 else ['the printed label edge is weak in some directions']) + ([hole_note] if hole_note else []),
                 'edge_coverage': round(coverage, 3), 'diameter_mm': round((da+db)/2/px_mm, 2),
                 **({'edge_confirmed_by': confirmed_by} if confirmed_by else {}),
                 'hole_mm': round(2*hole/px_mm, 2),
                 'source_bounds': [left, top, left+crop.shape[1], top+crop.shape[0]],
                 'source_label_ellipse': [float(lx), float(ly), float(da), float(db), float(q)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scan')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    dpi = scan_dpi.read(args.scan)
    result = {'verdict': 'TWIJFEL', 'why': ['label scan has no reliable dpi']}
    if dpi:
        crop, result = extract(args.scan, dpi)
        if crop is not None:
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA)).save(args.out, dpi=(dpi, dpi))
    print(json.dumps(result))


def closeup_label(image, px_mm, any_size=False):
    """Sleeve Transparant: label op een losse labelscan (het beeld zelf hooguit 140 mm). Terugval als label_boundary
    niets vindt (FBOW 2023 rood, Scan 18: zwart label op rood vinyl): een cirkel van 80-106 mm (Hough, op 4 px/mm, in
    grijs en in verzadiging), het middelpunt in het beeld. Geeft (score, x, y, r, dekking) in pixels van image, of None."""
    h, w = image.shape[:2]
    if not px_mm or (max(h, w) / px_mm > 140 and not any_size):
        return None
    f = 4.0 / px_mm
    im = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    g = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 5)
    sat = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1], 5)
    for img_, p2 in ((g, 40), (sat, 40), (g, 30)):
        c = cv2.HoughCircles(img_, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=80, param2=p2, minRadius=160, maxRadius=212)
        if c is not None:
            x, y, r = (float(v) / f for v in c[0][0])
            if 0 < x < w and 0 < y < h:
                return 50.0, x, y, r, .9
    return None


def transparent_hole(path, left, top, shape, px_mm):
    """Sleeve Transparant: een al uitgesneden gat in de bron (alfa 0, ±5-9 mm rond, binnen 15 mm van het midden van de
    uitsnede). Geeft het middelpunt in uitsnede-pixels, of None."""
    try:
        with Image.open(path) as im:
            if im.mode not in ('RGBA', 'LA'):
                return None
            a = np.asarray(im.getchannel('A'))
    except Exception:
        return None
    sub = a[max(0, top):top + shape[0], max(0, left):left + shape[1]]
    h, w = sub.shape
    zone = int(15 * px_mm)
    cy, cx = h // 2, w // 2
    win = (sub[max(0, cy - zone):cy + zone, max(0, cx - zone):cx + zone] == 0).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(win)
    best = None
    for i in range(1, n):
        d = 2 * np.sqrt(st[i][4] / np.pi) / px_mm
        if 5 <= d <= 9 and (best is None or st[i][4] > st[best][4]):
            best = i
    if best is None:
        return None
    return cen[best][0] + max(0, cx - zone), cen[best][1] + max(0, cy - zone), np.sqrt(st[best][4] / np.pi)


def complete_label_scan(path):
    """Recognize a full label with a broad, independently scanned surround."""
    dpi = scan_dpi.read(path)
    if not dpi:
        return False
    with Image.open(path) as image:
        width = image.width
        image.thumbnail((1600, 1600))
        scale = image.width / width
        preview = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    result = label_boundary(preview, dpi / 25.4 * scale)
    if result is None:
        result = closeup_label(preview, dpi / 25.4 * scale)      # losse labelscan (hooguit 140 mm)
    if result is None:
        return False
    _, x, y, radius, coverage = result
    margin = min(x, y, preview.shape[1]-x, preview.shape[0]-y) - radius
    # Sleeve Transparant: een losse labelscan (de scan zelf hooguit 140 mm) is bijgesneden tot het label; daar is 1,5 mm
    # vinyl rondom genoeg (FBOW 2009: 3-9 mm; FBOW 2023 rood: 2 mm). Een plaatscan met toevallig het hele label erop houdt de eis van 12 mm.
    need = 1.5 if max(preview.shape[:2]) / (dpi / 25.4 * scale) <= 140 else 12
    return (coverage >= .85 and margin >= need * dpi / 25.4 * scale
            and .25 < x/preview.shape[1] < .75 and .25 < y/preview.shape[0] < .75)


if __name__ == '__main__':
    main()
