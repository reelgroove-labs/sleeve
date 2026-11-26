#!/usr/bin/env python3
"""Lichtcorrectie per scannerkolom (transparant vinyl).

Feit: lamp en sensor geven niet in elke kolom van het glas evenveel licht. Op een
doorzichtige plaat schijnt het licht erdoor en zie je dat als smalle verticale
strepen (±1,5-2,5%), steeds op dezelfde kolom van het glas, waar de plaat ook ligt
(gemeten 19 nov: correlatie 0,7-0,98 tussen scans op andere plaatposities).

Meten:  scan_lichtkolom.py meet <profiel.json> <stitch.png> [<stitch.png> ...] --scans <map>
        (stitch = uitvoer van scan_stitch.py met .scans.json ernaast; plaat met label)
        Per deelscan: plaatring 62-142 mm, gedeeld door het eigen ringprofiel (groeven/rand
        weg), mediaan per kolom, alleen het smalle deel (<40 mm). Mediaan over alle scans.
Gebruik: scan_stitch.py ... --lichtkolom <profiel.json>
        Elke deelscan wordt bij het inlezen per kolom gedeeld door (1 + profiel).
        Zonder die optie verandert er niets.
"""
import json, os, sys
import numpy as np, cv2

SMAL_MM = 40.0     # alleen strepen smaller dan dit; breder licht (klep, glasrand) blijft
RING_MM = (62, 142)


def load_profile(path):
    """Profiel uit json: {'scan_size': [w, h], 'columns': [...]} (kolommen over de volle breedte)."""
    data = json.load(open(path))
    cols = np.asarray(data['columns'], np.float32)
    w, h = data['scan_size']
    if len(cols) < 16 or not w or not h:
        raise ValueError('leeg lichtkolomprofiel')
    return {'size': (int(w), int(h)), 'columns': cols}


def gain_for(profile, width, height):
    """Factor per kolom voor een beeld van deze maat; None als het niet bij dit glas hoort."""
    w0, h0 = profile['size']
    if abs(width / w0 - height / h0) > .01 * (width / w0):
        return None                                   # andere verhouding: niet hetzelfde glas/stand
    cols = profile['columns']
    x = (np.arange(width) + .5) * (len(cols) / width) - .5
    # alleen de smalle strepen (±2,5%); grotere waarden zijn de donkere glasrand (links tot +37% op de eerste 8 mm, een
    # meetfout van het profiel): daar niets doen. Anders werd die strook fel en vond Sleeve het label niet meer (20 nov).
    cols = np.where(np.abs(cols) > .045, 0.0, cols).astype(np.float32)
    p = np.interp(x, np.arange(len(cols)), cols)
    return (1.0 / (1.0 + p)).astype(np.float32)


def apply(bgr, profile, rows=None):
    """In place: deel elke kolom door (1 + profiel). rows = (start, stop, volle hoogte) bij banden."""
    h, w = bgr.shape[:2]
    full_h = rows[2] if rows else h
    g = gain_for(profile, w, full_h)
    if g is None:
        return False
    sl = slice(rows[0], rows[1]) if rows else slice(None)
    band = bgr[sl]
    band[:] = np.clip(band.astype(np.float32) * g[None, :, None] + .5, 0, 255).astype(np.uint8)
    return True


def _centre(png, pm):
    im = cv2.imread(png, cv2.IMREAD_REDUCED_COLOR_4)
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 2); q = pm / 4
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.5, 400, param1=80, param2=40, minRadius=int(47 * q), maxRadius=int(53 * q))
    if c is None:
        raise SystemExit(f'geen label gevonden in {png}')
    return c[0][0][0] * 4, c[0][0][1] * 4


def measure(pngs, scandir, q=4):
    profiles = []; size = None
    for png in pngs:
        j = json.load(open(png + '.scans.json')); pm = j['dpi'] / 25.4
        cx, cy = _centre(png, pm)
        for s in j['scans']:
            src = os.path.join(scandir, os.path.basename(s['scan']))
            g = cv2.imread(src, cv2.IMREAD_GRAYSCALE).astype(np.float32); H, W = g.shape
            size = size or (W, H)
            if (W, H) != size:
                continue
            A = cv2.getAffineTransform(np.float32([[0, 0], [W, 0], [W, H]]), np.float32(s['corners'][:3]))
            pc = cv2.invertAffineTransform(A) @ [cx, cy, 1]
            gs = cv2.resize(g, (W // q, H // q), interpolation=cv2.INTER_AREA)
            yy, xx = np.mgrid[:H // q, :W // q]
            r = np.hypot(xx - pc[0] / q, yy - pc[1] / q) / (pm / q); m = (r > RING_MM[0]) & (r < RING_MM[1])
            if m.sum() < 1000:
                continue
            ri = np.clip((r * 2).astype(int), 0, 400); prof = np.zeros(401)
            for i in np.unique(ri[m]):
                prof[i] = np.median(gs[(ri == i) & m])
            col = np.nanmedian(np.where(m, gs / np.maximum(prof[ri], 1), np.nan), 0)
            ok = np.isfinite(col); v = np.where(ok, col, 0); sg = SMAL_MM * pm / q
            sm = cv2.GaussianBlur(v[None], (0, 0), sg)[0] / np.maximum(cv2.GaussianBlur(ok[None].astype(np.float64), (0, 0), sg)[0], 1e-6)
            profiles.append(np.where(ok, col - sm, np.nan))
            print(os.path.basename(src), 'strepen p5/p95 %+.3f %+.3f' % tuple(np.nanpercentile(profiles[-1], [5, 95])))
    if len(profiles) < 4:
        raise SystemExit('te weinig deelscans (minstens 4) voor een betrouwbaar profiel')
    St = np.stack(profiles); P = np.nanmedian(St, 0); P = np.where(np.isfinite(P), P, 0)
    return size, P, len(profiles)


def main():
    if len(sys.argv) < 4 or sys.argv[1] != 'meet' or '--scans' not in sys.argv:
        sys.exit(__doc__)
    i = sys.argv.index('--scans'); scandir = sys.argv[i + 1]; args = sys.argv[2:i] + sys.argv[i + 2:]
    out, pngs = args[0], args[1:]
    size, P, n = measure(pngs, scandir)
    json.dump({'scan_size': list(size), 'columns': [round(float(x), 5) for x in P], 'scans': n,
               'about': 'lichtcorrectie per scannerkolom; factor = 1/(1+kolom)'}, open(out, 'w'))
    print('profiel %s: %d scans, p5/p95 %+.3f %+.3f' % (out, n, *np.percentile(P, [5, 95])))


if __name__ == '__main__':
    main()
