"""Resolutie van een scan, op feiten.

Feiten:
  - Een jpeg kan twee resoluties bevatten: in het JFIF-deel en in het EXIF-deel. Scanners
    schrijven de echte resolutie in EXIF; een JFIF-waarde zonder eenheid is alleen een
    verhouding, geen resolutie.
  - EXIF zonder eenheid betekent volgens de EXIF-norm: per inch.
  - Een flatbedscanner is hooguit A3 (±300 x 430 mm). Een scan die bij zijn opgeslagen
    resolutie groter zou zijn, heeft een verkeerde resolutie in het bestand.
"""
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SCANNER_MAX_MM = 450.0


def read(path):
    """Resolutie in dpi uit het bestand, of 0.0 als die er niet (betrouwbaar) in staat."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            exif = im.getexif()
            x, unit = exif.get(282), exif.get(296)
            candidates = []
            if x:
                try:
                    v = float(x)
                    if unit in (None, 2):
                        candidates.append(v)
                    elif unit == 3:
                        candidates.append(v * 2.54)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            jd, ju = im.info.get('jfif_density'), im.info.get('jfif_unit')
            if jd and ju in (1, 2):
                candidates.append(float(jd[0]) * (2.54 if ju == 2 else 1.0))
            d = im.info.get('dpi')
            if d and not candidates:
                candidates.append(float(d[0]))
    except Exception:
        return 0.0
    for v in candidates:
        if 50 < v < 5000 and max(w, h) / v * 25.4 <= SCANNER_MAX_MM:
            # Sleeve Transparant: is de scan op de helft van de opgeslagen resolutie precies de glasplaat (A4, ±216 x 297
            # mm) en op de opgeslagen resolutie niet, dan staat er een verdubbelde resolutie in het bestand (Wen D'r Maar
            # Aan 2023: 5096 x 7016 px 'op 1200 dpi', een kwart plaat; de echte scan is 600 dpi).
            if v >= 600 and is_glass(w, h, v / 2) and not is_glass(w, h, v):
                return v / 2
            return v
    return 0.0


# Sleeve 0.5.52 - feit: een programma dat scans samenvoegt of merkt, schrijft zijn naam in het EXIF-veld Software (For Bitter
# Or Worse 2009, 2022_scans: 'Scan 8 Panorama.jpg' = PTGui Pro 12.15, met watermerk 'www.ptgui.com'; 'Scan 10_stitch.jpg'
# = Microsoft ICE v2.0.3.0). Zo'n bestand is geen scan maar al samengesteld uit scans. Een scanner schrijft zijn eigen
# programma (Apple Image Capture, Epson Scan) of niets.
DERIVED_WORDS = ('ptgui', 'microsoft ice', 'image composite editor', 'panorama', 'stitch', 'hugin', 'autopano',
                 'photomerge', 'watermark')


def derived_by(path):
    """De naam van het samenvoeg- of merkprogramma uit EXIF Software, of None als het bestand een gewone scan is."""
    try:
        with Image.open(path) as im:
            software = str(im.getexif().get(305) or '').strip()
    except Exception:
        return None
    return software if software and any(w in software.lower() for w in DERIVED_WORDS) else None


def is_glass(w, h, dpi):
    """Is een scan van w x h pixels op deze resolutie precies de A4-glasplaat (±216 x 297 mm)?"""
    a, b = sorted((w / dpi * 25.4, h / dpi * 25.4))
    return abs(a - 215.8) <= 1.5 and abs(b - 297.0) <= 1.5


def size_px(path):
    with Image.open(path) as im:
        return im.size
