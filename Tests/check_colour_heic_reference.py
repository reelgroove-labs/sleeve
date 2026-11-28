"""Kleur van een iPhone-foto (0.5.59), feit: de kleurstap kan HEIC niet openen ('cannot identify image file', Hotel New York 2022
magenta, IMG_4532.HEIC). Sleeve zet een HEIC-referentie eerst met sips om naar JPEG; een JPEG/PNG blijft zoals hij is."""
import os, sys, subprocess, tempfile, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Resources'))
import studio_bridge as B
work = Path(tempfile.mkdtemp()); png = work / 'foto.png'
Image.fromarray(np.full((60, 80, 3), (200, 40, 120), np.uint8)).save(png)
heic = work / 'IMG_0001.HEIC'
r = subprocess.run(['/usr/bin/sips', '-s', 'format', 'heic', str(png), '--out', str(heic)], capture_output=True)
assert heic.is_file(), r.stderr
assert B.readable_reference(png, work) == png
out = B.readable_reference(heic, work)
assert out.suffix == '.jpg' and out.is_file(), out
with Image.open(out) as im: px = im.convert('RGB').getpixel((40, 30))
assert abs(px[0] - 200) < 12 and abs(px[1] - 40) < 12 and abs(px[2] - 120) < 12, px
print(f'kleurfoto HEIC wordt JPEG ({out.name}, kleur {px}); PNG blijft PNG')
