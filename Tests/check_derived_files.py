"""Een samengevoegd of gemerkt bestand is geen scan (0.5.52). For Bitter Or Worse 2009, 2022_scans: 'Scan 8 Panorama.jpg'
(PTGui Pro 12.15, watermerk www.ptgui.com) en 'Scan 10_stitch.jpg' (Microsoft ICE) kwamen in de stapel van kant A, en kant A
kreeg GOED. Het EXIF-veld Software noemt het programma; een scanner noemt zijn eigen programma of niets, een bewerking in
Photoshop is geen samenvoeging."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
from PIL import Image
import scan_dpi
d = tempfile.mkdtemp()
def make(name, software):
    p = os.path.join(d, name); im = Image.new('RGB', (64, 64), (200, 120, 40)); ex = Image.Exif()
    if software:
        ex[305] = software
    im.save(p, exif=ex, dpi=(300, 300)); return p
derived = {'PTGui Pro 12.15 (www.ptgui.com)', 'Microsoft ICE v2.0.3.0', 'Panorama Stitcher Mini 1.11.2', 'iWatermark Pro'}
plain = {'Apple Image Capture', 'Adobe Photoshop 21.2 (Macintosh)', 'EPSON Scan', None}
for i, sw in enumerate(sorted(derived)):
    assert scan_dpi.derived_by(make(f'd{i}.jpg', sw)) == sw, sw
for i, sw in enumerate(sorted(plain, key=str)):
    assert scan_dpi.derived_by(make(f'p{i}.jpg', sw)) is None, sw
print('samengevoegde of gemerkte bestanden (PTGui, Microsoft ICE, Panorama Stitcher, iWatermark) herkend; scans en Photoshop niet')
