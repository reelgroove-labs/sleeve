"""Kleurprofiel in de uitvoer. Feit: de bronscans zijn sRGB en Sleeve rekent in sRGB (scan_cutout.to_srgb), maar
OpenCV en PIL schrijven geen profiel mee. Voor een archief hoort het profiel in het bestand (gebruiker, 25 nov 2026).
Het iCCP-blok wordt direct na IHDR ingevoegd; de beelddata blijft byte voor byte gelijk (geen nieuwe compressie)."""
import struct, zlib, io

_PROFILE = None


def srgb_profile():
    global _PROFILE
    if _PROFILE is None:
        from PIL import ImageCms
        _PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    return _PROFILE


def _chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)


def tag_srgb(path):
    """Voegt een sRGB-profiel toe aan een PNG zonder iCCP- of sRGB-blok. Geeft True als het bestand is aangepast."""
    if not str(path).lower().endswith('.png'):
        return False
    with open(path, 'rb') as fh:
        data = fh.read()
    if data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
        return False
    pos = 8
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos + 4])[0]; kind = data[pos + 4:pos + 8]
        if kind in (b'iCCP', b'sRGB'):
            return False
        if kind in (b'IDAT', b'IEND'):
            break
        pos += 12 + length
    end_ihdr = 8 + 12 + struct.unpack('>I', data[8:12])[0]
    iccp = _chunk(b'iCCP', b'sRGB\x00\x00' + zlib.compress(srgb_profile(), 9))
    tmp = str(path) + '.icc-tmp'
    with open(tmp, 'wb') as fh:
        fh.write(data[:end_ihdr] + iccp + data[end_ihdr:])
    import os
    os.replace(tmp, path)
    return True
