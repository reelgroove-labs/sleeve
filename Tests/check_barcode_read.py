"""Een mislukte barcodelezing is geen 'geen barcode' (0.5.53). Graduated Fool 2022 groen 0009 gaf in één run stil niets
terug en de achterkant bleef een open vak; later las dezelfde scan de barcode in 2,8 s. Nu: één keer opnieuw, en lukt
het dan nog niet, dan staat de reden erbij."""
import os, sys, subprocess, json, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Resources', 'Tools'))
import scan_vakken as V
calls = []
def fake(results):
    it = iter(results)
    def run(cmd, **kw):
        calls.append(cmd); r = next(it)
        if isinstance(r, BaseException):
            raise r
        return types.SimpleNamespace(returncode=r[0], stdout=r[1], stderr=r[2])
    return run
ok = (0, json.dumps({'barcodes': ['0602445294169']}), '')
# 1. eerste lezing loopt vast, tweede lukt: barcode gevonden, geen probleem
V.subprocess.run = fake([subprocess.TimeoutExpired('ocr', 120), ok])
assert V.barcodes('/x/0009.tif') == ['0602445294169'] and '/x/0009.tif' not in V.BARCODE_PROBLEMS
# 2. twee keer te lang: leeg, met de reden
V.subprocess.run = fake([subprocess.TimeoutExpired('ocr', 120)] * 2)
assert V.barcodes('/x/0009.tif') == [] and 'longer than 120 s' in V.BARCODE_PROBLEMS['/x/0009.tif']
# 3. fout van de tekstherkenning: de foutregel komt in de reden
V.subprocess.run = fake([(1, '', 'Error: file not readable'), (1, '', 'Error: file not readable')])
assert V.barcodes('/x/0008.tif') == [] and 'file not readable' in V.BARCODE_PROBLEMS['/x/0008.tif']
# 4. gelezen maar geen barcode op de scan: leeg en GEEN probleem
V.subprocess.run = fake([(0, json.dumps({'barcodes': []}), '')])
assert V.barcodes('/x/0007.tif') == [] and '/x/0007.tif' not in V.BARCODE_PROBLEMS
print('barcode: na een mislukte lezing één keer opnieuw; blijft het mislukken, dan de reden; geen barcode is geen fout')
