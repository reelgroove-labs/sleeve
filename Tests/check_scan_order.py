"""Scan order for automatic arranging (scan_vakken.py): a running scanner counter is followed across own names and
days; when every named series restarts its own counter, the names give the order (us_new_white_back3, 15 Nov)."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'Resources/Tools'))
import scan_vakken as V
def order(names):
    V.set_order(names);return sorted(names,key=V.number_key)
# running counter, an own word in between (Queen 2021: the label scan belongs between 0017 and 0019)
q=['queen_20261114_0019.tif','queen_sidea_label_20261114_0018.tif','queen_20261114_0017.tif','queen_sideb_label_20261114_0023.tif','queen_20261114_0023.tif']
assert order(q)[:3]==['queen_20261114_0017.tif','queen_sidea_label_20261114_0018.tif','queen_20261114_0019.tif'],order(q)
assert not V.NAME_ORDER
# scanner names and two days in one folder
assert order(['Scan 10.jpeg','Scan 2.jpeg','Scan.jpeg','Scan 1.jpeg'])==['Scan.jpeg','Scan 1.jpeg','Scan 2.jpeg','Scan 10.jpeg']
assert order(['00 12.jpeg','00 1.jpeg','00.jpeg'])==['00.jpeg','00 1.jpeg','00 12.jpeg']
assert order(['IMG_20261114_0003.jpg','IMG_20261112_0012.jpg'])==['IMG_20261112_0012.jpg','IMG_20261114_0003.jpg']
assert order(['Scan 2026-11-14 at 10.32.05.jpg','Scan 2026-11-14 at 09.59.40.jpg'])[0].endswith('09.59.40.jpg')
# a counter per named part: the parts must stay together
b=[f'us_white_{p}_20261114_000{i}.tif' for p in ('voorkant','sideb','sidea') for i in (4,3,2,1)]+['us_white_sidea_label_20261114_0001.tif','us_white_achterkant_20261113_0005.tif']
got=order(b)
assert V.NAME_ORDER
assert got[:6]==['us_white_achterkant_20261113_0005.tif']+[f'us_white_sidea_20261114_000{i}.tif' for i in (1,2,3,4)]+['us_white_sidea_label_20261114_0001.tif'],got
print('Scan order: running counter across names and days; a counter per named part keeps the parts together.')
