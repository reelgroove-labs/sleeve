"""Rechtzetten en nummeren van losse vellen (0.5.53).
1. Een Discogs-overeenkomst met schaal 0 is ontaard en bepaalt geen stand (For Bitter Or Worse 2021 goud, tekstkant van het
   insert: 150 'punten' op de foto van de plaat, schaal 0; het vel bleef een kwartslag gedraaid).
2. Kleine tekst op een groot vel leest op een stuk op volle resolutie (die tekstkant: heel vel niets, midden 1685 tegen 566).
3. Het n-de deel met dezelfde naam in scanvolgorde krijgt _n, ook als twee delen tegelijk lopen (Queen For A Day 2021)."""
import os, sys, tempfile, types, numpy as np, cv2
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'Resources', 'Tools'))
import scan_feitenroute as FR, scan_rechtop as RT, scan_discogs as SD

# 1. ontaard tegen echt
img = np.full((400, 400, 3), 128, np.uint8)
SD.release_images = lambda url: (None, [img])
SD.matches = lambda small, photos, with_coverage=False: [(150, 0, np.float32([[0, 0, 300], [0, 0, 300]]), 0.0)]
assert FR.discogs_orientation(img, 'https://www.discogs.com/release/1') is None, 'ontaarde overeenkomst gaf een stand'
SD.matches = lambda small, photos, with_coverage=False: [(150, 0, np.float32([[0, -.5, 300], [.5, 0, 0]]), 0.6)]
assert FR.discogs_orientation(img, 'https://www.discogs.com/release/1') == (90, 150)

# 2. kleine lichte tekst op donker papier, een kwartslag gedraaid, op een vel van 3600 px
sheet = np.full((3600, 3600, 3), 45, np.uint8)
lines = ['Well have you ever met a girl like me before', 'So when I woke up there were three of us this morning',
         'You sure know how to treat me well tonight', "Don't want no flowers don't need love and understanding"]
for row in range(1100, 2500, 46):
    cv2.putText(sheet, lines[row // 46 % 4], (900, row), cv2.FONT_HERSHEY_SIMPLEX, .9, (215, 205, 180), 2, cv2.LINE_AA)
turned = cv2.rotate(sheet, cv2.ROTATE_90_CLOCKWISE)          # zo kwam het vel uit de scan
p = os.path.join(tempfile.mkdtemp(), 'insert.png'); cv2.imwrite(p, turned)
got = RT.centre_tile_text(p)
assert got and got['rotate_clockwise'] == 270 and got['verdict'] == 'LEESBAAR', got

# 3. vaste nummering in scanvolgorde
sys.path.insert(0, os.path.join(HERE, '..', 'Resources'))
import studio_bridge as B
cmds = []
class P:
    def __init__(self, cmd, **kw): cmds.append(cmd); self.stdout = types.SimpleNamespace(read=lambda: '{"part":"insert","verdict":"GOED"}'); self.stderr = iter(()); self.pid = 0
    def wait(self): return 0
B.subprocess.Popen = P
out = tempfile.mkdtemp(); from pathlib import Path
for n in (1, 2):
    B.feiten_part({}, 'insert', [f'/x/{n}.jpg'], Path(out), Path(out) / f'w{n}', None, 'vinyl', 300.0, None, n)
assert '--occurrence' not in cmds[0] and cmds[1][cmds[1].index('--occurrence') + 1] == '2', cmds
print('ontaarde Discogs-overeenkomst geeft geen stand; kleine tekst leest op volle resolutie; vellen genummerd in scanvolgorde')
