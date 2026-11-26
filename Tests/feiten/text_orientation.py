"""Regressions for short cover titles and independent numbered label tracks."""
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'Resources/Tools'))
import scan_rechtop as R

first={0:[],90:['ToGETHER ALONE'],180:[],270:['3NOIV 213H13901']}
second={0:[],90:['TOGETHER ALONE'],180:[],270:[]}
with patch.object(R, 'read_rotations', return_value=second):
    assert R.sparse_text_orientation('unused.png',first)['rotate_clockwise']==90
with patch.object(R, 'read_rotations', return_value={0:[],90:['different title'],180:[],270:[]}):
    assert R.sparse_text_orientation('unused.png',first) is None
ambiguous={0:['Together Alone'],90:['Together Alone'],180:[],270:[]}
with patch.object(R,'read_rotations',return_value=ambiguous):
    assert R.sparse_text_orientation('unused.png',ambiguous) is None
lines=[
 {'text':'2 Fluid Conduction 4:15','box':(150,600,800,655),'width':650,'height':55},
 {'text':'6 Mood Indigo 4:30','box':(250,700,800,750),'width':550,'height':50},
 {'text':'01 Run Away Together','box':(250,550,800,600),'width':550,'height':50},
 {'text':'00 Not a numbered track','box':(250,500,800,550),'width':550,'height':50},
 {'text':'2016 Universal Music','box':(0,10,900,45),'width':900,'height':35},
]
assert len(R.numbered_track_lines(lines,(1000,1000,3)))==3
print('Short text needs independent agreement; numbered label lines exclude circular copyright text.')
