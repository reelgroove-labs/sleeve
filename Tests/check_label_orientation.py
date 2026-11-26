from pathlib import Path
import sys,tempfile,json,cv2,numpy as np
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'Resources' / 'Tools'))
import scan_rechtop as R,scan_feitenroute as F
metadata={'artists':[{'name':'First'},{'name':'Second'}],'title':'Album','tracklist':[{'title':'Track one'},{'title':'Track two'}],'labels':[{'name':'Publisher'}]}
with patch.object(F,'release_data',return_value=metadata):
 assert F.release_words('x',disc=True)==['First / Second','Album','Track one','Track two']
 assert 'Publisher' in F.release_words('x')
lines=[{'text':'Song title','height':20,'box':(100,100,400,130)}]
title=[{'text':'Album','height':30,'box':(100,200,400,240)}]
with patch.object(R,'label_lines',side_effect=[lines,title]),patch.object(R,'line_angle',return_value=.5):
 angle,how=R.level_label(np.zeros((500,500,3),np.uint8),0,['Artist','Album','Song title'])
 assert angle==359.5 and how.startswith('tracklist'),(angle,how)
with tempfile.TemporaryDirectory() as folder:
 p=str(Path(folder)/'label.png');cv2.imwrite(p,np.full((1000,1000,3),240,np.uint8))
 tracks=[{'text':t,'direction':24.1,'height':20,'width':150} for t in ['First track','Second track','Third track']]
 words=['Artist','Album','First track','Second track','Third track']
 with patch.object(R,'label_lines',return_value=tracks),patch.object(R,'level_label',return_value=(0,'largest text: Album')):
  angle,scores,hits=R._disc_angle(p,words,10)
  assert abs(angle-24.1)<.001 and 'title only' not in hits,(angle,hits)
 tracks[-1]['direction']=30
 with patch.object(R,'label_lines',return_value=tracks),patch.object(R,'level_label',return_value=(0,'largest text: Album')):
  angle,scores,hits=R._disc_angle(p,words,10)
  assert 'title only' in hits,hits
print('Standcontroles geslaagd: woordrollen, behouden trackmeting, overeenstemmende richtingen en conflict.')

with tempfile.TemporaryDirectory() as folder:
 p=str(Path(folder)/'label.png');cv2.imwrite(p,np.full((300,300,3),240,np.uint8))
 ambiguous=(0,{0:1},['title only'])
 tracks=(27,{27:3},['Track one','Track two','level: tracklist (3 lines)'])
 with patch.object(R,'_disc_angle',side_effect=[ambiguous,ambiguous,tracks]):
  angle,scores,hits=R.disc_angle(p,['Artist','Album','Track one','Track two'],2)
  assert angle==207 and scores=={207:3} and R._clear((angle,scores,hits),['Artist','Album'])
 with patch.object(R,'_disc_angle',side_effect=[ambiguous,ambiguous,(27,{27:3},['level: largest text: Album'])]):
  assert R.disc_angle(p,['Artist','Album','Track one'],2)==ambiguous
 with patch.object(R,'_disc_angle',return_value=tracks) as probe:
  assert R.disc_angle(p,['Artist','Album','Track one'],2)==tracks and probe.call_count==1
print('Halve OCR-draai: alleen betrouwbare trackregels, correcte hoekterugrekening; titel blijft twijfel.')

faint=np.full((300,300,3),254,np.uint8);faint[100:150,60:250]=235
line={'text':'ArtistLong AlbumTitle','direction':90}
with patch.object(R,'ocr_lines',side_effect=[[line],[line]]):
 assert R.faint_known_lines(faint,['ArtistLong','AlbumTitle'])==[line]
with patch.object(R,'ocr_lines',side_effect=[[line],[dict(line,direction=270)]]):
 assert R.faint_known_lines(faint,['ArtistLong','AlbumTitle'])==[]
with patch.object(R,'ocr_lines',return_value=[]) as reader:
 assert R.faint_known_lines(np.full_like(faint,254),['ArtistLong'])==[] and not reader.called
print('Lichte tekst: alleen herhaalde bekende tekst met dezelfde stand op twee leesformaten; blanco en conflict afgewezen.')
