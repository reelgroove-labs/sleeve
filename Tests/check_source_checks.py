"""Source checks must reject real damage without aliasing native scan detail."""
from pathlib import Path
import sys,json
import cv2 as cv,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Resources/Tools'))
import scan_feiten as facts
import scan_stitch as stitch
import scan_final_check as final
import scan_window_detail as detail
from scan_paper_light import agreement,preserve_white,paper_samples
from scan_seam_probe import write_source_evidence
cv.setNumThreads(2)
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)

# Remembered source identity survives a filename change, never changed pixels
# or ambiguous duplicates. The files need not decode: this tests identity only.
import hashlib
chosen=root/'renamed-detail';chosen.mkdir(exist_ok=True)
front=chosen/'front.tif';front.write_bytes(b'original cover')
donor=chosen/'detail.tif';donor.write_bytes(b'original detail')
config={'front_sources':{front.name:hashlib.sha256(front.read_bytes()).hexdigest()},
        'window_detail':str(donor),'detail_sha256':hashlib.sha256(donor.read_bytes()).hexdigest()}
(chosen/'.sleeve-detail-scans.json').write_text(json.dumps(config))
assert detail.remembered_detail([str(front)])==str(donor)
renamed=chosen/'new filename.tif';donor.rename(renamed)
assert Path(detail.remembered_detail([str(front)])).resolve()==renamed.resolve()
duplicate=chosen/'duplicate.tif';duplicate.write_bytes(renamed.read_bytes())
assert detail.remembered_detail([str(front)]) is None
duplicate.unlink();renamed.write_bytes(b'changed detail')
assert detail.remembered_detail([str(front)]) is None
renamed.write_bytes(b'original detail');front.write_bytes(b'changed cover')
assert detail.remembered_detail([str(front)]) is None

# A larger printed ring cannot erase a separately observed standard spindle
# opening. A genuinely enlarged aperture, absent evidence or an existing
# normal-size fit must retain the previous measurement.
spindle=np.full((400,400,3),50,np.uint8)
cv.circle(spindle,(200,200),36,(210,210,210),-1)
cv.circle(spindle,(200,200),60,(250,250,250),7)
assert facts.supported_standard_hole(spindle,200,200,36,60,10,1)
assert not facts.supported_standard_hole(None,200,200,36,36,10,1)
assert not facts.supported_standard_hole(None,200,200,36,41,10,1), 'overlapping edge measurements must preserve the worn opening'
assert not facts.supported_standard_hole(spindle,200,200,36,60,10,.4)
large=np.full((400,400,3),50,np.uint8)
cv.circle(large,(200,200),60,(210,210,210),-1)
assert not facts.supported_standard_hole(large,200,200,36,60,10,1)
assert facts.supported_standard_hole(255-spindle,200,200,36,60,10,1)
print('Direct spindle-edge evidence beats a printed ring; enlarged or unproven openings stay unchanged.')

# OpenCV's windowed phase measurement mutates its inputs. Later source checks
# must still see the original image, even after several overlapping windows.
test_gray=np.full((192,192),250,np.uint8)
cv.line(test_gray,(12,96),(180,96),30,2)
cv.line(test_gray,(96,12),(96,180),50,2)
test_color=cv.cvtColor(test_gray,cv.COLOR_GRAY2BGR)
test_best=np.zeros(test_gray.shape,np.uint8);test_best[:,96:]=1
test_masks={k:np.ones(test_gray.shape,bool) for k in (0,1)}
real_phase=cv.phaseCorrelate;real_lines=stitch.seam_line_conflicts
phase_calls=[]
def counted_phase(*args):
    phase_calls.append(True)
    return real_phase(*args)
def intact_source_lines(grays,*args):
    assert all(np.array_equal(g,test_gray) for g in grays), 'Measurement overwrote source observations'
    return []
try:
    cv.phaseCorrelate=counted_phase;stitch.seam_line_conflicts=intact_source_lines
    stitch.stitch_quality({0:test_color,1:test_color},test_masks,[0,1],test_best,
                         192,192,1,np.ones(test_gray.shape),12)
    assert phase_calls, 'Regression test must exercise the windowed measurement'
finally:
    cv.phaseCorrelate=real_phase;stitch.seam_line_conflicts=real_lines

# Genuine dark rim segments survive dense measurement, at different native dpi.
for ppm in (6.,12.,24.):
 n=round(64*ppm);y,x=np.mgrid[:n,:n];r=np.hypot(x-(n-1)/2,y-(n-1)/2)
 depth=(n/2-r)/ppm;theta=np.degrees(np.arctan2(y-(n-1)/2,x-(n-1)/2))%360
 values=np.clip(180+8*np.sin(depth*2*np.pi/.16),0,255).astype('uint8')
 im=cv.cvtColor(values,cv.COLOR_GRAY2BGRA);im[:,:,3]=np.where(depth>=0,255,0)
 original=im.copy();mask=(depth>=.3)&(depth<=1.3)&(theta>=30)&(theta<60)
 im[mask,:3]=np.maximum(0,im[mask,:3].astype(int)-24)
 measures=dict(facts.vinyl_edge_tone(im,ppm,True)['sectors'])
 assert measures[45]>=18,(ppm,measures)
 assert np.array_equal(im[:,:,3],original[:,:,3])
 # Missing material must not count as evidence of a clean edge.
 im[:,:,3]=0
 assert facts.vinyl_edge_tone(im,ppm,True)['sectors']==[]

# Native raster aliasing is suppressed before comparing with area-reduced scans.
h=w=420;y,x=np.mgrid[:h,:w]
gray=(190+45*np.sin(x*2*np.pi/3.3)*np.cos(y*2*np.pi/3.7)).astype('uint8')
gray[150:180,60:360]=30
native=cv.cvtColor(gray,cv.COLOR_GRAY2BGRA)
factor=3;reference=cv.resize(native,(140,140),interpolation=cv.INTER_AREA)
yy,xx=np.mgrid[18:110,18:110].astype('float32');sx=(xx+.5)*factor-.5;sy=(yy+.5)*factor-.5
T=np.array([[factor,0,0],[0,factor,0]],float)
observed=stitch.sample_analysis_patch(native,sx,sy,T)
expected=reference[18:110,18:110]
point=cv.remap(native,sx,sy,cv.INTER_LINEAR)
error=lambda a:np.mean(abs(a[:,:,:3].astype(float)-expected[:,:,:3]))
assert error(observed)<3 and error(point)>10
broken=native.copy();broken[150:180,60:360,:3]=190
lost=stitch.sample_analysis_patch(broken,sx,sy,T)
assert abs(lost[36,30,:3].mean()-observed[36,30,:3].mean())>100

# Fractional registration keeps the same contrast and residual limits. A real
# missing stroke or loss of sharpness is still rejected after registration.
shape=np.full((65,65),220,np.float32)
cv.circle(shape,(32,32),17,40,4);cv.line(shape,(14,46),(50,46),40,3)
shape=cv.GaussianBlur(shape,(0,0),2)
shifted=cv.warpAffine(shape,np.float32([[1,0,.45],[0,1,-.45]]),(65,65),borderMode=cv.BORDER_REFLECT)
assert stitch.faithful_source_patch(shifted,shape,4)
shifted[10:30,10:54]=220
assert not stitch.faithful_source_patch(shifted,shape,4)
assert not stitch.faithful_source_patch(cv.GaussianBlur(shape,(0,0),5),shape,4)

# Every exact quarter turn maps source pixels to cv.rotate's actual destination.
scene=np.arange(35).reshape(5,7).astype('uint8')
for degrees in (0,90,180,270):
 rotated=np.rot90(scene,-degrees//90)
 matrix=final.quarter_turn_matrix(scene.shape,degrees)
 for y in range(5):
  for x in range(7):
   X,Y,_=np.rint(matrix@[x,y,1]).astype(int)
   assert rotated[Y,X]==scene[y,x]
try:final.quarter_turn_matrix(scene.shape,89.9)
except ValueError:pass
else:raise AssertionError('Uncertain geometry must not approve observations')

# A replaced detail needs native source agreement; a boundary, altered line,
# missing alpha or moved source cannot be cleared by the old observations.
y,x=np.mgrid[:200,:220];donor=np.uint8((x*7+y*11)%180+30)
donor=cv.cvtColor(donor,cv.COLOR_GRAY2BGR);path=root/'donor.png';cv.imwrite(str(path),donor)
image=cv.cvtColor(donor,cv.COLOR_BGR2BGRA)
meta={'source':str(path),'source_to_final_px':np.eye(3).tolist(),
      'window_corners_px':[[0,0],[219,0],[219,199],[0,199]],'quarter_turn_clockwise':0}
assert detail.verified_final_points(image,meta,[[100,100],[2,2]],[15,15]).tolist()==[True,False]
assert detail.verified_final_points(image[:,:,:3],meta,[[100,100]],[15])[0]
damaged=image.copy();damaged[90:110,99:102,:3]=255
assert not detail.verified_final_points(damaged,meta,[[100,100]],[15])[0]
damaged=image.copy();damaged[100,100,3]=0
assert not detail.verified_final_points(damaged,meta,[[100,100]],[15])[0]
assert not detail.verified_final_points(image,{},[[100,100]],[15])[0]
projective=np.array([[1.005,.007,8.3],[-.004,.998,6.7],[.00001,-.00002,1.]])
warped=cv.warpPerspective(donor,projective,(240,220),flags=cv.INTER_CUBIC)
warped=cv.cvtColor(warped,cv.COLOR_BGR2BGRA)
projected=dict(meta,source_to_final_px=projective.tolist(),
               window_corners_px=cv.perspectiveTransform(np.float32(meta['window_corners_px'])[:,None,:],projective)[:,0].tolist())
assert detail.verified_final_points(warped,projected,[[110,110]],[15])[0]

# Recheck only clears an old warning when the final pixels supply proof.
folder=root/'proof';folder.mkdir(exist_ok=True)
patch=np.full((47,47),220,np.float32);cv.circle(patch,(23,23),14,40,4)
points=np.array([[100,100,True,False,-1,-1]])
write_source_evidence(folder,{(0,1):points},{(0,1):[patch]},np.eye(3)[:2],12,{0:'a.tif',1:'b.tif'})
canvas=np.full((200,200,4),220,np.uint8);canvas[:,:,3]=255
canvas[77:124,77:124,:3]=patch[:,:,None]
why=['other uncertainty',final.PREFIX+' a.tif / b.tif (1 unconfirmed locations)']
checked,report=final.recheck(canvas,folder,np.eye(3),why)
assert checked==['other uncertainty'] and report['checked_observations']==1
checked,report=final.recheck(canvas[:,:,:3],folder,np.eye(3),why)
assert checked==['other uncertainty'] and report['checked_observations']==1
canvas[:,:,:3]=220
checked,report=final.recheck(canvas,folder,np.eye(3),why)
assert len(checked)==2 and report['pairs'][0]['unconfirmed']==1
checked,report=final.recheck(canvas,root/'absent',np.eye(3),why)
assert checked==why and report is None
print('Source checks passed: dense rim, native raster, geometry, final source evidence and real damage.')


# A measured white endpoint constrains exposure correction without replacing
# source colors. Dark ink and colored artwork keep their fitted correction.
a=np.full((100,120,3),250,np.float32);a[30:50,40:80]=[30,80,150]
m=np.ones(a.shape[:2],bool)
assert agreement([a,a],[m,m],4) is not None
assert agreement([a,a*.7],[m,m],4) is None
assert agreement([a],[m],4) is None
ramp=np.repeat(np.arange(256,dtype=np.float32)[None,:,None],3,axis=2)
for gain in (.8,.94,1.06,1.2):
 fitted=ramp*gain;restored=preserve_white(ramp,fitted,[255,255,255])
 assert np.all(restored>=np.minimum(ramp,fitted)-1e-4)
 assert np.all(restored<=np.maximum(ramp,fitted)+1e-4)
 assert np.all(np.diff(restored[0,:,0])>=0)
 assert np.array_equal(restored[:,:128],fitted[:,:128])
 assert np.array_equal(restored[:,-1],ramp[:,-1])
colored=np.full((10,10,3),[50,140,245],np.float32)
assert np.array_equal(preserve_white(colored,colored*.9,[253,254,255]),colored*.9)
print('Observed paper white retained; fitted midtones, ink, color and ordering preserved.')

# Numbered track rows include leading zeros. Coarse readable text by itself
# cannot approve a label whose track rows could not actually be levelled.
import scan_rechtop as upright
from unittest.mock import patch,mock_open
import io,contextlib
track={'text':'01 RUN AWAY TOGETHER','box':(300,600,800,650),'height':50,'width':500}
assert upright.numbered_track_lines([track],(1000,1000,3))==[track]
for how,expected in [('none','TWIJFEL'),('tracklist (5 lines)','GOED')]:
 captured=io.StringIO()
 with patch.object(sys,'argv',['scan_rechtop','unused','--disc','--words','unused.json']),patch('builtins.open',mock_open(read_data='[]')),patch.object(upright,'disc_angle',return_value=(0,{0:80,180:0},['level: '+how])),contextlib.redirect_stdout(captured):
  upright.main()
 assert json.loads(captured.getvalue())['verdict']==expected
print('Leading-zero tracks level correctly; unlevelled labels stay uncertain.')

# A short OCR word error must not discard an otherwise corroborated phrase.
# Wrong/missing words, a single brand name and conflicting directions remain
# uncertain. These are observed OCR strings, never album-specific code paths.
first={0:[],90:[', 3SIIOM tIo t131118 yO'],180:[],270:['FOR BITTER OR WORSE']}
second={0:[],90:['3SIIOM tio t131118 yO'],180:[],270:['ok BITTER OR WORSE']}
assert upright.corroborated_short_text(first,second)['rotate_clockwise']==270
assert upright.corroborated_short_text({0:['ANOUK']},{0:['ANOUK']}) is None
assert upright.corroborated_short_text({0:['FOR BITTER OR WORSE']},
                                     {0:['NOT BETTER AND WORSE']}) is None
assert upright.corroborated_short_text({0:['FOR BITTER OR WORSE'],90:['A BRIGHTER FUTURE']},
                                     {0:['ok BITTER OR WORSE'],90:['A BRIGHTER FUTURE']}) is None
assert upright.corroborated_short_text({0:['A BRIGHTER FUTURE']},
                                     {0:['A BRIGHTER FUTURE']})['rotate_clockwise']==0
print('Short phrase agreement survives a small OCR error; ambiguity remains uncertain.')

# Independent straight rows can establish a label baseline without a tracklist.
# Use real raster text at a known angle, including a conflicting layout.
def text_rows(angles, centers):
 canvas=np.full((1240,1240,3),240,np.uint8);lines=[]
 for i,(angle,cy) in enumerate(zip(angles,centers)):
  text='ARCHIVE TEXT ROW %02d'%i
  tile=np.full((120,800,3),240,np.uint8)
  cv.putText(tile,text,(45,75),cv.FONT_HERSHEY_SIMPLEX,1.5,(20,20,20),2,cv.LINE_AA)
  tile=cv.warpAffine(tile,cv.getRotationMatrix2D((400,60),angle,1),(800,120),borderValue=(240,240,240))
  yy,xx=np.where(tile[:,:,0]<120);x0,x1=xx.min()+220,xx.max()+221;y0,y1=yy.min()+cy-60,yy.max()+cy-59
  canvas[cy-60:cy+60,220:1020]=tile
  lines.append({'text':text,'box':(x0,y0,x1,y1),'width':x1-x0,'height':y1-y0,'direction':0})
 return canvas,lines
view,rows=text_rows([2.5]*4,[400,500,600,700])
measured=upright.parallel_label_text(view,rows)
assert measured and abs(measured[0]+2.5)<.2 and measured[1]>=3,measured
assert upright.parallel_label_text(view,rows[:2]) is None
assert upright.parallel_label_text(view,[rows[0]]*3) is None
view,rows=text_rows([2.5]*3+[-2.5]*3,[200,300,400,760,860,960])
assert upright.parallel_label_text(view,rows) is None
# A circular rim's diagonal fragments must not pass an axis-aligned centre box.
view=np.full((1240,1240,3),240,np.uint8);rows=[]
for degrees in (38,45,52):
 tile=np.full((100,100,3),240,np.uint8)
 cv.putText(tile,'PUBLIC',(25,55),cv.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv.LINE_AA)
 tile=cv.warpAffine(tile,cv.getRotationMatrix2D((50,50),45,1),(100,100),borderValue=(240,240,240))
 yy,xx=np.where(tile[:,:,0]<120)
 x=round(620+590*np.cos(np.radians(degrees)));y=round(620+590*np.sin(np.radians(degrees)))
 view[y-50:y+50,x-50:x+50]=np.minimum(view[y-50:y+50,x-50:x+50],tile)
 rows.append({'text':'PUBLIC','box':(xx.min()+x-50,yy.min()+y-50,xx.max()+x-49,yy.max()+y-49),'width':46,'height':10,'direction':45})
assert upright.parallel_label_text(view,rows) is None
# Widely separated pieces of a single sloping title are one baseline, too.
pieces=[{'text':'PUBLIC','box':(x,500+x*np.tan(np.radians(4)),x+70,515+(x+70)*np.tan(np.radians(4))),
         'width':70,'height':15,'direction':4} for x in (180,500,820)]
with patch.object(upright,'line_angle',return_value=4):
 assert upright.parallel_label_text(np.zeros((1240,1240,3),np.uint8),pieces) is None
with patch.object(upright,'label_lines',return_value=pieces),patch.object(upright,'numbered_track_lines',return_value=[]),patch.object(upright,'parallel_label_text',return_value=(4.,3)):
 assert upright.level_label(np.zeros((1240,1240,3),np.uint8),0,[])[1]=='none'
with patch.object(upright,'label_lines',side_effect=[[track],[]]),patch.object(upright,'line_angle',return_value=.5),patch.object(upright,'parallel_label_text',side_effect=AssertionError('Must preserve established track rows')):
 angle,how=upright.level_label(np.zeros((1000,1000,3),np.uint8),0,[])
 assert angle==359.5 and how=='tracklist (1 lines)'
print('Parallel text baselines measured; sparse, duplicate and competing rows rejected; track results preserved.')

# Ink must not turn the measured paper color gray at the inspection scale.
# Nor may independent channel maxima invent white from a colored mosaic.
page=np.full((16,20,3),251,np.float32);page[::2,::2]=30
sample,mask=paper_samples(page,np.ones(page.shape[:2],bool),20,16)
assert np.all(sample==251) and mask.all()
mosaic=np.zeros((16,20,3),np.float32)
for c in range(3):mosaic[c::3,:,c]=255
sample,_=paper_samples(mosaic,np.ones(mosaic.shape[:2],bool),20,16)
assert np.all(sample.min(2)==0)
print('Paper sampling keeps observed pixel colors and separates ink from white paper.')

clipped,_=paper_samples(np.full((8,8,3),270,np.float32),np.ones((8,8),bool),8,8)
assert clipped.max()==255
