import cv2,numpy as np,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Resources/Tools'))
import scan_label
pm=10.;dpi=254;size=1300;centre=size//2;out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path('/tmp/sleeve-label-material-check');out.mkdir(exist_ok=True,parents=True);results=[]
for diameter,ring in [(100,93),(80,72),(106,96),(100,88)]:
 im=np.full((size,size,3),(180,180,180),np.uint8);cv2.circle(im,(centre,centre),round(diameter/2*pm),(224,224,224),-1);cv2.circle(im,(centre,centre),round(ring/2*pm),(40,40,40),30 if ring==88 else 3);cv2.circle(im,(centre,centre),round(3.6*pm),(250,250,250),-1)
 cv2.putText(im,'SIDE ONE',(centre-110,centre+80),0,1,(30,30,30),2);p=out/f'material-{diameter}.png';cv2.imwrite(str(p),im)
 lab,facts=scan_label.extract(str(p),dpi,material_only=True);assert lab is not None,facts;actual=facts['diameter_mm'];assert abs(actual-diameter)<.8,(diameter,ring,facts)
 assert abs(actual-ring)>2,(actual,ring)
 left,top,right,bottom=facts['source_bounds'];assert lab.shape[:2]==(bottom-top,right-left)
 # Native mask extends to physical paper beyond the printed ring.
 y=round(centre-top);x=round(centre+(diameter/2-.5)*pm-left);assert lab[y,x,3]>250,(diameter,'paper outside print discarded')
 results.append({'physical_diameter_mm':diameter,'printed_ring_mm':ring,'measured_mm':actual});print(results[-1],flush=True)
(out/'Material-mask-controle.json').write_text(json.dumps(results,indent=2))

p=out/'printed-ring-without-material-edge.png';im=np.full((1300,1300,3),180,np.uint8);cv2.circle(im,(650,650),465,(30,30,30),3);cv2.circle(im,(650,650),36,(250,250,250),-1);cv2.putText(im,'SIDE ONE',(520,750),0,1,(30,30,30),2);cv2.imwrite(str(p),im)
label,facts=scan_label.extract(str(p),254,material_only=True);assert label is None, facts
print('PASS: printed circle without lasting material evidence is refused')
(out/'Geen-materiaalrand-controle.json').write_text(json.dumps(facts,indent=2))

# Known geometry: near-edge ink can be the first lasting departure from the
# inner printed band. The outer material is different and its actual step is
# a complete circle. Keep the original coverage thresholds; recover its edge.
yy,xx=np.ogrid[:1400,:1400];rad=np.hypot(xx-700,yy-700);ang=np.degrees(np.arctan2(yy-700,xx-700))%360
for sector in (60,80,90):
 im=np.empty((1400,1400,3),np.uint8);im[:]=(20,20,180);im[rad<500]=224
 im[(rad>460)&(rad<490)&(ang<sector)]=25
 boundary=scan_label.persistent_material_boundary(im,700,700,500,10.)
 assert boundary is not None,(sector,'visible material edge rejected')
 cx,cy,radius,coverage=boundary
 assert np.hypot(cx-700,cy-700)<1 and abs(radius-500)<1,(sector,boundary)
 assert coverage>=.8,(sector,boundary)
# A partial edge without enough sector support must still be refused.
im=np.full((1400,1400,3),224,np.uint8);im[(rad>=500)&(ang<180)]=(20,20,180)
assert scan_label.persistent_material_boundary(im,700,700,500,10.) is None
print('PASS: near-edge printing distinguished from outside material; partial edge refused')

# Zachte, ruizige papierrand (metallic vinyl, Together Alone goud B / Graduated Fool zilver B): de sterkste gradient springt
# binnen de overgang en haalt de ellips in te weinig richtingen. Het midden van de overgang bevestigt de rand rondom; de
# maat blijft die van de echte rand. Is de rand ergens echt niet te zien (plaat even donker als het label), dan geen masker.
rng=np.random.default_rng(7);S=2600;c=S/2;pm3=300/25.4;R=50*pm3
yy,xx=np.mgrid[:S,:S].astype(np.float32);rad3=np.hypot(xx-c,yy-c);ang3=np.degrees(np.arctan2(yy-c,xx-c))%360
def soft_label(soft_sectors,hidden=()):
 t=np.clip((rad3-(R-.6*pm3))/(1.2*pm3),0,1);soft=np.zeros_like(rad3,bool);gone=np.zeros_like(rad3,bool)
 for a0,a1 in soft_sectors:soft|=(ang3>=a0)&(ang3<a1)
 for a0,a1 in hidden:gone|=(ang3>=a0)&(ang3<a1)
 v=np.where(soft,40+130*t,np.where(rad3<R,40,170)).astype(np.float32);v[gone&(rad3>=R-.6*pm3)]=40
 v+=rng.normal(0,14,v.shape).astype(np.float32)*soft
 im=np.dstack([v*.9,v,v*1.1]).clip(0,255).astype(np.uint8);cv2.circle(im,(int(c),int(c)),int(3.6*pm3),(250,250,250),-1);return im
p=out/'soft-edge.png';cv2.imwrite(str(p),soft_label([(0,120),(180,300)]))
lab,facts=scan_label.extract(str(p),300,material_only=True)
assert lab is not None and facts['edge_coverage']>=.8 and 'edge_confirmed_by' in facts,facts
assert abs(facts['diameter_mm']-100)<.5,facts
p=out/'soft-edge-partly-hidden.png';cv2.imwrite(str(p),soft_label([(0,120),(180,300)],hidden=[(120,180),(300,345)]))
lab,facts=scan_label.extract(str(p),300,material_only=True)
assert lab is None or facts['edge_coverage']<.8,facts
print('PASS: soft paper edge confirmed by the midpoint of the transition; an edge hidden in places is not')
