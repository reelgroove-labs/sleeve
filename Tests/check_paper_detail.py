"""Unobserved canvas pixels cannot count as sharper print."""
from pathlib import Path
import sys,cv2,numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Resources'/'Tools'))
from scan_stitch import masked_detail_energy
mask=np.zeros((120,180),bool);mask[15:105,25:155]=True
paper=np.full(mask.shape,235,np.float32);padding=paper.copy();padding[~mask]=0
assert np.array_equal(masked_detail_energy(paper,mask,6),masked_detail_energy(padding,mask,6))
assert not masked_detail_energy(padding,mask,6).any()
printed=padding.astype(np.uint8);cv2.putText(printed,'TEXT',(35,75),cv2.FONT_HERSHEY_SIMPLEX,1.1,20,2);printed=printed.astype(np.float32)
assert masked_detail_energy(printed,mask,6)[30:90,40:140].max()>100
padding[~mask]=255
assert not masked_detail_energy(padding,mask,6).any()
print('Papierdetail: padding verandert geen scherpte; echte inkt blijft meetbaar.')

from scan_feiten import white_margin_shadow
px_mm=20
blank=np.full((80,1200,3),(250,128,128),np.float32)
blank[:10,:,0]=225
assert white_margin_shadow(blank,px_mm)>0
ink=blank.copy()
# Pale strokes are sparse enough to evade the former 3% percentile check.
ink[3:65,590:598,0]=230
assert white_margin_shadow(ink,px_mm)==0
# A continuous blank shadow is still measurable; colored artwork is retained.
colored=blank.copy();colored[:,:,1]=160
assert white_margin_shadow(colored,px_mm)==0
print('Marge: blanco schaduw meetbaar; lichte inkt en gekleurd drukwerk beschermd.')

from scan_feiten import trim_white_insert_sides
paper=np.full((1200,600,3),250,np.uint8);paper[:,-10:]=225
_,cut=trim_white_insert_sides(paper,20)
assert cut[1]>0
_,protected=trim_white_insert_sides(paper,20,[(300,500,595,600)])
assert protected[1]==0
_,unaffected=trim_white_insert_sides(paper,20,[(300,500,500,600)])
assert unaffected==cut
print('Herhaald gemeten tekstvak: snede door letters gestopt; veilige afstand verandert de snede niet.')
