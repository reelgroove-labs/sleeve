"""Identify source-observed white paper before correcting its midtones."""
import cv2 as cv
import numpy as np


def agreement(images, masks, px_mm, diagnostics=None):
    """Find a shared near-white material; printed colors may still need correction."""
    if len(images)<2:return None
    fractions=[];smooth=[];white_masks=[];anchors=[]
    for image,mask in zip(images,masks):
        if not mask.any():return None
        white=(image.min(2)>=245)&(np.ptp(image,axis=2)<=12)
        fractions.append(float(white[mask].mean()))
        white_masks.append(white&mask)
        anchors.append(np.quantile(image[white&mask],.95,axis=0).tolist() if (white&mask).any() else [0,0,0])
        smooth.append(cv.GaussianBlur(image.astype(np.float32),(0,0),max(.5,.5*px_mm)))
    if diagnostics is not None:diagnostics['white_fractions']=fractions
    if np.mean(fractions)<=.5:return None
    neighbors=[set() for _ in images];differences=[]
    for i in range(len(images)):
        for j in range(i+1,len(images)):
            overlap=cv.erode((masks[i]&masks[j]).astype(np.uint8),np.ones((3,3),np.uint8))>0
            if overlap.sum()<200:continue
            same_white=overlap&white_masks[i]&white_masks[j]
            if same_white.sum()<200:return None
            error=np.abs(smooth[i]-smooth[j]).max(2)
            foreground=overlap&((smooth[i].min(2)<245)|(smooth[j].min(2)<245))
            white_error=np.abs(images[i].astype(np.float32)-images[j]).max(2)
            whole=float(np.quantile(white_error[same_white],.95))
            printed=float(np.quantile(error[foreground],.9)) if foreground.sum()>=200 else whole
            differences.append([i,j,whole,printed])
            if diagnostics is not None:diagnostics.setdefault('pairs',[]).append([i,j,whole,printed])
            # Different lamp exposures may differ here; both observations
            # already identify the same nearly white material. Midtone
            # correction remains enabled, with its measured white endpoint.
            neighbors[i].add(j);neighbors[j].add(i)
    seen={0};pending=[0]
    while pending:
        for other in neighbors[pending.pop()]-seen:seen.add(other);pending.append(other)
    if len(seen)!=len(images):return None
    return {'white_fractions':fractions,'pair_differences':differences,'white_anchors':anchors}


def preserve_white(raw, corrected, anchor):
    """Limit correction near a measured white endpoint; do not invent white.

    Dark tones and colored ink keep the fitted correction. The highlight shoulder
    is twice the predicted movement of the measured white endpoint. Every
    result stays between its actual source value and the fitted correction.
    """
    original=np.asarray(raw,dtype=np.float32)
    endpoint=np.asarray(anchor,np.float32)
    gain=corrected/np.maximum(original,1)
    shoulder=np.maximum(2*np.abs(endpoint*(gain-1)),1)
    room=np.clip((endpoint-original)/shoulder,0,1)
    neutral=np.clip((22-np.ptp(original,axis=2))/10,0,1)
    strength=1-neutral[:,:,None]*(1-room)
    return original+(corrected-original)*strength


def paper_samples(raw, mask, width, height, step=4):
    """Select an actual light pixel per small region, without averaging ink
    into paper or combining maxima from different colored pixels."""
    cols,rows=width//step,height//step
    samples=np.empty((rows,cols,3),np.float32)
    valid=np.empty((rows,cols),bool)
    x=np.arange(cols)
    for row in range(rows):
        band=np.clip(np.asarray(raw[row*step:(row+1)*step,:cols*step],np.float32),0,255)
        pixels=band.reshape(step,cols,step,3).transpose(1,0,2,3).reshape(cols,step*step,3)
        covered=mask[row*step:(row+1)*step,:cols*step].reshape(step,cols,step).transpose(1,0,2).reshape(cols,step*step)
        light=np.where(covered,pixels.min(2),-1)
        pick=light.argmax(1)
        samples[row]=pixels[x,pick]
        valid[row]=covered.any(1)
    return samples,valid
