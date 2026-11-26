"""Repair interruptions of a physical line with agreeing, aligned scan pixels."""
import cv2
import numpy as np


def remap_rect(image, affine, left, top, width, height, step=1):
    matrix = affine.copy()
    matrix[:, 2] += matrix[:, :2] @ [left, top]
    matrix[:, :2] *= step
    return cv2.warpAffine(image, matrix, (width, height),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def line_strength(image, width, length):
    dark = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, np.ones((width, 1), np.uint8))
    return cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))


def spans(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def corroborated_gaps(merged, scans, px_mm):
    """Propose gaps bounded by a visible line in both original observations."""
    width = max(3, round(1.4 * px_mm) | 1)
    length = max(9, round(5.5 * px_mm) | 1)
    tolerance = max(1, round(.32 * px_mm))
    window = np.ones((2 * tolerance + 1, 1), np.uint8)
    strengths = []
    for gray in scans:
        valid = cv2.erode(np.uint8(gray > 0), np.ones((width, width), np.uint8))
        strengths.append(line_strength(gray, width, length) * valid)
    output = cv2.dilate(line_strength(merged, width, length), window)
    for i, first in enumerate(strengths):
        for second in strengths[i + 1:]:
            shared = np.minimum(first, cv2.dilate(second, window))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(shared >= 12))
            for component in range(1, count):
                x, y, w, h, area = stats[component]
                if w < 18 * px_mm or h > .1 * w or area < w:
                    continue
                section = np.s_[y:y+h, x:x+w]
                evidence = np.where(labels[section] == component, shared[section], 0)
                ys = evidence.argmax(axis=0)
                depth = evidence.max(axis=0)
                contrast = output[y + ys, np.arange(x, x+w)]
                weak = (depth >= 12) & (contrast < np.minimum(12, .5 * depth))
                shoulder = max(3, round(1.5 * px_mm))
                for begin, end in spans(weak):
                    if end-begin < 1.1 * px_mm or end-begin > 14 * px_mm:
                        continue
                    if begin < shoulder or end + shoulder > w:
                        continue
                    if (np.median(contrast[begin-shoulder:begin]) < 12
                            or np.median(contrast[end:end+shoulder]) < 12):
                        continue
                    yield i, x+begin, x+end, float(np.median(y + ys[begin:end]))


def repair_lines(image, sources, output_to_sources, dpi):
    """Only touch a gap when two scans independently show the same thin line."""
    if len(sources) < 2 or not dpi:
        return []
    px_mm = dpi / 25.4
    step = max(1, round(px_mm / 4.7))
    h, w = image.shape[:2]
    low_w, low_h = w // step, h // step
    scale = np.float32([[step, 0, 0], [0, step, 0]])
    small = cv2.warpAffine(image, scale, (low_w, low_h),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    scans = []
    for source, mapping in zip(sources, output_to_sources):
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        scans.append(remap_rect(source_gray, mapping, 0, 0, low_w, low_h, step))
    changed = []
    for vertical in (False, True):
        measures = corroborated_gaps(gray.T if vertical else gray,
                                    [s.T if vertical else s for s in scans], px_mm/step)
        for index, begin, end, center in measures:
            view = image.transpose(1, 0, 2) if vertical else image
            fade = round(.9 * px_mm)
            half = round(1.55 * px_mm)
            x0, x1 = begin*step-fade, end*step+fade
            y0, y1 = round(center*step)-half, round(center*step)+half+1
            if x0 < 0 or y0 < 0 or x1 > view.shape[1] or y1 > view.shape[0]:
                continue
            if any(v == vertical and max(x0, a) < min(x1, b) and abs(y0-c)<half
                   for v, a, b, c in changed):
                continue
            dst = view[y0:y1, x0:x1]
            if dst.shape[2] == 4 and dst[:, :, 3].min() < 250:
                continue
            matrix = output_to_sources[index]
            if vertical:
                matrix = matrix[:, [1, 0, 2]]
            original = remap_rect(sources[index], matrix, x0, y0, x1-x0, y1-y0)
            if not original.any(axis=2).all():
                continue
            # Match local lighting using strips beside the line. The line
            # itself is copied at native resolution from a single source.
            delta = dst[:, :, :3].astype(np.float32) - original
            border = max(2, half//4)
            above = np.median(delta[:border], axis=0)
            below = np.median(delta[-border:], axis=0)
            above, below = [cv2.GaussianBlur(row[None], (0, 0), max(1, fade))[0]
                            for row in (above, below)]
            y = np.linspace(0, 1, len(dst), dtype=np.float32)[:, None, None]
            corrected = original + (1-y)*above + y*below
            along = np.arange(dst.shape[1])
            across = np.arange(dst.shape[0])
            wx = np.clip(np.minimum(along, along[::-1]).astype(np.float32) / fade, 0, 1)
            wy = np.clip(np.minimum(across, across[::-1]).astype(np.float32) / (half-border), 0, 1)
            weight = (wy[:, None]*wx)[..., None]
            dst[:, :, :3] = np.clip(np.rint(dst[:, :, :3]*(1-weight) + corrected*weight), 0, 255)
            changed.append((vertical,x0,x1,y0))
    return changed
