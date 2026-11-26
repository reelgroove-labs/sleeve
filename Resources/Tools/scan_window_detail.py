"""Fit a complete detail scan to the measured four sides of a cover opening."""
import hashlib
import math
import json
from pathlib import Path

import cv2 as cv
import numpy as np


def window(image):
    scale = 1400 / max(image.shape[:2])
    preview = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    edges = cv.Canny(cv.cvtColor(preview, cv.COLOR_BGR2GRAY), 30, 85)
    lines = cv.HoughLinesP(edges, 1, np.pi / 1800, 70,
                          minLineLength=200, maxLineGap=35)
    if lines is None:
        raise ValueError('No frame edges')
    segments = lines.reshape(-1, 4).astype(float)
    vertical = [line for line in segments
                if abs(line[3] - line[1]) > abs(line[2] - line[0]) * 20]
    horizontal = [line for line in segments
                  if abs(line[2] - line[0]) > abs(line[3] - line[1]) * 20]
    best = None
    for left in vertical:
        for right in vertical:
            distance = (right[0] + right[2] - left[0] - left[2]) / 2
            if distance < .5 * preview.shape[1]:
                continue
            bottom = min(max(left[1], left[3]), max(right[1], right[3]))
            top = max(min(left[1], left[3]), min(right[1], right[3]))
            height = bottom - top
            if height < .12 * preview.shape[0] or not 1.8 < distance / height < 6:
                continue
            score = distance * height
            if best is None or score > best[0]:
                best = score, left, right, top, bottom, distance
    if best is None:
        raise ValueError('No corroborated rectangular window')
    _, left, right, top, bottom, width = best

    def candidates(y):
        matches = [line for line in horizontal
                   if abs(line[2] - line[0]) > .2 * width
                   and abs((line[1] + line[3]) / 2 - y) < .08 * width]
        if not matches:
            raise ValueError('Open frame')
        return matches

    def edge_score(line, y):
        return (abs((line[1] + line[3]) / 2 - y)
                + 20 * width / max(1, abs(line[2] - line[0])))

    bottom_line = min(candidates(bottom), key=lambda line: edge_score(line, bottom))

    def slope(line):
        return (line[3] - line[1]) / (line[2] - line[0])

    # Opposite edges of the flat, rectangular opening must be parallel.
    top_line = min(candidates(top),
                   key=lambda line: 2000 * abs(slope(line) - slope(bottom_line))
                   + edge_score(line, top))

    def intersect(first, second):
        point, direction = first[:2], first[2:] - first[:2]
        other_point, other_direction = second[:2], second[2:] - second[:2]
        along = np.linalg.solve(np.column_stack([direction, -other_direction]),
                                other_point - point)[0]
        return point + along * direction

    return np.float32([intersect(left, top_line), intersect(right, top_line),
                       intersect(right, bottom_line), intersect(left, bottom_line)]) / scale


def has_window(path):
    """Sleeve Transparant: ligt er in deze scan een volledig rechthoekig venster (in welke kwartslag ook)? Voor het
    automatisch herkennen van een detailscan van het venster in de voorkant (US white: de EYES-scan)."""
    image = cv.imread(str(path))
    if image is None:
        return False
    ratio = min(1., 2000 / max(image.shape[:2]))
    image = cv.resize(image, None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
    for k in range(4):
        try:
            window(np.rot90(image, k).copy())
            return True
        except (ValueError, np.linalg.LinAlgError):
            continue
    return False


def merge_detail(target, source_path, dpi):
    from scan_dpi import read
    source_dpi = read(source_path)
    if not source_dpi or abs(source_dpi-dpi) > 1:
        raise ValueError('The detail scan must have the same resolution as the cover')
    source = cv.imread(str(source_path))
    if source is None:
        raise ValueError('The detail scan could not be read')
    # Determine only the quarter turn from matching artwork. The frame itself
    # determines the placement, even if the insert behind the opening moved.
    sift = cv.SIFT_create(4500, contrastThreshold=.015)
    def features(image):
        ratio = min(1., 1500/max(image.shape[:2]))
        view = cv.resize(image[:, :, :3], None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
        keys, descriptors = sift.detectAndCompute(cv.cvtColor(view,cv.COLOR_BGR2GRAY), None)
        return keys, descriptors
    a,d = features(source)
    b,e = features(target)
    if d is None or e is None:
        raise ValueError('The detail has no matching printed features')
    pairs = [m for m,n in cv.BFMatcher().knnMatch(d,e,k=2) if m.distance < .72*n.distance]
    if len(pairs)<40:
        raise ValueError('Too little matching artwork in the detail')
    affine, inliers = cv.estimateAffinePartial2D(
        np.float32([a[m.queryIdx].pt for m in pairs]),
        np.float32([b[m.trainIdx].pt for m in pairs]), ransacReprojThreshold=2)
    if affine is None or inliers.sum()<40:
        raise ValueError('The detail could not be oriented against the cover')
    quarter = int(round(np.degrees(np.arctan2(affine[1,0],affine[0,0]))/90)) % 4
    if quarter:
        source=np.rot90(source,-quarter).copy()
    src,dst=window(source),window(target)
    lengths=lambda corners: np.array([np.linalg.norm(corners[(i+1)%4]-corners[i]) for i in range(4)])
    ratio=lengths(dst)/lengths(src)
    if np.max(np.abs(ratio-1)) > .025:
        raise ValueError('The detail opening differs by more than 2.5% from the cover')
    transform=cv.getPerspectiveTransform(src,dst)
    height,width=target.shape[:2]
    warped=cv.warpPerspective(source,transform,(width,height),flags=cv.INTER_CUBIC)
    mask=np.zeros((height,width),np.uint8)
    cv.fillConvexPoly(mask,np.rint(dst).astype(np.int32),255)
    # Include the frame and its narrow paper surround so that no part of the
    # old opening's boundary remains beside the newly registered frame.
    surround = max(2, round(dpi / 25.4))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*surround+1, 2*surround+1))
    mask = cv.dilate(mask, kernel)
    feather = max(2, .4*dpi/25.4)
    weight=np.minimum(cv.distanceTransform(mask,cv.DIST_L2,3)/feather,1).astype(np.float32)
    result=target.copy()
    for begin in range(0,height,256):
        rows=slice(begin,min(begin+256,height))
        alpha=weight[rows,:,None]
        result[rows,:,:3]=np.clip(np.rint(target[rows,:,:3]*(1-alpha)+warped[rows]*alpha),0,255)
    return result, {'source':str(source_path),'window_corners_px':dst.round(2).tolist(),
                    'source_to_final_px': transform.tolist(),
                    'quarter_turn_clockwise':quarter*90,'side_scale':ratio.round(5).tolist()}


def verified_final_points(image, detail, centers, radii):
    """Verify replaced observations against the designated native detail scan.

    Only patches wholly inside the opaque replacement qualify. Boundary and
    feather patches keep their existing uncertainty. No old source is used to
    judge artwork that was explicitly replaced by a different observation.
    """
    confirmed = np.zeros(len(centers), bool)
    if not detail or 'source_to_final_px' not in detail:
        return confirmed
    donor = cv.imread(detail['source'])
    if donor is None:
        return confirmed
    donor = np.rot90(donor, -round(detail['quarter_turn_clockwise']/90)).copy()
    corners = np.float32(detail['window_corners_px'])
    transform = np.asarray(detail['source_to_final_px'], float)
    for index, (center, radius) in enumerate(zip(centers, radii)):
        # The square patch must fit wholly inside the window.
        if cv.pointPolygonTest(corners, tuple(map(float, center)), True) <= radius*2**.5+2:
            continue
        x, y = np.rint(center).astype(int)
        r = math.ceil(radius)
        left, top, right, bottom = x-r, y-r, x+r+1, y+r+1
        if left < 0 or top < 0 or right > image.shape[1] or bottom > image.shape[0]:
            continue
        observed = image[top:bottom, left:right]
        if observed.shape[2] == 4 and observed[:, :, 3].min() < 250:
            continue
        offset = np.array([[1., 0, -left], [0, 1., -top], [0, 0, 1.]])
        expected = cv.warpPerspective(donor, offset@transform, (right-left, bottom-top),
                                      flags=cv.INTER_CUBIC)
        error = np.abs(observed[:, :, :3].astype(np.int16)-expected.astype(np.int16))
        confirmed[index] = np.quantile(error, .99) <= 1 and error.max() <= 4
    return confirmed


def remembered_detail(files):
    """Read the user's chosen detail source for this exact set of cover scans."""
    if not files:
        return None
    parents={Path(file).resolve().parent for file in files}
    if len(parents)!=1:
        return None
    config=next(iter(parents))/'.sleeve-detail-scans.json'
    try:
        data=json.loads(config.read_text())
        if set(data['front_sources'])!={Path(file).name for file in files}:
            return None
        def checksum(file):
            result=hashlib.sha256()
            with Path(file).open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    result.update(block)
            return result.hexdigest()
        if any(checksum(file)!=data['front_sources'][Path(file).name] for file in files):
            return None
        detail=Path(data['window_detail'])
        if not detail.is_file():
            # A filename is not the identity of the user's selected scan.
            # Recover a moved/renamed sibling only from its exact contents;
            # multiple copies remain ambiguous instead of picking by name.
            matches=[candidate for candidate in config.parent.iterdir()
                     if candidate.is_file() and candidate.suffix.lower() in
                     ('.tif','.tiff','.png','.jpg','.jpeg')
                     and candidate.name not in data['front_sources']
                     and checksum(candidate)==data['detail_sha256']]
            if len(matches)!=1:
                return None
            detail=matches[0]
        if checksum(detail)!=data['detail_sha256']:
            return None
        return str(detail)
    except (OSError,ValueError,KeyError,TypeError):
        return None
