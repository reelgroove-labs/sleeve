"""Choose an overlapping source for a record rim without altering its cut mask."""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


# Coordinates in millimetres relative to the existing cut. Positive is outside.
DEPTHS = np.arange(-5., 3.51, .1, dtype=np.float32)
ANGLES = 5760
WINDOW = 64                     # four degrees, about 10 mm on a 12-inch LP
STEP = 32
RIM = (DEPTHS >= -2.4) & (DEPTHS <= -.12)
REFERENCE = (DEPTHS >= -4.7) & (DEPTHS <= -3.5)
EDGE = np.flatnonzero((DEPTHS >= -.7) & (DEPTHS <= 2.65))


def radial_sample(image, x, y, transform):
    u = (transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]).astype(np.float32)
    v = (transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]).astype(np.float32)
    sampled = cv2.remap(image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid = (u >= 2) & (u < image.shape[1] - 2) & (v >= 2) & (v < image.shape[0] - 2)
    return sampled.astype(np.float32), valid


def edge_positions(samples, use_backing_colour):
    # A transition must persist across a short radial band. A single groove or
    # speck cannot by itself count as the physical boundary.
    smooth = cv2.blur(samples, (5, 3))
    difference = smooth[:, EDGE + 3] - smooth[:, EDGE - 3]
    derivative = np.linalg.norm(difference, axis=2)
    if use_backing_colour:
        inside = np.median(smooth[:, REFERENCE], axis=1)
        outside = np.median(smooth[:, DEPTHS >= 2.85], axis=1)
        chroma = outside - inside
        chroma -= chroma.mean(axis=1, keepdims=True)
        separation = np.linalg.norm(chroma, axis=1)
        use_colour = (separation >= 23) & (np.ptp(inside, axis=1) < 23)
        projection = np.sum(difference * chroma[:, None, :], axis=2) / np.maximum(separation[:, None], 1)
        # Pure shading changes brightness in all channels. The direction towards
        # the coloured backing gives an independent boundary on neutral vinyl.
        derivative[use_colour] = projection[use_colour] * 1.4
    pos = derivative.argmax(axis=1)
    strength = derivative[np.arange(len(samples)), pos]
    return DEPTHS[EDGE[pos]], strength


def choose_segments(current, candidates, use_backing_colour=False):
    """Return selected scans, improvements and matching exposure per angular sample."""
    count = len(current)
    choice = np.full(count, -1, np.int32)
    improvement = np.zeros(count, np.float32)
    correction = np.zeros((count, 3), np.float32)
    joins = np.zeros(count, np.float32)
    output_inner = np.median(current[:, REFERENCE], axis=1)
    output_profile = np.linalg.norm(current - output_inner[:, None], axis=2)
    output_smooth = cv2.blur(output_profile, (3, 5))
    inspected = (DEPTHS >= -3.45) & (DEPTHS <= -.05)
    def dark_detail(rgb):
        light = rgb.mean(axis=2)
        return cv2.GaussianBlur(light, (0, 0), 3) - light

    old_dark = dark_detail(current)
    evidence = [(sample, valid.all(axis=1), *edge_positions(sample, use_backing_colour), dark_detail(sample))
                for sample, valid in candidates]
    for start in range(0, count, STEP):
        ids = np.arange(start, start + WINDOW) % count
        usable = [
            i for i, (_, full, edge, strength, _) in enumerate(evidence)
            if full[ids].all() and np.quantile(strength[ids], .2) >= 19
            and np.ptp(np.quantile(edge[ids], [.1, .9])) <= .5
        ]
        if len(usable) < 2:
            continue
        chroma = np.ptp(np.median(output_inner[ids], axis=0))
        # On coloured vinyl, do not exchange groove texture for a smoother scan.
        # Score its very outer strip; on neutral vinyl inspect further in.
        judged = (DEPTHS >= -.85) & (DEPTHS <= -.05) if chroma >= 25 else RIM
        old_score = output_smooth[ids][:, judged]
        original_score = np.quantile(old_score, .9)
        for i in usable:
            source, _, edge, _, dark = evidence[i]
            if np.quantile(edge[ids], .1) < -.12:
                continue
            # The same boundary must be visible at the same position in a
            # second original, irrespective of their rim colours.
            if not any(j != i and np.quantile(np.abs(edge[ids] - evidence[j][2][ids]), .85) <= .4 for j in usable):
                continue
            inner = np.median(source[ids][:, REFERENCE], axis=1)
            gain = np.median(output_inner[ids] - inner, axis=0)
            if np.max(np.abs(gain)) > 18:
                continue
            matching = np.linalg.norm(output_inner[ids] - inner - gain, axis=1)
            if np.quantile(matching, .9) > 7:
                continue
            if chroma < 25:
                dark_here = dark[ids][:, inspected]
                if ((dark_here > 9) & (dark_here - old_dark[ids][:, inspected] > 8)).any():
                    continue
            full_profile = np.linalg.norm(source[ids] - inner[:, None], axis=2)
            source_profile = cv2.blur(full_profile, (3, 5))[:, judged]
            if np.quantile(full_profile[:, inspected], .997) > max(
                    20, np.quantile(output_profile[ids][:, inspected], .997) + 4):
                continue
            source_score = np.quantile(source_profile, .9)
            better_by = original_score - source_score
            if better_by < max(5., .3 * original_score):
                continue
            # Judge the whole segment; avoid a collection of one-pixel picks.
            if np.median(np.quantile(old_score, .8, axis=1)
                         - np.quantile(source_profile, .8, axis=1)) < 2:
                continue
            take = ids[better_by > improvement[ids]]
            choice[take], improvement[take], correction[take] = i, better_by, gain
            joins[take] = 1.2 if chroma >= 25 else 3.45
    return choice, correction, joins


def refine(crop, facts, metadata_path, paths):
    if crop.ndim != 3 or crop.shape[2] != 4 or 'cut_circle_px' not in facts:
        return []
    if crop.shape[0] != crop.shape[1]:
        return []
    dpi = float(facts.get('dpi') or 0)
    if dpi < 100:
        return []
    from scan_stitch import open_srgb

    px_mm = dpi / 25.4
    meta = json.loads(Path(metadata_path).read_text())
    raw_x, raw_y, radius = facts['cut_circle_px']
    left, top = round(raw_x - radius), round(raw_y - radius)
    cx, cy = raw_x - left, raw_y - top
    # Use the same centred circle as the cutout. The original centre is
    # retained separately to place the original scans on that cutout.
    centre = (len(crop) - 1) / 2
    if abs(cx - centre) > 2 or abs(cy - centre) > 2:
        return []
    theta = np.arange(ANGLES, dtype=np.float32) * (2 * np.pi / ANGLES)
    r = radius + DEPTHS[None, :] * px_mm
    x = centre + np.cos(theta[:, None]) * r
    y = centre + np.sin(theta[:, None]) * r
    current, _ = radial_sample(crop[:, :, :3], x, y, np.eye(2, 3))
    scans = {}
    for f in paths:
        name = Path(f).name
        scans[name] = None if name in scans else f
    candidates, mappings = [], []
    for scan in meta.get('scans', []):
        filename = scans.get(scan['scan'])
        if not filename:
            continue
        with open_srgb(filename) as im:
            w, h = im.size
            points = np.float32(scan['corners'][:3]) - np.float32([left, top])
            transform = cv2.getAffineTransform(points, np.float32([[0, 0], [w, 0], [w, h]]))
            candidates.append(radial_sample(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), x, y, transform))
        mappings.append((filename, transform))
    if len(candidates) < 2:
        return []
    baseline = choose_segments(current, candidates)
    alternative = choose_segments(current, candidates, use_backing_colour=True)
    baseline_rgb = crop.copy() if any(not np.array_equal(a, b) for a, b in zip(baseline, alternative)) else crop
    baseline_changes = render_segments(baseline_rgb, mappings, baseline, centre, radius, px_mm)
    if baseline_rgb is crop:
        return baseline_changes

    changes = render_segments(crop, mappings, alternative, centre, radius, px_mm)
    from scan_feiten import VINYL_EDGE_WARNING, vinyl_edge_tone

    previous = dict(vinyl_edge_tone(baseline_rgb, px_mm, True)['sectors'])
    examined = dict(vinyl_edge_tone(crop, px_mm, True)['sectors'])
    if previous and previous.keys() == examined.keys() and all(
            examined[angle] < VINYL_EDGE_WARNING or examined[angle] <= level
            for angle, level in previous.items()):
        return changes
    crop[:] = baseline_rgb
    return baseline_changes


def render_segments(crop, mappings, segments, centre, radius, px_mm):
    from scan_stitch import open_srgb

    choice, offset, joins = segments
    changes, patches = [], []
    half = len(crop)
    xx = np.arange(half, dtype=np.float32)[None, :] - centre
    for i, (filename, transform) in enumerate(mappings):
        use = choice == i
        if not use.any():
            continue
        # Feather only at the ends and on the inner join; alpha is untouched.
        padded = np.tile(use.astype(np.uint8), 3)
        along = cv2.distanceTransform(padded[None, :], cv2.DIST_L2, 3)[0, ANGLES:2 * ANGLES]
        alpha = np.clip(along / 12., 0, 1)
        changed = 0
        with open_srgb(filename) as im:
            source = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            for y0 in range(0, half, 192):
                tile = crop[y0:y0 + 192]
                yy = np.arange(y0, y0 + len(tile), dtype=np.float32)[:, None] - centre
                radial = (radius - np.hypot(xx, yy)) / px_mm
                near = (radial < 3.45) & (radial > -.12) & (tile[:, :, 3] > 0)
                ys, xs = np.where(near)
                angle = np.rint((np.arctan2(yy[ys, 0], xx[0, xs]) % (2 * np.pi)) * (ANGLES / (2 * np.pi))).astype(int) % ANGLES
                take = use[angle]
                ys, xs, angle = ys[take], xs[take], angle[take]
                if not len(xs):
                    continue
                u = (transform[0, 0] * xs + transform[0, 1] * (ys + y0) + transform[0, 2]).astype(np.float32)
                v = (transform[1, 0] * xs + transform[1, 1] * (ys + y0) + transform[1, 2]).astype(np.float32)
                for start in range(0, len(xs), 12000):
                    sl = slice(start, start + 12000)
                    pixels = cv2.remap(source, u[sl][None], v[sl][None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0].astype(np.float32)
                    row, col, ang = ys[sl], xs[sl], angle[sl]
                    weight = alpha[ang] * np.clip((joins[ang] - radial[row, col]) / (.35 * joins[ang]), 0, 1)
                    old = tile[row, col, :3]
                    new = np.clip(np.rint(old + weight[:, None] * (pixels + offset[ang] - old)), 0, 255).astype(np.uint8)
                    updated = np.any(new != old, axis=1)
                    changed += np.count_nonzero(updated)
                    if updated.any():
                        patches.append((row[updated] + y0, col[updated], new[updated]))
        if changed:
            changes.append((Path(filename).name, int(use.sum()), int(changed)))
    for rows, cols, pixels in patches:
        crop[rows, cols, :3] = pixels
    return changes
