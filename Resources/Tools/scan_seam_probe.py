"""Save native-resolution review crops of unresolved seams on request."""
import json
from pathlib import Path
import tempfile

import cv2
from PIL import Image, ImageDraw


def write_seam_diagnostics(directory, part, image, points, dpi):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=(part or 'part') + '-', dir=root))
    height, width = image.shape[:2]
    radius = max(32, round((dpi or 300) / 25.4 * 4))
    groups = []
    for (x, y), reason, first, second in points:
        x, y = round(float(x)), round(float(y))
        point = {'pixel': [x, y], 'reason': reason, 'sources': [first, second]}
        group = next((g for g in groups if max(abs(x - g[0]), abs(y - g[1])) <= radius // 2), None)
        if group is None:
            group = (x, y, [])
            groups.append(group)
        group[2].append(point)

    index, tiles = [], []
    for number, (x, y, found) in enumerate(groups, 1):
        left, right = max(0, x - radius), min(width, x + radius + 1)
        top, bottom = max(0, y - radius), min(height, y + radius + 1)
        entry = {'points': found}
        if left < right and top < bottom:
            name = f'{number:02d}.png'
            crop = image[top:bottom, left:right]
            if not cv2.imwrite(str(folder / name), crop):
                raise OSError(f'Could not save seam crop: {name}')
            entry.update(image=name, origin_px=[left, top])
            tile = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA))
            preview = Image.new('RGB', tile.size, '#808080')
            preview.paste(tile, mask=tile.getchannel('A'))
            draw = ImageDraw.Draw(preview)
            for point in found:
                col, row = point['pixel']
                col, row = col - left, row - top
                draw.rectangle((col - 4, row - 4, col + 4, row + 4), outline='#ff2255')
            tiles.append((number, preview))
        index.append(entry)
    (folder / 'locations.json').write_text(json.dumps({
        'part': part, 'dpi': dpi, 'merge_size_px': [width, height],
        'unresolved_points': len(points), 'crops': index,
    }, indent=2))
    if tiles:
        width = max(tile.width for _, tile in tiles) + 16
        height = max(tile.height for _, tile in tiles) + 26
        cols = min(4, len(tiles))
        overview = Image.new('RGB', (cols * width, ((len(tiles) + cols - 1) // cols) * height), '#333333')
        draw = ImageDraw.Draw(overview)
        for i, (number, tile) in enumerate(tiles):
            x, y = (i % cols) * width + 8, (i // cols) * height
            draw.text((x, y + 3), f'{number:02d}', fill='white')
            overview.paste(tile, (x, y + 20))
        overview.save(folder / 'Overview.png')
    return folder


def write_source_evidence(folder, locations, evidence, transform, px_mm, names):
    """Retain original observations so later stages can check the final pixels."""
    import numpy as np
    arrays = {'analysis_to_raw': np.asarray(transform), 'px_mm': np.asarray(px_mm)}
    pairs = []
    for index, (pair, points) in enumerate((p, v) for p, v in locations.items() if p and p[0] != 'other'):
        patches = evidence.get(pair, [])
        example = next((patch for patch in patches if patch is not None), None)
        if example is None:
            continue
        key = str(index)
        arrays['points_' + key] = np.asarray(points)
        arrays['source_' + key] = np.stack([patch if patch is not None else np.zeros_like(example)
                                           for patch in patches])
        arrays['valid_' + key] = np.asarray([patch is not None for patch in patches])
        others = evidence.get(('other',) + tuple(pair))
        if others is not None and len(others) == len(patches):
            # Sleeve Transparant: het stukje uit de andere scan van het paar (zie seam_line_conflicts)
            arrays['other_' + key] = np.stack([q if q is not None and q.shape == example.shape else np.zeros_like(example)
                                              for q in others])
            arrays['othervalid_' + key] = np.asarray([q is not None and q.shape == example.shape for q in others])
        pairs.append({'key': key, 'sources': [names[pair[0]], names[pair[1]]]})
    arrays['pairs'] = np.asarray(json.dumps(pairs))
    np.savez_compressed(Path(folder) / 'source-evidence.npz', **arrays)
