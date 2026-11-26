#!/usr/bin/env python3
"""Run Sleeve from the original scans and compare exports without changing references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads(Path(__file__).with_suffix('.json').read_text())


def scan_key(paths):
    return tuple(sorted(Path(p).name for p in paths))


def read_result(path, groups):
    run = json.loads(path.read_text())
    expected = {}
    counts = {}
    for group in groups:
        part = group['part']
        counts[part] = counts.get(part, 0) + 1
        key = part if counts[part] == 1 else f"{part}_{counts[part]}"
        expected[(part, scan_key(group['files']))] = key
    parts = {}
    for row in run['results']:
        part = row['part']
        key = expected.get((part, scan_key(row.get('scans', []))), part)
        parts[key] = row
    return parts


def engine_signature():
    """Catch source edits during a repeat test before mixing engine versions."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'Resources').rglob('*.py')):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def signature(path):
    with Image.open(path) as image:
        w, h = image.size
        dpi = (image.info.get('dpi') or (None,))[0]
        rgba = image.convert('RGBA')
        alpha = rgba.getchannel('A')
        centre_alpha = alpha.getpixel((w // 2, h // 2))
        corners = [alpha.getpixel(p) for p in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]]
        alpha_digest = hashlib.sha256(alpha.tobytes()).hexdigest()
        rgba.thumbnail((360, 360), Image.Resampling.LANCZOS)
        thumb = Image.new('RGB', rgba.size, '#808080')
        thumb.paste(rgba, mask=rgba.getchannel('A'))
        thumb = np.asarray(thumb)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'pixels': [w, h], 'dpi': None if dpi is None else round(dpi, 2),
            'alpha_sha256': alpha_digest, 'hole_alpha': centre_alpha, 'corner_alpha': corners,
            'png_sha256': digest.hexdigest()}, thumb


def compare(file, reference):
    found, a = signature(file)
    want, b = signature(reference)
    x = cv2.resize(a, (360, 360), interpolation=cv2.INTER_AREA).astype(np.int16)
    y = cv2.resize(b, (360, 360), interpolation=cv2.INTER_AREA).astype(np.int16)
    same_png = found['png_sha256'] == want['png_sha256']
    same_alpha = found['pixels'] == want['pixels'] and found['alpha_sha256'] == want['alpha_sha256']
    result = {'same_png': same_png, 'same_alpha': same_alpha,
              'mean_preview_rgb_difference': round(float(np.abs(x - y).mean()), 3),
              'pixels': found['pixels'], 'reference_pixels': want['pixels'],
              'dpi': found['dpi'], 'reference_dpi': want['dpi']}
    if same_alpha and not same_png:
        # Quantify changes at native resolution when the masks agree. Use the
        # same background compositing in both images, including at the edge.
        with Image.open(file) as output_image, Image.open(reference) as archive_image:
            width, height = output_image.size
            visible_pixels = changed = 0
            largest = 0.0
            strongest = None
            tile = max(16, round((found['dpi'] or 300) / 25.4))
            stride = tile * 8
            for top in range(0, height, stride):
                bottom = min(height, top + stride)
                box = (0, top, width, bottom)
                first = np.asarray(output_image.crop(box).convert('RGBA'), dtype=np.int16)
                second = np.asarray(archive_image.crop(box).convert('RGBA'), dtype=np.int16)
                alpha = first[:, :, 3] / 255
                delta = np.abs(first[:, :, :3] - second[:, :, :3]).max(axis=2) * alpha
                visible_pixels += np.count_nonzero(alpha)
                changed += np.count_nonzero(delta >= 8)
                h, w = delta.shape
                cropped = delta[:h // tile * tile, :w // tile * tile]
                if cropped.size:
                    tiles = cropped.reshape(h // tile, tile, w // tile, tile).mean(axis=(1, 3))
                    row, col = np.unravel_index(tiles.argmax(), tiles.shape)
                    if tiles[row, col] > largest:
                        largest = float(tiles[row, col])
                        strongest = [int(col * tile), int(top + row * tile)]
            result.update(visible_rgb_delta_ge_8_pct=round(100 * changed / max(1, visible_pixels), 4),
                          peak_tile_delta=round(largest, 2), peak_tile_origin_px=strongest,
                          peak_tile_side_px=tile)
    return result


def local_copy(source, folder):
    """Bridge resolves links; it must resolve to a scratch copy, not the NAS."""
    dest = folder / source.name
    if dest.exists():
        if dest.stat().st_size != source.stat().st_size:
            raise ValueError(f'Source changed during test: {source}')
        return dest
    cloned = subprocess.run(['/bin/cp', '-c', '-p', str(source), str(dest)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if cloned.returncode:
        shutil.copy2(source, dest)
    return dest


def run_trial(case, source, target):
    scans = target / 'scans'
    output = target / 'out'
    state = target.parent / 'source-files'
    state.mkdir(parents=True, exist_ok=True)
    scans.mkdir(parents=True)
    output.mkdir()
    grouping_seconds = None
    if case.get('blind'):
        allowed = {'.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
        for original in sorted(source.iterdir()):
            if original.is_file() and original.suffix.lower() in allowed:
                (scans / original.name).symlink_to(local_copy(original.resolve(), state))
        guess = {
            'action': 'suggestGroups', 'format': 'vinyl',
            'discogs': f"https://www.discogs.com/release/{case['discogs']}",
            'slots': case.get('slots', ['front', 'back', 'sidea', 'sideb']),
            'inputs': [{'path': str(p)} for p in scans.iterdir()],
        }
        grouping_request = target / 'grouping-request.json'
        grouping_request.write_text(json.dumps(guess))
        started = time.monotonic()
        with (target / 'grouping.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'Resources/studio_bridge.py'), str(grouping_request)],
                           stdout=log, stderr=subprocess.STDOUT, check=False)
        grouping_seconds = round(time.monotonic() - started, 1)
        all_events = [json.loads(line) for line in (target / 'grouping.log').read_text().splitlines()
                      if line.startswith('{')]
        events = [event['groups'] for event in all_events if event.get('event') == 'groups']
        groups = events[-1] if events else []
        found_parts = {group['part'] for group in groups}
        grouped = {Path(f).name for g in groups for f in g['files']}
        complete = (set(guess['slots']).issubset(found_parts)
                    and 'unknown' not in found_parts and grouped == {f.name for f in scans.iterdir()})
        (target / 'grouping.json').write_text(json.dumps({'seconds': grouping_seconds, 'groups': groups,
                                                          'complete': complete}, indent=2))
        if not complete:
            return 0.0, 0, {}, grouping_seconds
    else:
        groups = list(case.get('groups') or json.loads((source / '.anouk-referenties.json').read_text())['scan_groups'])
        for part, pattern in case.get('extraGroups', {}).items():
            if not any(row['part'] == part for row in groups):
                groups.append({'part': part, 'files': [f.name for f in sorted(source.glob(pattern))]})
        if (source / '.sleeve-glas.json').is_file():
            shutil.copy2(source / '.sleeve-glas.json', state / '.sleeve-glas.json')
    if case.get('blind') or not (source / '.sleeve-glas.json').is_file():
        (state / '.sleeve-glas.json').unlink(missing_ok=True)
    rows = []
    for n, group in enumerate(groups):
        if not group['files']:
            raise ValueError(f"Missing scan group: {group['part']}")
        for name in group['files']:
            original = (source / Path(name).name).resolve(strict=True)
            link = scans / Path(name).name
            if not link.exists():
                link.symlink_to(local_copy(original, state))
            rows.append({'path': str(link), 'part': group['part'], 'group': f'g{n}',
                         'reference': False, 'partial': False, 'source': ''})
    request = {'action': 'scan', 'inputs': rows, 'discogs': f"https://www.discogs.com/release/{case['discogs']}",
               'useNames': True, 'useStand': True, 'format': 'vinyl', 'packaging': 'sleeve',
               'year': case['year'], 'output': str(output)}
    req = target / 'request.json'
    req.write_text(json.dumps(request, indent=2))
    env = dict(os.environ)
    env.pop('SLEEVE_STITCH_CACHE', None)
    t = time.monotonic()
    with (target / 'run.log').open('w') as log:
        proc = subprocess.run([sys.executable, str(ROOT / 'Resources/studio_bridge.py'), str(req)],
                              env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = round(time.monotonic() - t, 1)
    found = list(output.glob('*/Studio-verwerking.json'))
    return elapsed, proc.returncode, read_result(found[0], groups) if len(found) == 1 else {}, grouping_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scans', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=2, choices=(1, 2))
    parser.add_argument('--case-file', type=Path, help='add release descriptions to test another source folder')
    parser.add_argument('cases', nargs='+')
    parser.add_argument('--require-complete', action='store_true',
                        help='require every output, its native DPI, expected transparency and identical repeat pixels')
    args = parser.parse_args()
    cases = dict(CASES)
    if args.case_file:
        cases.update(json.loads(args.case_file.read_text()))
    unknown = set(args.cases) - cases.keys()
    if unknown:
        parser.error('Unknown case(s): ' + ', '.join(sorted(unknown)))
    dest = args.out.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    fingerprint = engine_signature()
    (dest / 'Engine.txt').write_text(fingerprint + '\n')
    report = {}
    for key in args.cases:
        case = cases[key]
        root = args.references / case['reference'] if case.get('reference') else None
        expected = case.get('files') or {
            k: case.get('filePrefix', '') + k + '.png' for k in case.get('slots', ('front', 'back', 'sidea', 'sideb'))}
        first = {}
        trials = []
        report[key] = trials
        for i in range(1, args.repeat + 1):
            if engine_signature() != fingerprint:
                raise RuntimeError('Sleeve code changed during this run; use a separate test folder for the new version')
            print(f'START {key} {i}/{args.repeat}', flush=True)
            target = dest / key / str(i)
            elapsed, status, parts, grouping_seconds = run_trial(case, args.scans / case['source'], target)
            if engine_signature() != fingerprint:
                raise RuntimeError('Sleeve code changed during this run; this result cannot be compared')
            checked = time.monotonic()
            trial = {'render_seconds': elapsed, 'grouping_seconds': grouping_seconds, 'exit_code': status, 'parts': {}}
            for part, filename in expected.items():
                row = parts.get(part, {})
                path = Path(row['path']) if row.get('path') else None
                item = {k: row.get(k) for k in ('verdict', 'why', 'dpi', 'size_mm', 'upright', 'rim_sources')}
                item['output'] = str(path) if path else None
                item['reference'] = str(root / filename) if root else None
                if path and path.is_file():
                    sig, _ = signature(path)
                    item['image'] = sig
                    item['source_dpi_preserved'] = sig['dpi'] == case['dpi']
                    if part.startswith(('side', 'label')):
                        item['transparent_cut'] = sig['hole_alpha'] == 0 and not any(sig['corner_alpha'])
                    if root and (root / filename).is_file():
                        item['archive_comparison'] = compare(path, root / filename)
                    if i == 2 and part in first:
                        item['repeat_comparison'] = compare(path, first[part])
                    if i == 1:
                        first[part] = path
                else:
                    item['finding'] = 'missing output'
                trial['parts'][part] = item
            trial['check_seconds'] = round(time.monotonic() - checked, 1)
            trials.append(trial)
            (dest / 'Report.json').write_text(json.dumps(report, indent=2))
            print(f'DONE {key} {i}: {elapsed}s render, {trial["check_seconds"]}s check; '
                  + ', '.join(f'{p} {r["verdict"] or "MISSING"}' for p, r in trial['parts'].items()), flush=True)

    if args.require_complete:
        complete = all(trial['exit_code'] == 0 and all(
            item.get('image') and item.get('source_dpi_preserved')
            and item.get('transparent_cut', True)
            and (i == 0 or item.get('repeat_comparison', {}).get('same_png'))
            for item in trial['parts'].values())
            for trials in report.values() for i, trial in enumerate(trials))
        if not complete:
            raise SystemExit('An export is missing or differs in DPI, transparency or repeat output; see Report.json')


if __name__ == '__main__':
    main()
