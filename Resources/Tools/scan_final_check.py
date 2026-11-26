"""Recheck source observations after the final crop, orientation and detail merge."""
import json
from pathlib import Path
import numpy as np

PREFIX = 'cover line changes at a seam:'


def quarter_turn_matrix(shape, clockwise):
    height, width = shape[:2]
    if clockwise is None or abs(clockwise/90-round(clockwise/90)) > 1e-6:
        raise ValueError('No exact quarter turn geometry')
    quarter = round(clockwise/90) % 4
    return np.asarray(([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       [[0, -1, height-1], [1, 0, 0], [0, 0, 1]],
                       [[-1, 0, width-1], [0, -1, height-1], [0, 0, 1]],
                       [[0, 1, 0], [-1, 0, width-1], [0, 0, 1]])[quarter], float)


def recheck(image, folder, raw_to_final, why, detail=None):
    """Replace only supported seam warnings; absent evidence preserves doubt."""
    from scan_stitch import unresolved_seam_points
    from scan_window_detail import verified_final_points
    file = Path(folder)/'source-evidence.npz'
    if not file.is_file():
        return why, None
    notes, replacements, checked, restored, diagnostics = [], {}, 0, 0, []
    with np.load(file, allow_pickle=False) as data:
        mapping = raw_to_final @ np.vstack([data['analysis_to_raw'], [0, 0, 1]])
        mapping = mapping[:2]
        ppm = float(data['px_mm'])
        density = max(np.linalg.norm(mapping[:, 0]), np.linalg.norm(mapping[:, 1]))
        radius = max(7, round(2.4*ppm) | 1)//2 + max(3, round(.45*ppm) | 1) + max(1, round(.32*ppm))
        for pair in json.loads(str(data['pairs'])):
            key = pair['key'];points = data['points_'+key]
            sources = data['source_'+key];valid = data['valid_'+key]
            evidence = [patch if ok else None for patch, ok in zip(sources, valid)]
            if 'other_'+key in data.files:
                # Sleeve Transparant: ook de andere scan van het paar telt als bewijs
                evidence = [(patch, other if ok2 else None) for patch, other, ok2
                            in zip(evidence, data['other_'+key], data['othervalid_'+key])]
            centers = np.c_[points[:, :2], np.ones(len(points))] @ mapping.T
            changed = verified_final_points(image, detail, centers, np.full(len(points), radius*density)) if detail else np.zeros(len(points), bool)
            remaining = ~changed
            pending = []
            count = unresolved_seam_points(image, points[remaining], mapping, ppm, pending,
                                           evidence=[e for e, keep in zip(evidence, remaining) if keep])
            for x, y, reason in pending:
                diagnostics.append((mapping@[x, y, 1], reason, *pair['sources']))
            names = ' / '.join(pair['sources'])
            replacements[names] = count
            checked += len(points);restored += int(changed.sum())
            notes.append({'sources': pair['sources'], 'checked': len(points),
                          'confirmed_from_detail': int(changed.sum()), 'unconfirmed': count})
    updated, seen = [], set()
    for note in why:
        if note.startswith(PREFIX):
            names = note[len(PREFIX):].strip().rsplit(' (', 1)[0]
            if names in replacements:
                seen.add(names)
                if replacements[names]:
                    updated.append(f'{PREFIX} {names} ({replacements[names]} unconfirmed locations)')
                continue
        updated.append(note)
    for names, count in replacements.items():
        if names not in seen and count:
            updated.append(f'{PREFIX} {names} ({count} unconfirmed locations)')
    report = {'checked_observations': checked, 'confirmed_from_detail': restored, 'pairs': notes,
              'raw_to_final_px': np.asarray(raw_to_final).tolist()}
    if diagnostics:
        from scan_seam_probe import write_seam_diagnostics
        report['details'] = str(write_seam_diagnostics(Path(folder)/'final', 'final', image,
                                                      diagnostics, ppm*density*25.4))
    return updated, report
