"""Read historical header tables and complete LAMMPS thermo/chunk outputs.

Compatibility functions keep the return conventions used by the paper scripts.
Malformed or truncated input is rejected; no missing frame is silently plotted.
"""
from pathlib import Path
import re

import numpy as np


def parse_global_output(filename):
    """Read a header-only table or a log with identical repeated thermo headers.

    Exact duplicate records are removed. Conflicting duplicate steps need an
    explicit restart merge; CPU/time columns are never silently rewritten.
    """
    headers, records = None, []
    in_table = False
    candidate = None
    with Path(filename).open() as stream:
        for line_number, raw in enumerate(stream, 1):
            parts = raw.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            if parts[0] in ('Step', 'Time'):
                if headers is not None and parts != headers:
                    raise ValueError(f'{filename}:{line_number}: thermo columns changed')
                headers, in_table = parts, True
                continue
            if not in_table:
                # Standalone radial/theory tables use headings such as 'frac_r
                # Temp'. Accept a generic heading only when followed by a numeric
                # row of the same width; ordinary log preambles are skipped.
                if all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_/%.:-]*', word) for word in parts):
                    candidate = parts
                    continue
                if candidate is None:
                    continue
                try:
                    probe = [float(v) for v in parts]
                except ValueError:
                    candidate = None
                    continue
                if len(probe) != len(candidate):
                    candidate = None
                    continue
                headers, in_table = candidate, True
            try:
                row = [float(v) for v in parts]
            except ValueError:
                continue  # Warnings and timing summaries are not samples.
            if len(row) != len(headers):
                raise ValueError(f'{filename}:{line_number}: incomplete table row')
            if not np.all(np.isfinite(row)):
                raise ValueError(f'{filename}:{line_number}: non-finite table row')
            if records and 'Step' in headers and row[headers.index('Step')] == records[-1][headers.index('Step')]:
                if row != records[-1]:
                    raise ValueError(f'{filename}:{line_number}: conflicting duplicate step')
                continue
            records.append(row)
    if headers is None or not records:
        raise ValueError(f'No numeric table found in {filename}')
    values = np.asarray(records)
    if 'Step' in headers and np.any(np.diff(values[:, headers.index('Step')]) <= 0):
        raise ValueError(f'{filename}: steps are not strictly increasing; merge restarts first')
    return {h: values[:, i].tolist() for i, h in enumerate(headers)}, headers


def iter_ave_chunk_blocks(filename):
    """Yield (step, column names, rows), retaining per-block coordinates."""
    headers = None
    previous_step = -np.inf
    with Path(filename).open() as stream:
        for raw in stream:
            parts = raw.strip().split()
            if not parts:
                continue
            if parts[0] == '#':
                if len(parts) > 1 and parts[1] == 'Chunk':
                    current = parts[1:]
                    if headers is not None and current != headers:
                        raise ValueError(f'{filename}: chunk columns changed')
                    headers = current
                continue
            if headers is None or len(parts) != 3:
                raise ValueError(f'{filename}: missing chunk column or block header')
            step_value, count_value, total_count = map(float, parts)
            step, count = int(step_value), int(count_value)
            if step != step_value or count != count_value or count < 1 or step <= previous_step:
                raise ValueError(f'{filename}: invalid or non-increasing block at {step_value}')
            rows = []
            for _ in range(count):
                line = stream.readline()
                if not line:
                    raise ValueError(f'{filename}: truncated chunk block at step {step}')
                row = [float(v) for v in line.split()]
                if len(row) != len(headers) or not np.all(np.isfinite(row)):
                    raise ValueError(f'{filename}: invalid chunk row at step {step}')
                rows.append(row)
            values = np.asarray(rows)
            if not np.array_equal(values[:, 0], np.arange(1, count + 1)):
                raise ValueError(f'{filename}: chunk indices must be consecutive')
            if np.any(values[:, 2] < 0) or not np.isclose(values[:, 2].sum(), total_count, rtol=2e-5, atol=1e-6):
                raise ValueError(f'{filename}: inconsistent particle counts at step {step}')
            previous_step = step
            yield step, headers, values


def parse_ave_chunk_output(filename, attr_name='temp'):
    """Read one field on a fixed radial grid; reject adaptive grids explicitly.

    Use iter_ave_chunk_blocks for intentionally changing radial coordinates.
    """
    steps, values = [], []
    radii, available = None, None
    for step, headers, block in iter_ave_chunk_blocks(filename):
        available = headers[2:]
        if attr_name not in available:
            raise ValueError(f'Attribute {attr_name!r} not found. Available: {available}')
        coordinates = block[:, 1]
        if radii is None:
            radii = coordinates.copy()
        elif coordinates.shape != radii.shape or not np.allclose(coordinates, radii, rtol=1e-10, atol=1e-10):
            raise ValueError(f'{filename}: radial grid changes; use iter_ave_chunk_blocks')
        steps.append(step)
        values.append(block[:, headers.index(attr_name)])
    if not steps:
        raise ValueError(f'No chunk blocks found in {filename}')
    return np.asarray(steps), radii, np.asarray(values), available


def compute_pressure_from_virial_density(virial_vals, number_density_vals, scalefactor):
    """EP stress trace (atm*A^3) times EP number density gives pressure in atm.

    The paper recipes pass scalefactor=1 for EP stress and EP number density.
    """
    return -(np.asarray(virial_vals) * np.asarray(number_density_vals) * scalefactor) / 3.0
