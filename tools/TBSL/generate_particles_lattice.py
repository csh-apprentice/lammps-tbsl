#!/usr/bin/env python3
import numpy as np
import argparse, math, time, os, shutil, struct, heapq, multiprocessing

# ---------- optional progress bar ----------
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))

# ---------- target radial pdf over s = r/Rb ----------
def pdf_s(s, xa, xb):
    return np.clip(xa * s**4 + xb * s**2, 0.0, None)

# ---------- ensemble helpers (your constants) ----------
def compute_scalers(N_ensem):
    pi = np.pi
    boltzmann = 1.380649e-23
    m = 39.95          # g/mol
    d = 3.66           # Angstrom
    initialR = 4.5e-6  # m
    Tinfty = 300       # K
    Pinfty = 101325    # Pa
    V = (4/3) * pi * (initialR**3)
    N_real = (Pinfty * V) / (boltzmann * Tinfty)
    g = N_real / float(N_ensem)
    rscale = g**(1.0/3.0)
    d_ensem = rscale * d
    m_ensem = g * m
    return d_ensem, m_ensem, g

# ---------- legacy full lattice (kept for small-N testing) ----------
def lattice_points_in_sphere(R_in, h):
    maxn = int(np.floor(R_in / h))
    coords = np.arange(-maxn, maxn + 1, dtype=np.int32)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    pts = np.stack([(X * h), (Y * h), (Z * h)], axis=-1).reshape(-1, 3)
    r2 = np.sum(pts**2, axis=1)
    keep = r2 <= (R_in * R_in + 1e-12)
    return pts[keep], r2[keep]

def weights_from_pdf(pts, r2, Rb, xa, xb):
    r = np.sqrt(r2, dtype=np.float64)
    s = r / float(Rb)
    w_num = pdf_s(s, xa, xb)
    eps = 1e-30
    base = w_num / np.maximum(r2, eps)
    base[s > 1.0] = 0.0
    return base

def choose_points(pts, weights, N, seed):
    wsum = weights.sum()
    if wsum <= 0: raise RuntimeError("All weights are zero; check xa/xb/R_in.")
    prob = weights / wsum
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=N, replace=False, p=prob)
    return pts[idx]

# ---------- serial streaming (fixed heap logic; correct distribution) ----------
def lattice_block_iter(R_in, h, block):
    maxn = int(np.floor(R_in / h))
    xs_all = np.arange(-maxn, maxn + 1, dtype=np.int32)
    ys_all = np.arange(-maxn, maxn + 1, dtype=np.int32)
    zs_all = np.arange(-maxn, maxn + 1, dtype=np.int32)
    for i0 in range(0, xs_all.size, block):
        xs = xs_all[i0:i0 + block]
        X, Y, Z = np.meshgrid(xs, ys_all, zs_all, indexing='ij')
        pts = np.stack([(X * h), (Y * h), (Z * h)], axis=-1).reshape(-1, 3)
        r2 = np.sum(pts**2, axis=1)
        keep = r2 <= (R_in * R_in + 1e-12)
        if np.any(keep):
            yield pts[keep], r2[keep]

def choose_points_stream(Rb, xa, xb, R_in, h, N, seed, block=96, show_progress=True):
    rng = np.random.default_rng(seed)
    heap = []  # store (-k, x, y, z). We keep N smallest k => largest -k.
    xs_count = int(np.floor(R_in / h)) * 2 + 1
    slabs = (xs_count + block - 1) // block
    pbar = tqdm(total=slabs, desc="Streaming slabs (serial)", unit="slab", ncols=80) if show_progress else None

    for pts_blk, r2_blk in lattice_block_iter(R_in, h, block):
        r = np.sqrt(r2_blk, dtype=np.float64)
        s = r / float(Rb)
        wnum = pdf_s(s, xa, xb)
        w = wnum / np.maximum(r2_blk, 1e-30)
        w[s > 1.0] = 0.0
        for p, ww in zip(pts_blk, w):
            if ww <= 0.0: continue
            k = -math.log(rng.random()) / float(ww)  # smaller is better
            item = (-k, float(p[0]), float(p[1]), float(p[2]))
            if len(heap) < N:
                heapq.heappush(heap, item)
            else:
                # Correct: replace worst (smallest -k) only if new -k is larger (i.e., k is smaller)
                if item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
        if pbar: pbar.update(1)

    if pbar: pbar.close()
    chosen = np.empty((min(N, len(heap)), 3), dtype=np.float64)
    for i, (_, x, y, z) in enumerate(heap):
        chosen[i] = (x, y, z)
    return chosen

# ---------- helpers for parallel sharded exact selection ----------
def _iter_yz_tiles(maxn, tile):
    ys_all = np.arange(-maxn, maxn + 1, dtype=np.int32)
    zs_all = np.arange(-maxn, maxn + 1, dtype=np.int32)
    for y0 in range(0, ys_all.size, tile):
        ys = ys_all[y0:y0+tile]
        for z0 in range(0, zs_all.size, tile):
            zs = zs_all[z0:z0+tile]
            yield ys, zs

def _make_slab_chunks(R_in, h, block, W, seed):
    xs_count = int(np.floor(R_in / h)) * 2 + 1
    starts = np.arange(0, xs_count, block, dtype=np.int64)
    # randomize assignment to de-skew workers
    rng = np.random.default_rng(seed)
    rng.shuffle(starts)
    return [s for s in np.array_split(starts, W) if s.size]

def _worker_pilot_topk(args):
    (xs_range, Rb, xa, xb, R_in, h, block, tile, Kpilot, seed) = args
    rng = np.random.default_rng(seed)
    maxn = int(np.floor(R_in / h))
    xs_all = np.arange(-maxn, maxn + 1, dtype=np.int32)
    heap = []  # store (-k,)
    for i0 in xs_range:
        xs = xs_all[i0:i0+block]
        if xs.size == 0: continue
        xs_f = xs.astype(np.float64) * h
        x2 = xs_f**2
        for ys, zs in _iter_yz_tiles(maxn, tile):
            ysf = ys.astype(np.float64) * h
            zsf = zs.astype(np.float64) * h
            r2 = x2[:,None,None] + (ysf**2)[None,:,None] + (zsf**2)[None,None,:]
            keep = r2 <= (R_in*R_in + 1e-12)
            if not keep.any(): continue
            r = np.sqrt(r2, where=keep, out=np.zeros_like(r2))
            s = np.divide(r, float(Rb), where=keep, out=np.zeros_like(r))
            wnum = np.clip(xa*s**4 + xb*s**2, 0.0, None)
            w = np.divide(wnum, np.maximum(r2,1e-30), where=keep, out=np.zeros_like(r2))
            pos = (keep & (w > 0))
            cnt = int(pos.sum())
            if cnt == 0: continue
            U = rng.random(cnt)
            keys = -np.log(U) / w[pos]
            for k in keys:
                item = (-float(k),)
                if len(heap) < Kpilot:
                    heapq.heappush(heap, item)
                else:
                    if item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
    return np.array([-t[0] for t in heap], dtype=np.float64)

def _worker_emit_leq_tau(args):
    (xs_range, Rb, xa, xb, R_in, h, block, tile, tau, seed, shard_path) = args
    rng = np.random.default_rng(seed)
    maxn = int(np.floor(R_in / h))
    xs_all = np.arange(-maxn, maxn + 1, dtype=np.int32)
    with open(shard_path, "wb") as f:
        for i0 in xs_range:
            xs = xs_all[i0:i0+block]
            if xs.size == 0: continue
            xs_f = xs.astype(np.float64) * h
            x2 = xs_f**2
            for ys, zs in _iter_yz_tiles(maxn, tile):
                ysf = ys.astype(np.float64) * h
                zsf = zs.astype(np.float64) * h
                r2 = x2[:,None,None] + (ysf**2)[None,:,None] + (zsf**2)[None,None,:]
                keep = r2 <= (R_in*R_in + 1e-12)
                if not keep.any(): continue
                r = np.sqrt(r2, where=keep, out=np.zeros_like(r2))
                s = np.divide(r, float(Rb), where=keep, out=np.zeros_like(r))
                wnum = np.clip(xa*s**4 + xb*s**2, 0.0, None)
                w = np.divide(wnum, np.maximum(r2,1e-30), where=keep, out=np.zeros_like(r2))
                pos = (keep & (w > 0))
                cnt = int(pos.sum())
                if cnt == 0: continue
                U = rng.random(cnt)
                keys = -np.log(U) / w[pos]
                ix, iy, iz = np.nonzero(pos)
                x = xs_f[ix].astype(np.float32)
                y = (ysf[iy]).astype(np.float32)
                z = (zsf[iz]).astype(np.float32)
                mask = keys <= tau
                # write <k:float64, x/y/z:float32>
                for k, xx, yy, zz in zip(keys[mask], x[mask], y[mask], z[mask]):
                    f.write(struct.pack("<dfff", float(k), float(xx), float(yy), float(zz)))
    return shard_path

def _sort_shard_desc(shard_path):
    dt = np.dtype([("k","<f8"),("x","<f4"),("y","<f4"),("z","<f4")])
    arr = np.fromfile(shard_path, dtype=dt)
    sorted_path = shard_path + ".sorted"
    if arr.size:
        arr = arr[np.argsort(arr["k"])[::-1]]
        arr.tofile(sorted_path)
    else:
        open(sorted_path, "wb").close()
    try: os.remove(shard_path)
    except: pass
    return sorted_path, int(arr.size)

def _merge_topN_to_coords_bin(sorted_paths, N, out_bin_path):
    rec = "<dfff"; rec_sz = struct.calcsize(rec)
    files = [open(p, "rb") for p in sorted_paths]

    def read_block(f, nrec):
        data = f.read(nrec * rec_sz)
        if not data: return None
        return np.frombuffer(data, dtype=[("k","<f8"),("x","<f4"),("y","<f4"),("z","<f4")])

    # seed heap with first item from each shard
    bufN = 100_000
    bufs = [read_block(f, bufN) for f in files]
    idxs = [0] * len(files)
    heap = []
    for j, b in enumerate(bufs):
        if b is not None and b.size:
            k,x,y,z = b["k"][0], b["x"][0], b["y"][0], b["z"][0]
            heap.append((-k, j, float(x), float(y), float(z)))
    heapq.heapify(heap)

    with open(out_bin_path, "wb") as outf, tqdm(total=N, desc="Merge-select top-N", unit="atom", ncols=80) as pbar:
        count = 0
        while heap and count < N:
            negk, j, x, y, z = heapq.heappop(heap)
            outf.write(struct.pack("<fff", x, y, z))  # coords only
            count += 1
            pbar.update(1)
            idxs[j] += 1
            b = bufs[j]
            if b is None: continue
            if idxs[j] >= b.size:
                bufs[j] = read_block(files[j], bufN); idxs[j] = 0; b = bufs[j]
                if b is None or b.size == 0: 
                    continue
            k,x,y,z = b["k"][idxs[j]], b["x"][idxs[j]], b["y"][idxs[j]], b["z"][idxs[j]]
            heapq.heappush(heap, (-k, j, float(x), float(y), float(z)))
    for f in files:
        try: f.close()
        except: pass

# ---------- LAMMPS writer (parallel shards -> concat) ----------
def _split_ranges(N, parts):
    parts = max(1, min(parts, N))
    base = N // parts
    extra = N % parts
    ranges = []
    start = 0
    for i in range(parts):
        n = base + (1 if i < extra else 0)
        end = start + n
        if n > 0:
            ranges.append((start, end))
        start = end
    return ranges

def _write_atoms_chunk_from_memmap(args):
    (coords_mmap_path, N, start, end, tmpfile) = args
    arr = np.memmap(coords_mmap_path, dtype="<f4", mode="r", shape=(N, 3))
    with open(tmpfile, "w") as f:
        lines = []
        la = lines.append
        for i in range(start, end):
            x, y, z = arr[i, 0], arr[i, 1], arr[i, 2]
            la(f"{i+1} 0 1 0 {x:.6f} {y:.6f} {z:.6f}\n")
            if len(lines) >= 200_000:
                f.writelines(lines); lines.clear()
        if lines: f.writelines(lines)
    del arr
    return tmpfile

def _write_vel_chunk(args):
    (start, end, tmpfile) = args
    with open(tmpfile, "w") as f:
        lines = []
        la = lines.append
        for i in range(start, end):
            la(f"{i+1} 0.0 0.0 0.0\n")
            if len(lines) >= 400_000:
                f.writelines(lines); lines.clear()
        if lines: f.writelines(lines)
    return tmpfile

def _concat_files(out_fh, paths, desc):
    for p in tqdm(paths, desc=desc, unit="part", ncols=80):
        with open(p, "r") as fh:
            shutil.copyfileobj(fh, out_fh, length=1024*1024)
        try: os.remove(p)
        except: pass

def write_lammps_from_coords_bin(path, coords_bin_path, N, m_ensem, R_header, workers=1):
    tmpdir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(tmpdir, exist_ok=True)

    size_bytes = os.path.getsize(coords_bin_path)
    expected = N * 3 * 4
    if size_bytes != expected:
        raise RuntimeError(f"[coords.bin] size {size_bytes} != expected {expected}")

    with open(path, "w") as f:
        f.write("LAMMPS data file via lattice radial sampling (min-dist enforced)\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"1 atom types\n\n")
        f.write(f"{-R_header:.15f} {R_header:.15f} xlo xhi\n")
        f.write(f"{-R_header:.15f} {R_header:.15f} ylo yhi\n")
        f.write(f"{-R_header:.15f} {R_header:.15f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write(f"1 {m_ensem:.6f}\n\n")
        f.write("Atoms # full\n\n")

    parts = _split_ranges(N, max(1, workers))
    atom_tmp_paths = [os.path.join(tmpdir, f".atoms_part_{i:04d}.tmp") for i in range(len(parts))]
    if workers <= 1:
        _write_atoms_chunk_from_memmap((coords_bin_path, N, 0, N, atom_tmp_paths[0]))
        with open(path, "a") as f:
            _concat_files(f, [atom_tmp_paths[0]], desc="Concat atoms")
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            list(tqdm(
                pool.imap_unordered(
                    _write_atoms_chunk_from_memmap,
                    [(coords_bin_path, N, s, e, atom_tmp_paths[i]) for i, (s, e) in enumerate(parts)]
                ),
                total=len(parts),
                desc="Atoms shards",
                unit="part",
                ncols=80
            ))
        with open(path, "a") as f:
            _concat_files(f, [atom_tmp_paths[i] for i in range(len(parts))], desc="Concat atoms")

    with open(path, "a") as f:
        f.write("\nVelocities\n\n")

    vel_tmp_paths = [os.path.join(tmpdir, f".vel_part_{i:04d}.tmp") for i in range(len(parts))]
    if workers <= 1:
        _write_vel_chunk((0, N, vel_tmp_paths[0]))
        with open(path, "a") as f:
            _concat_files(f, [vel_tmp_paths[0]], desc="Concat vel")
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            list(tqdm(
                pool.imap_unordered(
                    _write_vel_chunk,
                    [(s, e, vel_tmp_paths[i]) for i, (s, e) in enumerate(parts)]
                ),
                total=len(parts),
                desc="Vel shards",
                unit="part",
                ncols=80
            ))
        with open(path, "a") as f:
            _concat_files(f, [vel_tmp_paths[i] for i in range(len(parts))], desc="Concat vel")

# ---------- parallel exact selection (two-pass threshold) ----------
def choose_points_stream_parallel(Rb, xa, xb, R_in, h, N, seed, block=96, num_cpus=None,
                                  tile=64, pilot_factor=2.0, tau_slack=1.02,
                                  out_dir=".", show_progress=True):
    W = (os.cpu_count() or 1) if (num_cpus is None or num_cpus == 0) else max(1, int(num_cpus))
    slab_chunks = _make_slab_chunks(R_in, h, block, W, seed+12345)
    W = len(slab_chunks)
    print(f"[Parallel] Workers={W}, block={block}, tile={tile}")

    seed_seq = np.random.SeedSequence(seed)
    seeds = seed_seq.spawn(W)

    # ---- PASS A (pilot) ----
    Kpilot = max(1, int(math.ceil(N / W) * pilot_factor))
    pilot_tasks = [(slab_chunks[i], Rb, xa, xb, R_in, h, block, int(tile), Kpilot, seeds[i]) for i in range(W)]
    if W == 1:
        pilot_keys = [_worker_pilot_topk(pilot_tasks[0])]
    else:
        with multiprocessing.Pool(W) as pool:
            it = pool.imap_unordered(_worker_pilot_topk, pilot_tasks)
            pilot_keys = list(tqdm(it, total=W, desc="Pilot", unit="worker", ncols=80)) if show_progress else list(it)
    pilot_keys = np.concatenate(pilot_keys) if len(pilot_keys) else np.empty((0,), dtype=np.float64)
    if pilot_keys.size == 0:
        raise RuntimeError("Pilot produced zero keys; check parameters.")
    take_idx = min(N-1, pilot_keys.size-1)
    tau = np.partition(pilot_keys, take_idx)[take_idx] * float(tau_slack)
    print(f"[Pilot] tau={tau:.6e} (keys={pilot_keys.size:,}, slack={tau_slack})")

    # ---- PASS B (select) ----
    shard_paths = [os.path.join(out_dir, f".sel_shard_{i:04d}.bin") for i in range(W)]
    emit_tasks = [(slab_chunks[i], Rb, xa, xb, R_in, h, block, int(tile), float(tau), seeds[i], shard_paths[i]) for i in range(W)]
    if W == 1:
        _worker_emit_leq_tau(emit_tasks[0])
    else:
        with multiprocessing.Pool(W) as pool:
            it = pool.imap_unordered(_worker_emit_leq_tau, emit_tasks)
            list(tqdm(it, total=W, desc="Select", unit="shard", ncols=80)) if show_progress else list(it)

    # sort + merge
    sorted_paths, kept_total = [], 0
    for sp in tqdm(sorted(shard_paths), desc="Sort shards", unit="shard", ncols=80):
        sp2, kept = _sort_shard_desc(sp)
        sorted_paths.append(sp2); kept_total += kept
    if kept_total < N:
        for sp in sorted_paths:
            try: os.remove(sp)
            except: pass
        raise RuntimeError(f"After select, kept {kept_total:,} < N={N:,}. Increase slack or pilot_factor.")

    coords_bin = os.path.join(out_dir, "._coords.bin")
    _merge_topN_to_coords_bin(sorted_paths, N, coords_bin)
    for sp in sorted_paths:
        try: os.remove(sp)
        except: pass

    # load coords_bin → (N,3) float32 → float64 for downstream write formatting
    arr = np.memmap(coords_bin, dtype="<f4", mode="r")
    if arr.size != N*3:
        raise RuntimeError(f"coords.bin floats {arr.size} != {N*3}")
    xyz = np.asarray(arr).reshape(N,3).astype(np.float64)
    del arr
    try: os.remove(coords_bin)
    except: pass
    return xyz

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10000, help="Target number of atoms")
    ap.add_argument("--Rb", type=float, default=3.086969205287614e4, help="Outer sphere radius")
    ap.add_argument("--xa", type=float, default=-1.5781)
    ap.add_argument("--xb", type=float, default= 3.9468)
    ap.add_argument("--min_dist_factor", type=float, default=2.0, help="min_dist = factor * d_ensem")
    ap.add_argument("--avoid_boundary", action="store_true", help="Use R_in = Rb - d_ensem")
    ap.add_argument("--rin_override", type=float, default=None, help="Explicit R_in (overrides avoid_boundary)")
    ap.add_argument("--header_use_rin", action="store_true", help="Use ±R_in in header bounds when applicable")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--output", type=str, default="initialize_lattice_radial_1e4.lammpsdata")
    ap.add_argument("--block", type=int, default=96, help="x-slab size (memory control for enumeration)")
    ap.add_argument("--tile", type=int, default=64, help="YZ tile size (memory control within a slab)")
    ap.add_argument("--legacy_select", action="store_true", help="Legacy full-array selection (small N only).")
    ap.add_argument("--parallel", action="store_true", help="Parallel exact selection (recommended for large N).")
    ap.add_argument("--cpus", type=int, default=0, help="Workers for --parallel (0 = all CPUs).")
    ap.add_argument("--pilot_factor", type=float, default=2.0, help="Pilot top-K safety multiplier (>=1).")
    ap.add_argument("--tau_slack", type=float, default=1.02, help="Slack factor on tau to ensure ≥N in pass B.")
    args = ap.parse_args()

    t0 = time.time()
    N = int(args.N)
    Rb = float(args.Rb)
    xa, xb = float(args.xa), float(args.xb)

    d_ensem, m_ensem, g = compute_scalers(N)
    h = float(args.min_dist_factor) * d_ensem

    if args.rin_override is not None:
        R_in = float(args.rin_override)
        if not (0.0 < R_in <= Rb):
            raise ValueError("--rin_override must satisfy 0 < R_in <= Rb")
        print(f"[Boundary] Using explicit R_in = {R_in:.6f} (Rb={Rb:.6f})")
    elif args.avoid_boundary:
        R_in = Rb - d_ensem
        print(f"[Boundary] Using R_in = Rb - d_ensem = {R_in:.6f}")
    else:
        R_in = Rb
        print(f"[Boundary] Using full R_in = Rb = {R_in:.6f}")

    print(f"[Lattice] min_dist h = {h:.6f} (factor {args.min_dist_factor} * d_ensem {d_ensem:.6f})")

    # ---------- selection ----------
    if args.legacy_select:
        print("[Selection] LEGACY full-array (may OOM for large N).")
        pts, r2 = lattice_points_in_sphere(R_in, h)
        M = pts.shape[0]
        if M < N:
            raise RuntimeError(f"Not enough lattice sites ({M}) to place N={N}. Adjust min_dist or R_in.")
        w = weights_from_pdf(pts, r2, Rb, xa, xb)
        if (w.sum() <= 0) or np.all(w == 0):
            raise RuntimeError("All weights zero; check xa/xb and R_in/Rb.")
        chosen = choose_points(pts, w, N, args.seed)

    elif args.parallel:
        print(f"[Selection] PARALLEL exact selection (two-pass threshold).")
        chosen = choose_points_stream_parallel(
            Rb, xa, xb, R_in, h, N, args.seed,
            block=args.block, num_cpus=args.cpus, tile=args.tile,
            pilot_factor=args.pilot_factor, tau_slack=args.tau_slack,
            out_dir=os.path.dirname(os.path.abspath(args.output)) or ".",
            show_progress=True
        )
        M = None
    else:
        print("[Selection] SERIAL streaming (fixed heap logic).")
        chosen = choose_points_stream(
            Rb, xa, xb, R_in, h, N, args.seed, block=args.block, show_progress=True
        )
        M = None

    # ---------- write ----------
    R_header = R_in if (args.header_use_rin and (args.avoid_boundary or args.rin_override is not None)) else Rb

    # Write via temporary coords.bin (already produced in parallel path). If serial/legacy, make one now.
    if not args.parallel:
        coords_bin = os.path.join(os.path.dirname(os.path.abspath(args.output)) or ".", "._coords.bin")
        with open(coords_bin, "wb") as f:
            # store float32 xyz to reduce I/O
            f.write(np.asarray(chosen, dtype="<f4").tobytes())
    else:
        coords_bin = os.path.join(os.path.dirname(os.path.abspath(args.output)) or ".", "._coords.bin")
        # parallel path already produced & removed it inside choose_points_stream_parallel; recreate from chosen:
        with open(coords_bin, "wb") as f:
            f.write(np.asarray(chosen, dtype="<f4").tobytes())

    workers = (os.cpu_count() or 1) if (args.cpus is None or args.cpus == 0) else max(1, int(args.cpus))
    print(f"[Write] Emitting LAMMPS data with {workers} writer(s)...")
    write_lammps_from_coords_bin(args.output, coords_bin, chosen.shape[0], m_ensem, R_header, workers=workers)
    try: os.remove(coords_bin)
    except: pass

    dt = time.time() - t0
    sites_msg = f"lattice sites (legacy)" if args.legacy_select else "streamed lattice"
    print(f"[Done] Wrote {args.output} with {chosen.shape[0]:,} atoms, {sites_msg}, elapsed {dt:.2f}s")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
