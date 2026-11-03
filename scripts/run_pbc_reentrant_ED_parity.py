"""ED Gamma0 scans of the random AFM Ising ring in a longitudinal field (PBC, L16 vs L17).

Wall-parity test of the reentrant section: on a ring the number of parallel bonds has the
parity of L, so PBC L16 has walls in pairs (like OBC L17) and PBC L17 has a forced odd number
(like OBC L16). Plan, predictions and frozen rules:
documentation/PLAN_PBC_ED_PARITY_L16_L17_20261023.md.

H = sum_b J_b Z_b Z_{b+1} - Gamma0 sum_i p_i X_i - h sum_i Z_i, with b = L-1 closing the ring.
In the fixed_mean ensemble J does not depend on Gamma0 and Gamma_i = Gamma0 * p_i exactly, so
the diagonal part and X_p are built once per realization (bit flips, as in
scripts/ed_reentrant_average_pbc_obc_review_v1.py) and checked against
clean_tfim_tn.ed.random_ising.build_random_ising_hamiltonian (--check).

Per seed the scan runs upwards in Gamma0 with eigsh(k=2) warm-started from the previous ground
state. Gamma0 = 0 is solved exactly on the diagonal: the state is the equal-weight mixture of
the degenerate classical minima, and the degeneracy is stored. Saved per point: raw <Z_i Z_j>
(Czz) of the two lowest states, C = Cstag with the circular phase (-1)^d, d = min(|i-j|, L-|i-j|)
(identical to (-1)^|i-j| for even L, the only consistent choice for odd L), <Z_i>, <X_i>,
half-ring entropy, E0, E1, residuals. Ground states (float32) are kept for the seeds that pass
the frozen physical selection on C(R), R = floor(L/2) - 1 (--save-states selected).

Run (one block):
  python scripts/run_pbc_reentrant_ED_parity.py --L 17 --h 2.5 --seed-start 5000 --seed-stop 5049
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")   # one thread per block: the dispatcher runs blocks in parallel

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.classical import exact_classical_ground_state  # noqa: E402
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from analyze_pbc_reentrant_selected_seed_topology import select_physical_seeds  # noqa: E402

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
GRID = np.round(np.arange(49) * 0.0125, 6)          # OBC L16/L17 grid: Gamma0 = 0-0.6, step 0.0125
SELECTION = {"endpoint_points": 2, "negative_margin": 0.02, "positive_amplitude": 0.05,
             "minimum_lobe_width": 0.10, "minimum_positive_run": 2}
DEGENERACY_TOL = 1e-9
SCHEMA_VERSION = 1


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=float)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-stop", type=int, help="inclusive")
    parser.add_argument("--save-states", choices=("selected", "none", "all"), default="selected")
    parser.add_argument("--output", type=Path, help="default: results/random_ising_gamma_ensemble/<campaign>")
    parser.add_argument("--check", action="store_true", help="only compare with the reference Hamiltonian")
    return parser.parse_args()


def campaign_name(L, h, a, b):
    return f"pbc-reentrant-ED-parity-L{L}-h{round(h * 10000):05d}-s{a}-{b}-v1"


def finite(value):
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value.item() if isinstance(value, np.generic) else value


def write_json(path, payload):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(finite(payload), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                         # OneDrive may hold the file for a moment
            time.sleep(0.2 * (attempt + 1))
    raise PermissionError(f"could not replace {path}")


def operators(L, J, p, h):
    states = np.arange(2 ** L)
    z = (1 - 2 * ((states[:, None] >> (L - 1 - np.arange(L))[None, :]) & 1)).astype(np.int8)
    bonds = [(i, (i + 1) % L) for i in range(L)]
    diag = sum(Jb * z[:, a].astype(float) * z[:, b] for Jb, (a, b) in zip(J, bonds)) - h * z.sum(axis=1)
    flips = [states ^ (1 << (L - 1 - i)) for i in range(L)]
    X = sparse.csr_matrix((np.concatenate([np.full(states.size, p[i]) for i in range(L)]),
                           (np.concatenate([states] * L), np.concatenate(flips))), shape=(2 ** L, 2 ** L))
    return z, np.asarray(diag, float), X, flips


def circular_phase(L):
    d = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    return (-1.0) ** np.minimum(d, L - d)


def half_ring_entropy(psi, L):
    s = np.linalg.svd(psi.reshape(2 ** (L // 2), -1), compute_uv=False) ** 2
    s = s[s > 1e-16]
    return float(-np.sum(s * np.log(s)))


def c_of_r(cstag, R):
    L = cstag.shape[-1]
    i = np.arange(L)
    return cstag[..., i, (i + R) % L].mean(-1)


def scan(L, h, seed, keep_states):
    base = generate_disorder_base(L, seed, bc="periodic")
    J = np.asarray(base.realize(1.0).J)
    p = np.asarray(base.realize(1.0).Gamma)            # Gamma_i = Gamma0 * p_i in the fixed_mean ensemble
    z, diag, X, flips = operators(L, J, p, h)
    zf = z.astype(float)
    D = sparse.diags(diag)
    phase = circular_phase(L)
    n = GRID.size
    out = {"Czz": np.empty((n, L, L)), "Czz_first_excited": np.empty((n, L, L)), "mz": np.empty((n, L)),
           "mx": np.empty((n, L)), "half_ring_entropy": np.empty(n), "E0": np.empty(n), "E1": np.empty(n),
           "residual_E0": np.zeros(n), "residual_E1": np.zeros(n), "eigsh_retry": np.zeros(n, bool)}
    states = np.empty((n, 2 ** L), np.float32) if keep_states else None
    classical = exact_classical_ground_state(L=L, J=J, h=h, bc="periodic")
    order = np.argsort(diag, kind="stable")
    ground = np.flatnonzero(diag <= diag[order[0]] + DEGENERACY_TOL)
    if abs(diag[order[0]] - classical.energy) > 1e-9:
        raise RuntimeError(f"classical solver mismatch: {diag[order[0]]} vs {classical.energy}")
    v0 = None
    for k, g in enumerate(GRID):
        if g == 0.0:
            w0 = np.zeros(diag.size)
            w0[ground] = 1.0 / ground.size              # equal-weight mixture of the classical minima
            w1 = np.zeros(diag.size)
            w1[order[1]] = 1.0                          # gap 0 when the classical minimum is degenerate
            E = diag[order[:2]]
            psi = np.sqrt(w0)
        else:
            H = D - g * X
            try:
                E, vecs = eigsh(H, k=2, which="SA", v0=v0, tol=1e-10, ncv=20, maxiter=20000)
            except ArpackNoConvergence:
                out["eigsh_retry"][k] = True
                E, vecs = eigsh(H, k=2, which="SA", v0=v0, tol=1e-10, ncv=80, maxiter=100000)
            idx = np.argsort(E)
            E, vecs = E[idx], vecs[:, idx]
            psi = vecs[:, 0]
            w0, w1 = psi ** 2, vecs[:, 1] ** 2
            out["residual_E0"][k] = np.linalg.norm(H @ psi - E[0] * psi)
            out["residual_E1"][k] = np.linalg.norm(H @ vecs[:, 1] - E[1] * vecs[:, 1])
        out["E0"][k], out["E1"][k] = E[0], E[1]
        out["Czz"][k] = (zf.T * w0) @ zf
        out["Czz_first_excited"][k] = (zf.T * w1) @ zf
        out["mz"][k] = w0 @ zf
        out["mx"][k] = [float(psi @ psi[f]) for f in flips] if g > 0 else 0.0   # mixture at Gamma0 = 0
        out["half_ring_entropy"][k] = half_ring_entropy(psi, L) if g > 0 else 0.0
        if states is not None:
            states[k] = psi
        v0 = psi if g > 0 else psi + 1e-3 * np.random.default_rng(seed).normal(size=psi.size)
    out["Cstag"] = phase[None] * out["Czz"]
    out["Cstag_first_excited"] = phase[None] * out["Czz_first_excited"]
    meta = {"classical_degeneracy_Gamma0_0": int(ground.size),
            "classical_parallel_bonds_Gamma0_0": int(np.sum(classical.spins * np.roll(classical.spins, -1) == 1))}
    return J, p, out, states, meta


def check():
    from clean_tfim_tn.ed.random_ising import build_random_ising_hamiltonian
    for L in (8, 9):
        base = generate_disorder_base(L, 11, bc="periodic")
        for g in (0.2, 0.45):
            rr = base.realize(g)
            ref = build_random_ising_hamiltonian(L, rr.J, rr.Gamma, h=2.5, bc="periodic")
            _, diag, X, _ = operators(L, base.realize(1.0).J, base.realize(1.0).Gamma, 2.5)
            err = abs(ref - (sparse.diags(diag) - g * X)).max()
            print(f"L{L} Gamma0={g}: max |H_ref - H| = {err:.2e}", flush=True)
            if err > 1e-12:
                raise RuntimeError("Hamiltonian mismatch")


def main():
    args = arguments()
    if args.check:
        check()
        return
    if None in (args.L, args.h, args.seed_start, args.seed_stop):
        raise SystemExit("--L, --h, --seed-start and --seed-stop are required")
    if args.L not in (16, 17):
        raise ValueError("this campaign is frozen to L = 16, 17")
    R = args.L // 2 - 1
    name = campaign_name(args.L, args.h, args.seed_start, args.seed_stop)
    out_dir = (args.output or BASE / name).resolve()
    (out_dir / "seeds").mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_stop + 1))
    manifest = {"schema_version": SCHEMA_VERSION, "campaign": name, "L": args.L, "h": args.h, "R": R,
                "bc": "periodic", "ensemble": "fixed_mean", "operators": "Pauli Sigmaz/Sigmax",
                "correlators": "raw, not connected", "matrix": "/data/C = Cstag, circular phase (-1)^min(d, L-d)",
                "Gamma0": GRID.tolist(), "seeds": [seeds[0], seeds[-1]], "save_states": args.save_states,
                "selection": SELECTION, "plan": "documentation/PLAN_PBC_ED_PARITY_L16_L17_20261023.md"}
    mpath = out_dir / "manifest.json"
    if mpath.exists():
        if json.loads(mpath.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"{mpath} exists with a different frozen setup")
    else:
        write_json(mpath, manifest)
    status_path = out_dir / "campaign_status.json"
    jobs = []
    for s in seeds:
        done = (out_dir / "seeds" / f"scan_s{s:04d}.npz").exists() and (out_dir / "seeds" / f"scan_s{s:04d}.json").exists()
        target = out_dir / "seeds" / f"scan_s{s:04d}.npz"
        jobs.append({"seed": s, "status": "complete" if done else "pending",
                     "target_file": str(target.relative_to(ROOT) if target.is_relative_to(ROOT) else target)})
    status = {"campaign": name, "status": "running", "pid": os.getpid(), "jobs": jobs,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(status_path, status)
    print(f"{name}: {sum(j['status'] == 'complete' for j in jobs)}/{len(jobs)} seeds already complete", flush=True)
    for job in jobs:
        if job["status"] == "complete":
            continue
        s, t0 = job["seed"], time.time()
        job["status"] = "running"
        write_json(status_path, status)
        try:
            J, p, out, states, meta = scan(args.L, args.h, s, keep_states=args.save_states != "none")
            curve = c_of_r(out["Cstag"], R)
            rows, sel = select_physical_seeds(GRID, np.array([s]), curve[None], **SELECTION)
            selected = bool(len(sel))
            keep = args.save_states == "all" or (args.save_states == "selected" and selected)
            if keep:
                np.savez_compressed(out_dir / "seeds" / f"states_s{s:04d}.npz", Gamma0=GRID, psi=states)
            np.savez_compressed(out_dir / "seeds" / f"scan_s{s:04d}.npz", Gamma0_values=GRID, J=J, Gamma_pattern=p,
                                C=out["Cstag"].astype(np.float32), Cstag=out["Cstag"].astype(np.float32),
                                **{k: v for k, v in out.items() if k not in ("Cstag",)})
            gap = out["E1"] - out["E0"]
            info = {"schema_version": SCHEMA_VERSION, "seed": s, "L": args.L, "h": args.h, "R": R,
                    "selected_physical": selected, "selection_row": rows[0], "states_saved": keep,
                    "C_R": curve.round(6).tolist(), "min_gap": float(gap[1:].min()),
                    "near_degenerate_points": int(np.sum(gap[1:] < 1e-8)),
                    "max_residual_E0": float(out["residual_E0"].max()), "max_residual_E1": float(out["residual_E1"].max()),
                    "eigsh_retries": int(out["eigsh_retry"].sum()), "wall_seconds": round(time.time() - t0, 2), **meta}
            write_json(out_dir / "seeds" / f"scan_s{s:04d}.json", info)
            job["status"] = "complete"
            print(f"seed {s}: {info['wall_seconds']} s, selected={selected}, max residual {info['max_residual_E0']:.1e}",
                  flush=True)
        except Exception:                                # keep going; the block can be rerun (complete seeds are skipped)
            job["status"] = "failed"
            job["error"] = traceback.format_exc(limit=3)
            print(job["error"], flush=True)
        write_json(status_path, status)
    status["status"] = "complete" if all(j["status"] == "complete" for j in jobs) else "incomplete"
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(status_path, status)
    print(f"{name}: {status['status']}", flush=True)


if __name__ == "__main__":
    main()
