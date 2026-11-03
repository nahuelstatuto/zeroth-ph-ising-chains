"""Complete OBC L32 analysis for the matched 1000-seed cohort.

Physical selection precedes topology.  The first 300 prospective seeds and
the 700-seed extension remain explicit replication strata.  H0 uses the full
persistence-landscape L2 velocity; H1 uses the transferred absolute lifetime
thresholds from L16.  Outputs are exploratory and stay outside paper_figures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_clean_L64_new_topology import h1_diagram, mst_deaths  # noqa: E402
from analyze_obc_L32_existing_H0_H1_replication import (  # noqa: E402
    GRID, LAYERS, THRESHOLDS, landscape_from_deaths,
)
from analyze_pbc_reentrant_selected_seed_topology import select_physical_seeds  # noqa: E402


BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OLD = BASE / "L32-reentrant-h23125-prospective-H0-H1-v1" / "analysis"
OUTPUT = BASE / "L32-reentrant-h23125-all1000-complete-v1" / "analysis"
GAMMA = np.linspace(0.0, 0.6, 25)
MID = 0.5 * (GAMMA[:-1] + GAMMA[1:])
STEP = 0.025
CENTRAL = np.round(np.arange(0.008, 0.0151, 0.001), 3)
SELECTION = {
    "endpoint_points": 2,
    "negative_margin": 0.02,
    "positive_amplitude": 0.05,
    "minimum_lobe_width": 0.10,
    "minimum_positive_run": 2,
}
STRICT = {
    "endpoint_maximum": -0.04,
    "positive_amplitude": 0.10,
    "minimum_lobe_width": 0.15,
    "minimum_positive_run": 5,
}
N_BOOT = 5000
N_PERM = 10000
RNG_SEED = 20261013


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    fields.extend(key for row in rows[1:] for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_sources() -> list[dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    with (OLD / "source_manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            records[seed] = {
                "seed": seed, "cohort": "old_150_449", "campaign": row["campaign"],
                "npz": ROOT / row["source_npz"], "sha256": row["sha256"],
            }
    for campaign in sorted(BASE.glob("L32-reentrant-h23125-extension1000-c64-s*-*-v1")):
        config_path, status_path = campaign / "campaign_config.json", campaign / "campaign_status.json"
        if not config_path.is_file() or not status_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        valid = (
            config["L"] == 32 and config["bc"] == "open"
            and config["ensemble"] == "fixed_mean" and config["chi_max"] == 64
            and config["h_values"] == [2.3125] and config["mps_save_policy"] == "selected"
            and np.allclose(config["Gamma0_values"], GAMMA)
        )
        if not valid:
            raise RuntimeError(f"protocol mismatch: {campaign}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        for job in status["jobs"]:
            seed = int(job["seed"])
            if not 450 <= seed <= 1149:
                continue
            if job.get("status") not in {"complete", "reused"}:
                raise RuntimeError(f"incomplete seed {seed}: {job.get('status')}")
            if seed in records:
                raise RuntimeError(f"duplicate seed {seed}")
            path = Path(job["source_directory"]) / "continuation_summary.npz"
            records[seed] = {
                "seed": seed, "cohort": "new_450_1149", "campaign": campaign.name,
                "npz": path, "sha256": "",
            }
    expected = set(range(150, 1150))
    if set(records) != expected:
        raise RuntimeError(f"seed mismatch: missing={sorted(expected-set(records))}, unexpected={sorted(set(records)-expected)}")
    return [records[seed] for seed in sorted(records)]


def high_contrast(row: dict[str, object]) -> bool:
    return bool(
        row["selected"]
        and float(row["low_field_mean"]) <= STRICT["endpoint_maximum"]
        and float(row["high_field_mean"]) <= STRICT["endpoint_maximum"]
        and float(row["maximum_signed_Cstag"]) >= STRICT["positive_amplitude"]
        and float(row["lobe_width"]) >= STRICT["minimum_lobe_width"]
        and int(row["positive_run_points"]) >= STRICT["minimum_positive_run"]
    )


def sign_class(curve: np.ndarray) -> str:
    values = np.asarray(curve > 0, np.int8)
    compressed = [int(values[0])]
    for value in values[1:]:
        if int(value) != compressed[-1]: compressed.append(int(value))
    return {
        (0,): "always_negative", (1,): "always_positive",
        (1, 0): "single_exit", (0, 1): "single_entry",
        (0, 1, 0): "raw_reentrant",
    }.get(tuple(compressed), "complex")


def topological_seed(task: tuple[Path, bool]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    path, calculate_h1 = task
    with np.load(path, allow_pickle=False) as data:
        matrices = np.asarray(data["Cstag"], float)
    landscapes = np.empty((GAMMA.size, LAYERS, GRID.size), np.float32)
    counts = np.zeros((GAMMA.size, THRESHOLDS.size), np.uint16) if calculate_h1 else None
    scalars = np.zeros((GAMMA.size, 4), np.float32) if calculate_h1 else None
    for gi, matrix in enumerate(matrices):
        landscapes[gi] = landscape_from_deaths(mst_deaths(matrix)[:LAYERS])
        if calculate_h1:
            diagram = h1_diagram(matrix)
            lifetime = diagram[:, 1] - diagram[:, 0] if diagram.size else np.empty(0)
            counts[gi] = np.sum(lifetime[:, None] > THRESHOLDS[None, :], axis=0)
            total = float(np.sum(lifetime))
            effective = float(total*total/np.sum(lifetime**2)) if lifetime.size and np.sum(lifetime**2) > 0 else 0.0
            scalars[gi] = [float(np.max(lifetime)) if lifetime.size else 0.0, total, float(lifetime.size), effective]
    derivative = np.diff(landscapes.astype(float), axis=0) / np.diff(GAMMA)[:, None, None]
    return derivative.astype(np.float32), counts, scalars


def peak(curves: np.ndarray, cut: float = STEP) -> np.ndarray:
    eligible = np.flatnonzero(MID - STEP/2 >= cut-1e-12)
    return MID[eligible[np.argmax(curves[..., eligible], axis=-1)]]


def positive_peak(curve: np.ndarray) -> float:
    values = np.maximum(np.diff(np.asarray(curve, float))/np.diff(GAMMA), 0.0)
    values[0] = -np.inf
    return float(MID[np.argmax(values)]) if np.max(values) > 0 else float("nan")


def active_crossing(fraction: np.ndarray) -> float:
    hit = np.flatnonzero(fraction >= .5)
    return float(GAMMA[hit[0]]) if hit.size else float("nan")


def q(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, float); finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {key: float("nan") for key in ("ci_low", "ci50_low", "median", "ci50_high", "ci_high")}
    values = np.quantile(finite, [.025, .25, .5, .75, .975])
    return dict(zip(("ci_low", "ci50_low", "median", "ci50_high", "ci_high"), map(float, values), strict=True))


def paired(indices: np.ndarray, velocity: np.ndarray, selection_rows: list[dict[str, object]], cut: float, rng: np.random.Generator) -> dict[str, object]:
    entry = np.asarray([selection_rows[int(i)]["entry_crossing"] for i in indices], float)
    peaks = peak(velocity[indices], cut)
    residual = peaks-entry
    mae = float(np.mean(np.abs(residual)))
    null = np.empty(N_PERM)
    for draw in range(N_PERM): null[draw] = np.mean(np.abs(peaks-rng.permutation(entry)))
    rho = spearmanr(entry, peaks)
    return {
        "cut": float(cut), "seed_count": int(indices.size),
        "within_one_step": int(np.sum(np.abs(residual) <= STEP+1e-12)),
        "MAE": mae, "median_AE": float(np.median(np.abs(residual))),
        "mean_signed_error": float(np.mean(residual)), "median_signed_error": float(np.median(residual)),
        "pearson": float(np.corrcoef(entry, peaks)[0, 1]), "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
        "permutation_p_MAE": float((1+np.sum(null <= mae))/(N_PERM+1)),
        "entry_median": float(np.median(entry)), "peak_median": float(np.median(peaks)),
    }


def h1_summary(indices: np.ndarray, count: np.ndarray, exits: np.ndarray | None, rng: np.random.Generator) -> list[dict[str, object]]:
    n = indices.size; draws = rng.integers(0, n, size=(N_BOOT, n)); rows = []
    for ti, threshold in enumerate(THRESHOLDS):
        local = count[indices, :, ti]
        growth = positive_peak(np.median(local, axis=0))
        crossing = active_crossing(np.mean(local > 0, axis=0))
        boot_growth = np.empty(N_BOOT); boot_crossing = np.empty(N_BOOT)
        for start in range(0, N_BOOT, 250):
            stop = min(start+250, N_BOOT); sample = local[draws[start:stop]]
            med = np.median(sample, axis=1); frac = np.mean(sample > 0, axis=1)
            for j in range(stop-start):
                boot_growth[start+j] = positive_peak(med[j]); boot_crossing[start+j] = active_crossing(frac[j])
        row = {
            "threshold": float(threshold), "median_count_growth_peak": growth,
            **{f"growth_{k}": v for k, v in q(boot_growth).items()},
            "active_fraction_50_crossing": crossing,
            **{f"active_{k}": v for k, v in q(boot_crossing).items()},
            "active_fraction_final": float(np.mean(local[:, -1] > 0)), "mean_cycles_final": float(np.mean(local[:, -1])),
        }
        if exits is not None:
            row.update(physical_exit_median=float(np.median(exits)), growth_minus_exit=float(growth-np.median(exits)), active_minus_exit=float(crossing-np.median(exits)))
        rows.append(row)
    return rows


def cutoff_protocol(name: str, indices: np.ndarray, velocity: np.ndarray, rng: np.random.Generator) -> tuple[list[dict[str, object]], dict[str, object]]:
    cuts = np.arange(STEP, .2501, STEP); local = velocity[indices]
    draws = rng.integers(0, indices.size, size=(N_BOOT, indices.size))
    weights = np.zeros((N_BOOT, indices.size), np.float32)
    np.add.at(weights, (np.arange(N_BOOT)[:, None], draws), 1.0/indices.size)
    boot_mean = weights @ local; mean = np.mean(local, axis=0)
    rows = []; estimates = []
    for cut in cuts:
        estimate = float(peak(mean[None, :], float(cut))[0]); boot = peak(boot_mean, float(cut))
        unique, counts = np.unique(boot, return_counts=True); mode = int(np.argmax(counts)); first = float(cut+STEP/2)
        rows.append({
            "cohort": name, "seed_count": int(indices.size), "minimum_Gamma_state": float(cut),
            "first_retained_midpoint": first, "aggregate_peak": estimate,
            "distance_from_boundary_steps": float((estimate-first)/STEP),
            "bootstrap_ci_low": float(np.quantile(boot,.025)), "bootstrap_ci_high": float(np.quantile(boot,.975)),
            "bootstrap_mode": float(unique[mode]), "bootstrap_mode_fraction": float(counts[mode]/N_BOOT),
            "bootstrap_within_one_step": float(np.mean(np.abs(boot-estimate) <= STEP+1e-12)),
        }); estimates.append(estimate)
    runs=[]; start=0; estimates=np.asarray(estimates)
    for stop in range(1, estimates.size+1):
        if stop == estimates.size or not np.isclose(estimates[stop], estimates[start]):
            rr=rows[start:stop]
            run={
                "cut_start": float(cuts[start]), "cut_stop": float(cuts[stop-1]), "cut_count": int(stop-start), "peak": float(estimates[start]),
                "maximum_boundary_distance_steps": float(max(x["distance_from_boundary_steps"] for x in rr)),
                "minimum_bootstrap_mode_fraction": float(min(x["bootstrap_mode_fraction"] for x in rr)),
            }
            run["accepted"] = bool(run["cut_count"] >= 3 and run["maximum_boundary_distance_steps"] >= 2-1e-12 and run["minimum_bootstrap_mode_fraction"] >= .5)
            runs.append(run); start=stop
    accepted=[run for run in runs if run["accepted"]]
    return rows, {"runs": runs, "selected_peak": accepted[0]["peak"] if accepted else None, "selected_cut": accepted[0]["cut_start"] if accepted else None}


def h1_individual(indices: np.ndarray, count: np.ndarray, total: np.ndarray, exits: np.ndarray) -> dict[str, object]:
    ti=int(np.flatnonzero(np.isclose(THRESHOLDS,.010))[0]); local=count[indices,:,ti]
    count_peaks=np.asarray([positive_peak(curve) for curve in local]); valid=np.isfinite(count_peaks)
    total_peaks=np.asarray([positive_peak(curve) for curve in total[indices]])
    def metrics(values: np.ndarray, target: np.ndarray) -> dict[str, object]:
        good=np.isfinite(values); values=values[good]; target=target[good]
        return {"detected":int(good.sum()),"within_one_step":int(np.sum(np.abs(values-target)<=STEP+1e-12)),"MAE":float(np.mean(np.abs(values-target))),"spearman":float(spearmanr(values,target).statistic),"median_peak":float(np.median(values))}
    return {"cycle_count_tau_0p010":metrics(count_peaks,exits),"total_persistence_growth":metrics(total_peaks,exits)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records=load_sources(); seeds=np.asarray([r["seed"] for r in records],int)
    physical=np.empty((1000,25)); branch=np.zeros((1000,25),bool); saturated=np.zeros((1000,25),bool); manifest=[]
    for i,record in enumerate(records):
        path=Path(record["npz"])
        with np.load(path,allow_pickle=False) as data:
            matrices=np.asarray(data["Cstag"],float)
            if matrices.shape != (25,32,32) or not np.allclose(data["Gamma0_values"],GAMMA): raise RuntimeError(f"invalid scan {path}")
            physical[i]=np.diagonal(matrices,offset=15,axis1=1,axis2=2).mean(axis=1)
            branch[i]=data["branch_disagreement"]
            saturated[i]=np.maximum(data["forward_max_bond_dimension"],data["backward_max_bond_dimension"])>=64
        digest=record["sha256"] or sha256(path)
        manifest.append({"seed":int(record["seed"]),"cohort":record["cohort"],"campaign":record["campaign"],"source_npz":str(path.relative_to(ROOT)),"sha256":digest})

    selection_rows,selected=select_physical_seeds(GAMMA,seeds,physical,**SELECTION)
    classes=np.asarray([sign_class(curve) for curve in physical])
    for i,row in enumerate(selection_rows):
        row.update(cohort=records[i]["cohort"],sign_class=classes[i],high_contrast=high_contrast(row),branch_disagreement_points=int(np.sum(branch[i])),chi_saturated_points=int(np.sum(saturated[i])))
    strict=np.asarray([i for i in selected if selection_rows[int(i)]["high_contrast"]],int)

    old=np.load(OLD/"analysis_arrays.npz",allow_pickle=False)
    if not np.array_equal(old["seeds"],seeds[:300]) or not np.array_equal(old["thresholds"],THRESHOLDS): raise RuntimeError("old L32 archive mismatch")
    h1_count=np.zeros((1000,25,THRESHOLDS.size),np.uint16); h1_scalar=np.zeros((1000,25,4),np.float32)
    h1_count[:300]=old["H1_cycle_count"]; h1_scalar[:300]=old["H1_scalar"]
    velocity=np.empty((1000,24),np.float32)
    derivative_sum={name:np.zeros((24,LAYERS,GRID.size),float) for name in ("all1000","reentrant","high_contrast","old_reentrant","new_reentrant")}
    selected_set=set(selected.tolist()); strict_set=set(strict.tolist())
    tasks=[(Path(r["npz"]),i>=300) for i,r in enumerate(records)]
    with ThreadPoolExecutor(max_workers=12) as executor:
        for i,(derivative,counts,scalars) in enumerate(executor.map(topological_seed,tasks,chunksize=1)):
            velocity[i]=np.sqrt(np.trapezoid(np.sum(derivative.astype(float)**2,axis=1),GRID,axis=1)/LAYERS)
            if counts is not None: h1_count[i]=counts; h1_scalar[i]=scalars
            derivative_sum["all1000"]+=derivative
            if i in selected_set:
                derivative_sum["reentrant"]+=derivative; derivative_sum["old_reentrant" if i<300 else "new_reentrant"]+=derivative
            if i in strict_set: derivative_sum["high_contrast"]+=derivative
            if (i+1)%25==0 or i==999: print(f"topology {i+1}/1000",flush=True)

    cohorts={
        "all1000":np.arange(1000),"reentrant":selected,"high_contrast":strict,
        "old_reentrant":selected[seeds[selected]<450],"new_reentrant":selected[seeds[selected]>=450],
    }
    velocity_mean={name:np.sqrt(np.trapezoid(np.sum((derivative_sum[name]/idx.size)**2,axis=1),GRID,axis=1)/LAYERS) for name,idx in cohorts.items()}
    rng=np.random.default_rng(RNG_SEED)
    cutoff_rows=[]; cutoff={}
    for name,idx in cohorts.items():
        rows,result=cutoff_protocol(name,idx,velocity,rng); cutoff_rows+=rows; cutoff[name]=result
    paired_primary={name:paired(idx,velocity,selection_rows,STEP,rng) for name,idx in cohorts.items() if name!="all1000"}
    paired_protocol={}
    for name,idx in cohorts.items():
        if name=="all1000": continue
        cut=cutoff[name]["selected_cut"]
        paired_protocol[name]=paired(idx,velocity,selection_rows,float(cut),rng) if cut is not None else None

    h1_tables={}
    for name,idx in cohorts.items():
        exits=None if name=="all1000" else np.asarray([selection_rows[int(i)]["exit_crossing"] for i in idx],float)
        h1_tables[name]=h1_summary(idx,h1_count,exits,rng)
    class_indices={"reentrant_robust":selected,"single_exit":np.flatnonzero(classes=="single_exit"),"always_negative":np.flatnonzero(classes=="always_negative")}
    class_h1={name:h1_summary(idx,h1_count,None,rng) for name,idx in class_indices.items()}
    individual_h1={name:h1_individual(idx,h1_count,h1_scalar[:,:,1],np.asarray([selection_rows[int(i)]["exit_crossing"] for i in idx],float)) for name,idx in cohorts.items() if name!="all1000"}

    # Independent old/new physical and H1 bootstrap comparison at tau=0.010.
    old_idx,new_idx=cohorts["old_reentrant"],cohorts["new_reentrant"]
    rate_test=fisher_exact([[old_idx.size,300-old_idx.size],[new_idx.size,700-new_idx.size]])
    replication={"selection_rate_fisher":{"old":[int(old_idx.size),300],"new":[int(new_idx.size),700],"odds_ratio":float(rate_test.statistic),"p":float(rate_test.pvalue)}}

    write_csv(OUTPUT/"source_manifest.csv",manifest); write_csv(OUTPUT/"physical_selection_all_1000.csv",selection_rows); write_csv(OUTPUT/"H0_cutoff_stability.csv",cutoff_rows)
    for name,rows in h1_tables.items(): write_csv(OUTPUT/f"H1_thresholds_{name}.csv",rows)
    for name,rows in class_h1.items(): write_csv(OUTPUT/f"H1_physical_class_{name}.csv",rows)

    np.savez_compressed(OUTPUT/"analysis_arrays.npz",Gamma0=GAMMA,Gamma0_mid=MID,seeds=seeds,physical_curves=physical,selected_indices=selected,strict_indices=strict,sign_class=classes,branch_disagreement=branch,chi_saturated=saturated,H0_velocity=velocity,H0_velocity_of_mean_reentrant=velocity_mean["reentrant"],H1_cycle_count=h1_count,H1_scalar=h1_scalar,H1_scalar_names=np.asarray(["maximum_lifetime","total_persistence","cycle_count","effective_cycle_count"]),thresholds=THRESHOLDS)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12.8,9.2),constrained_layout=True)
    ax=axes[0,0]; ax.plot(GAMMA,np.mean(physical,axis=0),color=".25",lw=2,label="all 1000"); ax.plot(GAMMA,np.mean(physical[selected],axis=0),color="#0072B2",lw=2,label=f"robust reentrant ({selected.size})"); ax.fill_between(GAMMA,np.quantile(physical[selected],.25,axis=0),np.quantile(physical[selected],.75,axis=0),color="#0072B2",alpha=.15); ax.axhline(0,color="k",lw=.8); ax.set(title="A  Frozen physical selection",xlabel=r"$\Gamma_0$",ylabel="signed long-distance Cstag"); ax.legend(frameon=False); ax.grid(alpha=.15)
    accepted_cut=cutoff["reentrant"]["selected_cut"]; cut=accepted_cut if accepted_cut is not None else STEP; entries=np.asarray([selection_rows[int(i)]["entry_crossing"] for i in selected]); peaks=peak(velocity[selected],cut)
    ax=axes[0,1]; lim=max(np.max(entries),np.max(peaks))+.025; line=np.linspace(0,lim,100); ax.fill_between(line,line-STEP,line+STEP,color=".8",alpha=.5); ax.plot(line,line,"k-",lw=1); ax.scatter(entries,peaks,s=32,color="#009E73",edgecolor="white",lw=.4,alpha=.8); pp=paired_protocol["reentrant"] or paired_primary["reentrant"]; cut_note=f"accepted cut={cut:.3f}" if accepted_cut is not None else f"primary cut={cut:.3f}; no plateau accepted"; ax.text(.02,.96,f"{cut_note}\n{pp['within_one_step']}/{pp['seed_count']} within one step\nMAE={pp['MAE']:.4f}, r={pp['pearson']:.2f}",transform=ax.transAxes,va="top"); ax.set(xlim=(0,lim),ylim=(0,lim),aspect="equal",title="B  Seed-paired H0",xlabel="physical entry",ylabel="H0 landscape peak"); ax.grid(alpha=.15)
    ax=axes[1,0]; ax.plot(MID,np.mean(velocity[selected],axis=0),color="#0072B2",lw=2,label="mean velocities"); scaled=velocity_mean["reentrant"]*np.max(np.mean(velocity[selected],axis=0))/np.max(velocity_mean["reentrant"]); ax.plot(MID,scaled,color="#D55E00",lw=2,label="velocity mean (rescaled)"); ax.axvline(np.median(entries),color="k",ls="--",label="median entry"); ax.axvline(cut,color=".4",ls=":",label="accepted cut" if accepted_cut is not None else "primary lower cut"); ax.set(xlim=(0,.4),title="C  Aggregate H0 response",xlabel=r"$\Gamma_0$ midpoint",ylabel="landscape velocity"); ax.legend(frameon=False); ax.grid(alpha=.15)
    ax=axes[1,1]; ti=int(np.flatnonzero(np.isclose(THRESHOLDS,.010))[0]); colors={"reentrant_robust":"#0072B2","single_exit":"#D55E00","always_negative":".35"}
    for name,idx in class_indices.items(): ax.plot(GAMMA,np.mean(h1_count[idx,:,ti]>0,axis=0),lw=2,color=colors[name],label=f"{name.replace('_',' ')} ({idx.size})")
    ax.axhline(.5,color="k",ls=":"); ax.set(title=r"D  H1 activation, $\tau=0.010$",xlabel=r"$\Gamma_0$",ylabel="fraction with at least one cycle",ylim=(-.02,1.02)); ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.15)
    fig.suptitle("OBC L=32, 1000-seed complete analysis"); fig.savefig(OUTPUT/"L32_all1000_overview.png",dpi=240); fig.savefig(OUTPUT/"L32_all1000_overview.pdf"); plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12.6,8.8),constrained_layout=True); colors={"old":"#0072B2","new":"#D55E00"}
    ax=axes[0,0]; rates=np.asarray([old_idx.size/300,new_idx.size/700]); ax.plot([0,1],rates,"o",ms=9); ax.set(xticks=[0,1],xticklabels=["old 300","new 700"],ylabel="robust-reentrant rate",title=f"A  Physical replication (Fisher p={rate_test.pvalue:.3g})",ylim=(0,max(rates)*1.5)); ax.grid(axis="y",alpha=.2)
    ax=axes[0,1]
    for label,idx,color in (("old",old_idx,colors["old"]),("new",new_idx,colors["new"])):
        e=np.asarray([selection_rows[int(i)]["entry_crossing"] for i in idx]); c=cutoff[label+"_reentrant"]["selected_cut"] or STEP; ax.scatter(e,peak(velocity[idx],c),s=30,color=color,alpha=.7,edgecolor="white",lw=.3,label=f"{label}, cut={c:.3f}")
    line=np.linspace(0,.6,100); ax.fill_between(line,line-STEP,line+STEP,color=".8",alpha=.5); ax.plot(line,line,"k-",lw=1); ax.set(xlim=(0,.6),ylim=(0,.6),aspect="equal",xlabel="physical entry",ylabel="H0 peak",title="B  H0 replication"); ax.legend(frameon=False); ax.grid(alpha=.15)
    ax=axes[1,0]
    for label,idx,color in (("old",old_idx,colors["old"]),("new",new_idx,colors["new"])):
        ax.plot(GAMMA,np.mean(h1_count[idx,:,ti]>0,axis=0),color=color,lw=2,label=label); ax.axvline(np.median([selection_rows[int(i)]["exit_crossing"] for i in idx]),color=color,ls="--")
    ax.axhline(.5,color="k",ls=":"); ax.set(xlabel=r"$\Gamma_0$",ylabel="H1-active fraction",title=r"C  H1 replication, $\tau=0.010$",ylim=(-.02,1.02)); ax.legend(frameon=False); ax.grid(alpha=.15)
    ax=axes[1,1]
    for label,name,color in (("old","old_reentrant",colors["old"]),("new","new_reentrant",colors["new"])):
        rows=h1_tables[name]; thresholds=np.asarray([r["threshold"] for r in rows]); crossings=np.asarray([r["active_fraction_50_crossing"] for r in rows]); ax.plot(thresholds,crossings,"o-",color=color,label=label)
    ax.set(xlabel="H1 lifetime threshold",ylabel="50% active crossing",title="D  Threshold stability"); ax.legend(frameon=False); ax.grid(alpha=.15)
    fig.suptitle("OBC L=32: old-versus-new cohort replication"); fig.savefig(OUTPUT/"L32_all1000_old_new_replication.png",dpi=240); fig.savefig(OUTPUT/"L32_all1000_old_new_replication.pdf"); plt.close(fig)

    summary={
        "status":"complete exploratory L32 1000-seed analysis","created_utc":datetime.now(timezone.utc).isoformat(),
        "conditions":{"L":32,"h":2.3125,"bc":"open","chi":64,"ensemble":"fixed_mean","matrix":"C=Cstag","distance":"D=1-C internally"},
        "input":{"seeds":[150,1149],"count":1000,"old_count":300,"new_count":700,"Gamma0_points":25,"Gamma0_step":STEP},
        "physical_selection":{"topology_blind":True,"rule":SELECTION,"selected_count":int(selected.size),"old_selected":int(old_idx.size),"new_selected":int(new_idx.size),"high_contrast_count":int(strict.size),"sign_classes":dict(Counter(classes.tolist())),"entry_median":float(np.median(entries)),"exit_median":float(np.median([selection_rows[int(i)]["exit_crossing"] for i in selected]))},
        "replication":replication,
        "QC":{"branch_disagreement_points":int(np.sum(branch)),"branch_disagreement_seeds":int(np.sum(np.any(branch,axis=1))),"chi_saturated_points":int(np.sum(saturated)),"chi_saturated_seeds":int(np.sum(np.any(saturated,axis=1))),"selected_branch_disagreement_points":int(np.sum(branch[selected])),"selected_chi_saturated_points":int(np.sum(saturated[selected])),"policy":"flags retained; no exclusions"},
        "H0_paired_primary_excluding_classical_interval":paired_primary,"H0_cutoff_protocol":cutoff,"H0_paired_protocol_selected_cut":paired_protocol,
        "H0_aggregate":{name:{"mean_of_velocities_peak":float(peak(np.mean(velocity[idx],axis=0)[None,:])[0]),"velocity_of_mean_peak":float(peak(velocity_mean[name][None,:])[0])} for name,idx in cohorts.items()},
        "H1_central_thresholds_reentrant":[row for row in h1_tables["reentrant"] if np.any(np.isclose(row["threshold"],CENTRAL))],
        "H1_tau_0p010_by_class":{name:next(row for row in rows if np.isclose(row["threshold"],.010)) for name,rows in class_h1.items()},
        "H1_individual":individual_h1,
        "guardrails":["Physical selection precedes topology.","Old 300 and new 700 remain auditable replication strata.","Absolute H1 thresholds are transferred from L16.","The L32 grid step is 0.025, twice the L16 step.","QC flags are retained.","No finite-L point is called a thermodynamic boundary."],
    }
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
