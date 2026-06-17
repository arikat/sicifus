"""Calibrate the empirical ΔΔG scorer's term weights against experiment.

The empirical scorer (:class:`sicifus.empirical.EmpiricalScorer`) writes ΔΔG as
a weighted sum of physical terms.  With *physical-prior* weights the absolute
terms are sensible but the linear combination is not tuned to reproduce
experimental ΔΔG.  This script fits the weights:

    ΔΔG_exp  ≈  Σ_k  w_k · Δterm_k

where ``Δterm_k`` are the *unweighted* per-term deltas the scorer already
produces (obtained here by running it with all weights = 1).  The fit is
**non-negative** least squares — every term enters with a physically fixed sign,
so weights are constrained ≥ 0 — and is reported with **leave-one-out
cross-validation**, the honest generalisation metric.

The result is written to ``src/sicifus/data/empirical_weights.json``, which the
scorer loads automatically.

Datasets
--------
Two formats, two physical modes:

* ``--format simple``  CSV with columns ``pdb,chain,mutation,ddg`` (and optional
  ``mutated_chains`` for binding).  ``pdb`` is a path or a 4-letter PDB id
  (auto-downloaded + cleaned).  Use for ProTherm-style folding-stability data.
* ``--format skempi``  a SKEMPI v2 export (semicolon-separated).  Binding
  affinities are converted to ΔΔG_bind = RT·ln(Kd_mut / Kd_wt); single point
  mutations only.  Implies ``--mode binding``.

* ``--mode stability``  fits against folding ΔΔG via ``score_mutation``.
* ``--mode binding``  fits against binding ΔΔG via ``score_binding_mutation``
  (needs the full complex; ``mutated_chains`` identifies the mutated partner).

With no ``--dataset`` the embedded barnase (1BNI) stability set is used.

Usage:
    python examples/calibrate_empirical.py [--write]
    python examples/calibrate_empirical.py --dataset protherm.csv --write
    python examples/calibrate_empirical.py --dataset skempi_v2.csv --format skempi --write
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from sicifus.mutate import MutationEngine
from sicifus.empirical import EmpiricalScorer

# Terms that are fit. mc_entropy is omitted when its delta is identically zero
# (no Gly/Pro mutations) and tied to sc_entropy when the JSON is written.
FIT_TERMS = ["vdw", "clash", "solvH", "solvP", "hbond", "elec", "sc_entropy"]
ALL_TERMS = FIT_TERMS[:-1] + ["mc_entropy", "sc_entropy"]

GAS_CONSTANT = 1.987204e-3  # kcal/(mol·K)

_DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "sicifus" / "data"


def weights_path(mode: str) -> Path:
    """Output JSON path for a calibration mode (stability vs binding differ)."""
    name = "empirical_weights.json" if mode == "stability" \
        else "empirical_weights_binding.json"
    return _DATA_DIR / name

# Embedded fallback: barnase 1BNI, chain A, structural numbering, folding ΔΔG.
DEFAULT_RECORDS = [
    {"pdb": "examples/1BNI.pdb", "chain": "A", "mutation": m, "ddg": d}
    for m, d in [
        ("H18K", 1.19), ("I55V", 0.29), ("K62R", 0.48), ("K66A", -0.25),
        ("T79V", -0.29), ("S85A", 0.12), ("I88L", 0.28), ("L89V", 0.27),
        ("V36L", -0.23), ("F56A", 1.77), ("Y24A", 1.02),
    ]
]


# ---------------------------------------------------------------------------
# Structure fetching
# ---------------------------------------------------------------------------

def fetch_pdb(pdb_ref: str, cache_dir: str = "validation_data") -> str:
    """Resolve ``pdb_ref`` to a local cleaned PDB path.

    A path ending in ``.pdb`` is returned as-is.  A 4-letter id is downloaded
    from RCSB and cleaned: HETATM/water/ligand records are stripped (PDBFixer
    cannot build over them) while every protein chain is kept (needed for the
    binding cycle).  Cleaned files are cached under ``cache_dir``.
    """
    if pdb_ref.endswith(".pdb"):
        return pdb_ref

    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True, parents=True)
    clean_path = cache / f"{pdb_ref}_clean.pdb"
    if clean_path.exists():
        return str(clean_path)

    url = f"https://files.rcsb.org/download/{pdb_ref}.pdb"
    print(f"    downloading {pdb_ref} ...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    keep = [
        ln for ln in raw.splitlines()
        if ln.startswith(("ATOM", "TER", "MODEL", "ENDMDL"))
    ]
    clean_path.write_text("\n".join(keep) + "\nEND\n")
    return str(clean_path)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_simple_csv(path: str) -> list:
    """Load a ``pdb,chain,mutation,ddg[,mutated_chains]`` CSV into records."""
    import csv

    records = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rec = {
                "pdb": row["pdb"].strip(),
                "chain": row["chain"].strip(),
                "mutation": row["mutation"].strip(),
                "ddg": float(row["ddg"]),
            }
            mc = row.get("mutated_chains", "").strip()
            if mc:
                rec["mutated_chains"] = set(mc.replace(",", ""))
            records.append(rec)
    return records


_SKEMPI_MUT = re.compile(r"^([A-Z])([A-Za-z])(\d+)([A-Z])$")


def parse_skempi(path: str) -> list:
    """Parse a SKEMPI v2 (semicolon-separated) export into binding records.

    Keeps single point mutations with both affinities present.  ΔΔG_bind is
    RT·ln(Kd_mut / Kd_wt) at the row's temperature (default 298 K).  The mutated
    partner's chains are derived from the ``#Pdb`` field (e.g. ``1CSE_E_I`` →
    groups ``E`` and ``I``); ``mutated_chains`` is whichever group holds the
    mutated chain.

    Uses the ``Mutation(s)_PDB`` column (author/deposited numbering, matching the
    downloaded structure) — NOT ``Mutation(s)_cleaned``, which SKEMPI renumbers
    to a canonical sequence that does not match the PDB residue numbers.
    """
    import csv

    records, skipped = [], 0
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            try:
                mstr = row["Mutation(s)_PDB"].strip()
                if "," in mstr:  # multi-mutation — skip
                    skipped += 1
                    continue
                m = _SKEMPI_MUT.match(mstr)
                if not m:
                    skipped += 1
                    continue
                wt, mchain, pos, mut = m.groups()

                kd_mut = float(row["Affinity_mut_parsed"])
                kd_wt = float(row["Affinity_wt_parsed"])
                if not (kd_mut > 0 and kd_wt > 0):
                    skipped += 1
                    continue
                temp = re.match(r"\s*(\d+)", row.get("Temperature", "") or "")
                T = float(temp.group(1)) if temp else 298.0
                ddg = GAS_CONSTANT * T * np.log(kd_mut / kd_wt)

                pdb_field = row["#Pdb"].split("_")
                pdb_id = pdb_field[0]
                groups = pdb_field[1:]  # e.g. ['E', 'I'] or ['HL', 'A']
                mutated_group = next((g for g in groups if mchain in g), None)
                if mutated_group is None:
                    skipped += 1
                    continue

                records.append({
                    "pdb": pdb_id,
                    "chain": mchain,
                    "mutation": f"{wt}{pos}{mut}",
                    "ddg": float(ddg),
                    "mutated_chains": set(mutated_group),
                })
            except (KeyError, ValueError):
                skipped += 1
    print(f"  parsed {len(records)} usable SKEMPI rows ({skipped} skipped)")
    return records


# ---------------------------------------------------------------------------
# Calibration matrix
# ---------------------------------------------------------------------------

def _subsample(records: list, args) -> list:
    """Apply --pdbs / --max-per-pdb / --max-records filters (deterministic)."""
    import random
    from collections import defaultdict

    rng = random.Random(args.seed)
    if args.pdbs:
        keep = {p.strip().upper() for p in args.pdbs.split(",")}
        records = [r for r in records if r["pdb"].upper() in keep]
    if args.max_per_pdb:
        by_pdb = defaultdict(list)
        for r in records:
            by_pdb[r["pdb"]].append(r)
        out = []
        for rs in by_pdb.values():
            rng.shuffle(rs)
            out.extend(rs[: args.max_per_pdb])
        records = out
    if args.max_records and len(records) > args.max_records:
        records = rng.sample(records, args.max_records)
    return records


def build_matrix(records: list, n_builds: int, mode: str):
    """Return (X, y, labels): mean unweighted per-term deltas and exp ΔΔG."""
    engine = MutationEngine(work_dir="/tmp/calib_mut")
    raw = EmpiricalScorer(
        work_dir="/tmp/calib_emp", weights={k: 1.0 for k in ALL_TERMS})

    X, y, labels = [], [], []
    for rec in records:
        mut, chain = rec["mutation"], rec["chain"]
        tag = f"{rec['pdb']}:{chain}{mut}"
        try:
            pdb_path = fetch_pdb(rec["pdb"])
            per_build = []
            for _ in range(n_builds):
                wt, mutant = engine.build_pdb_pair(pdb_path, [mut], chain=chain)
                if mode == "binding":
                    res = raw.score_binding_mutation(
                        wt, mutant, [mut], rec["mutated_chains"], chain=chain)
                else:
                    res = raw.score_mutation(wt, mutant, [mut], chain=chain)
                dmap = dict(zip(res.energy_terms["term"].to_list(),
                                res.energy_terms["delta"].to_list()))
                per_build.append([dmap[t] for t in FIT_TERMS])
            X.append(np.mean(per_build, axis=0))
            y.append(rec["ddg"])
            labels.append(tag)
            print(f"  {tag}: ok")
        except Exception as e:  # noqa: BLE001 — skip bad rows, don't poison fit
            msg = str(e).splitlines()
            detail = msg[1].strip() if len(msg) > 1 else (msg[0] if msg else "")
            print(f"  {tag}: SKIPPED ({detail[:70]})")
    return np.asarray(X), np.asarray(y), labels


def _nnls(A, b):
    from scipy.optimize import nnls
    w, _ = nnls(A, b)
    return w


def _stats(pred, true):
    r = float(np.corrcoef(pred, true)[0, 1]) if len(true) > 1 else float("nan")
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    return r, rmse, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="CSV path; omit for embedded barnase set")
    ap.add_argument("--format", choices=["simple", "skempi"], default="simple")
    ap.add_argument("--mode", choices=["stability", "binding"], default="stability")
    ap.add_argument("--n-builds", type=int, default=4,
                    help="structure builds averaged per mutation (noise control)")
    ap.add_argument("--pdbs", help="comma-separated PDB ids to keep (subset)")
    ap.add_argument("--max-per-pdb", type=int, default=0,
                    help="cap mutations kept per PDB (0 = no cap)")
    ap.add_argument("--max-records", type=int, default=0,
                    help="cap total records after filtering (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling")
    ap.add_argument("--write", action="store_true",
                    help="write fitted weights to empirical_weights.json")
    args = ap.parse_args()

    if args.dataset is None:
        records, mode = DEFAULT_RECORDS, "stability"
    elif args.format == "skempi":
        records, mode = parse_skempi(args.dataset), "binding"
    else:
        records, mode = load_simple_csv(args.dataset), args.mode

    records = _subsample(records, args)

    print(f"Building calibration matrix ({len(records)} records, mode={mode}, "
          f"{args.n_builds} builds each)...")
    X, y, labels = build_matrix(records, args.n_builds, mode)
    print(f"Usable points: {len(y)}")
    if len(y) < len(FIT_TERMS):
        print("WARNING: fewer points than free terms — the fit is "
              "underdetermined and will overfit. Add more data.")

    w = _nnls(X, y)
    r, rmse, mae = _stats(X @ w, y)
    print("\nFitted weights:")
    for t, wt in zip(FIT_TERMS, w):
        print(f"  {t:11s} {wt:.4f}")
    print(f"\nIn-sample  R={r:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

    # Leave-one-out cross-validation (honest generalisation).
    n = len(y)
    loo = np.array([
        X[i] @ _nnls(X[np.arange(n) != i], y[np.arange(n) != i])
        for i in range(n)
    ])
    rl, rmsel, mael = _stats(loo, y)
    print(f"LOO-CV     R={rl:.3f}  RMSE={rmsel:.3f}  MAE={mael:.3f}")

    if args.write:
        wd = dict(zip(FIT_TERMS, [round(float(x), 4) for x in w]))
        wd["mc_entropy"] = wd["sc_entropy"]  # tied: not identifiable from data
        payload = {
            "_provenance": (
                f"Calibrated by examples/calibrate_empirical.py "
                f"(non-negative least squares, mode={mode}, n={n}). "
                f"Recalibrate on ProTherm/SKEMPI for production."
            ),
            "_metrics": {"n": n, "mode": mode, "loo_cv_r": round(rl, 3),
                         "loo_cv_rmse": round(rmsel, 3), "loo_cv_mae": round(mael, 3)},
            "weights": wd,
        }
        out = weights_path(mode)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {out}")
        if mode == "binding":
            print("  (binding regime — load explicitly: "
                  "EmpiricalScorer(weights=...); the stability default "
                  "empirical_weights.json is left untouched.)")
    else:
        print("\n(dry run — pass --write to update empirical_weights.json)")


if __name__ == "__main__":
    main()
