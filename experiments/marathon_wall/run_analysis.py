"""Driver: one Box's-loop turn on the Boston Marathon 5K split grid.

Usage::

    uv run python -m experiments.marathon_wall.run_analysis --stage all

Stages are independent so an expensive one can be re-run alone:

``data``        hygiene + metric + exploratory figure, no sampling
``fit``         the three competing models on the real splits
``validate``    LOO/LOO-PIT, pi by stratum, Smyth comparison, invariance
``simulate``    the two calibration studies (skew-normal null, year shock)
``changepoint`` the Task 6 piecewise model on a subsample

Where to put the data
---------------------
Drop any of these in ``--data-dir`` and they are picked up automatically:

* ``marathon_results_2015.csv`` / ``2016`` / ``2017`` -- the Kaggle "Boston
  Marathon" dumps (``rojour`` schema, ``HH:MM:SS`` splits, 15K mat present).
* ``results_2013.csv`` / ``results_2014.csv`` -- the ``llimllib/bostonmarathon``
  scrape of baa.org (``llimllib`` schema, decimal-minute splits, no 15K mat).

Every race found is analysed; the grouping adapts to whichever years are
present.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final, Mapping, Sequence

import arviz as az
import numpy as np
import pandas as pd

from . import plots
from .changepoint import SegmentPanel, build_changepoint_model, summarise_changepoints
from .data import SCHEMAS, SplitSchema, load_races
from .diagnostics import (
    bayes_decision,
    calibrate_loss_ratio,
    compare_models,
    decision_boundary,
    loo_table,
    posterior_responsibility,
)
from .metric import MetricWindows, build_metric, segment_pace_table
from .models import GroupIndex, build_model, sample_model
from .validation import (
    SMYTH_2021_WALL_RATE,
    compare_to_smyth_rule,
    early_shape_test,
    simulate_skewnormal_null,
    simulate_year_shock,
    trajectory_invariance,
    wall_fraction_by_stratum,
)

_FILE_PATTERNS: Final[tuple[tuple[str, str, int], ...]] = (
    ("marathon_results_2015.csv", "rojour", 2015),
    ("marathon_results_2016.csv", "rojour", 2016),
    ("marathon_results_2017.csv", "rojour", 2017),
    ("results_2013.csv", "llimllib", 2013),
    ("results_2014.csv", "llimllib", 2014),
)

_MODELS: Final[tuple[str, ...]] = ("gaussian", "mixture", "skewnormal")


def discover_sources(data_dir: Path) -> dict[int, tuple[Path, SplitSchema]]:
    """Locate every supported results file in ``data_dir``."""
    found: dict[int, tuple[Path, SplitSchema]] = {}
    for filename, schema_name, year in _FILE_PATTERNS:
        path = data_dir / filename
        if path.exists():
            found[year] = (path, SCHEMAS[schema_name])
    if not found:
        raise FileNotFoundError(
            f"no recognised results file in {data_dir}; expected one of "
            + ", ".join(name for name, _, _ in _FILE_PATTERNS)
        )
    return found


def grouping_columns(metric: pd.DataFrame) -> list[str]:
    """Group by sex crossed with year when several years are present, else wave."""
    return ["sex", "year"] if metric["year"].nunique() > 1 else ["sex", "wave"]


def _ability_quantile_table(metric: pd.DataFrame, n_bins: int = 30) -> pd.DataFrame:
    binned = pd.qcut(metric["ability"], n_bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(pd.unique(binned.dropna())):
        member = (binned == b).to_numpy()
        rows.append(
            {
                "ability": float(metric.loc[member, "ability"].median()),
                "q10": float(metric.loc[member, "y_raw"].quantile(0.10)),
                "q50": float(metric.loc[member, "y_raw"].quantile(0.50)),
                "q90": float(metric.loc[member, "y_raw"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def stage_data(data_dir: Path, out_dir: Path, fig_dir: Path, quantile: float) -> pd.DataFrame:
    sources = discover_sources(data_dir)
    tidy, hygiene = load_races(sources)
    hygiene.to_csv(out_dir / "hygiene.csv", index=False)

    metric, reference = build_metric(tidy, quantile=quantile)
    metric.to_csv(out_dir / "metric.csv", index=False)
    reference.to_frame().to_csv(out_dir / "terrain_reference.csv", index=False)

    segments = segment_pace_table(tidy)
    profile = segments.median(numeric_only=True)
    profile.to_csv(out_dir / "segment_profile.csv")

    plots.use_report_style()
    plots.figure_data(
        profile,
        reference.to_frame(),
        _ability_quantile_table(metric),
        metric,
        fig_dir / "wall_data.pdf",
    )
    print(f"[data] n={len(metric)}  mean(y)={metric['y'].mean():.4f}  sd={metric['y'].std():.4f}  "
          f"skew={metric['y'].skew():.3f}")
    return metric


def stage_fit(metric: pd.DataFrame, out_dir: Path, **sampler_kwargs: object) -> dict[str, az.InferenceData]:
    groups = GroupIndex.from_columns(metric, grouping_columns(metric))
    print(f"[fit] grouping = {grouping_columns(metric)}  ({groups.n_groups} groups)")
    print(groups.counts().to_string())

    y = metric["y"].to_numpy(dtype=float)
    fitted: dict[str, az.InferenceData] = {}
    for name in _MODELS:
        print(f"[fit] sampling {name} ...", flush=True)
        idata = sample_model(build_model(name, y, groups), **sampler_kwargs)
        idata.to_netcdf(out_dir / f"idata_{name}.nc")
        fitted[name] = idata
    return fitted


def stage_validate(
    metric: pd.DataFrame,
    fitted: Mapping[str, az.InferenceData],
    out_dir: Path,
    fig_dir: Path,
) -> dict[str, object]:
    groups = GroupIndex.from_columns(metric, grouping_columns(metric))
    y = metric["y"].to_numpy(dtype=float)
    results: dict[str, object] = {}

    # (a) predictive comparison -------------------------------------------------
    comparison = compare_models(fitted)
    comparison.to_csv(out_dir / "loo_compare.csv")
    table = loo_table(fitted)
    table.to_csv(out_dir / "loo_table.csv", index=False)
    print(comparison.to_string())
    print(table.to_string())
    results["loo_compare"] = json.loads(comparison.reset_index().to_json(orient="records"))
    results["loo_table"] = json.loads(table.to_json(orient="records"))

    # responsibilities and the decision rule ------------------------------------
    mixture = fitted["mixture"]
    responsibility = posterior_responsibility(mixture, y, groups.codes)
    metric = metric.assign(responsibility=responsibility)
    metric.to_csv(out_dir / "metric_scored.csv", index=False)

    pi_frame = wall_fraction_by_stratum(mixture, groups)
    pi_frame.to_csv(out_dir / "pi_by_group.csv", index=False)
    print(pi_frame.to_string())
    results["pi_by_group"] = json.loads(pi_frame.to_json(orient="records"))

    # (b) comparison against the rule-based incidence ---------------------------
    rule_rows: list[dict[str, object]] = []
    for sex, target in SMYTH_2021_WALL_RATE.items():
        member = (metric["sex"] == sex).to_numpy()
        if member.sum() == 0:
            continue
        equal_cost = float(np.mean(bayes_decision(responsibility[member], loss_ratio=1.0)))
        loss_ratio, threshold = calibrate_loss_ratio(responsibility[member], target)
        codes_for_sex = np.unique(groups.codes[member])
        upper = [decision_boundary(mixture, int(c), side="upper") for c in codes_for_sex]
        lower = [decision_boundary(mixture, int(c), side="lower") for c in codes_for_sex]
        rule_rows.append(
            {
                "sex": sex,
                "n": int(member.sum()),
                "pi_posterior_mean": float(pi_frame.loc[
                    [str(g).startswith(sex) for g in pi_frame["group"]], "pi_mean"
                ].mean()),
                "equal_cost_rate": equal_cost,
                "smyth_rate": target,
                "loss_ratio_matching_smyth": float(loss_ratio),
                "threshold_matching_smyth": float(threshold),
                "decision_boundary_y": float(np.nanmean(upper)),
                "spurious_lower_boundary_y": float(np.nanmean(lower)),
            }
        )
    rule_frame = pd.DataFrame(rule_rows)
    rule_frame.to_csv(out_dir / "decision_rule.csv", index=False)
    print(rule_frame.to_string())
    results["decision_rule"] = json.loads(rule_frame.to_json(orient="records"))

    # (c) trajectory invariance -------------------------------------------------
    segments = segment_pace_table(metric)
    invariance = trajectory_invariance(metric, responsibility, segments)
    invariance.to_csv(out_dir / "invariance.csv", index=False)
    print(invariance.to_string())
    results["invariance"] = json.loads(invariance.to_json(orient="records"))

    smyth = compare_to_smyth_rule(metric, responsibility)
    smyth.to_csv(out_dir / "smyth_rule.csv", index=False)
    print(smyth.to_string())
    results["smyth_rule"] = json.loads(smyth.to_json(orient="records"))

    early = early_shape_test(metric, responsibility, segments)
    early.to_csv(out_dir / "early_shape.csv", index=False)
    print(early.to_string())
    results["early_shape"] = json.loads(early.to_json(orient="records"))

    # figures -------------------------------------------------------------------
    plots.use_report_style()
    grid = np.linspace(-0.3, 0.7, 700)
    weights = groups.counts().to_numpy(dtype=float)
    densities = {name: plots.pooled_predictive_density(idata, name, grid, weights)
                 for name, idata in fitted.items()}
    plots.figure_model(metric, groups.codes, grid, densities, fitted, pi_frame,
                       SMYTH_2021_WALL_RATE, fig_dir / "wall_model.pdf")
    plots.figure_invariance(invariance, fig_dir / "wall_invariance.pdf")
    return results


def stage_simulate(out_dir: Path, **sampler_kwargs: object) -> dict[str, object]:
    print("[simulate] skew-normal null ...", flush=True)
    null = simulate_skewnormal_null(**sampler_kwargs)
    null.to_csv(out_dir / "sim_skewnormal_null.csv", index=False)
    print(null.to_string())

    print("[simulate] planted year shock ...", flush=True)
    shock = simulate_year_shock(**sampler_kwargs)
    shock.to_csv(out_dir / "sim_year_shock.csv", index=False)
    print(shock.to_string())
    return {
        "skewnormal_null": json.loads(null.to_json(orient="records")),
        "year_shock": json.loads(shock.to_json(orient="records")),
    }


def stage_changepoint(
    metric: pd.DataFrame,
    out_dir: Path,
    *,
    max_athletes: int,
    **sampler_kwargs: object,
) -> dict[str, object]:
    groups = GroupIndex.from_columns(metric, grouping_columns(metric))
    segments = segment_pace_table(metric)
    panel = SegmentPanel.from_frame(metric, segments, groups, max_athletes=max_athletes)
    print(f"[changepoint] panel {panel.log_pace.shape} over {panel.segment_names}")
    model = build_changepoint_model(panel)
    idata = sample_model(model, compute_log_likelihood=False, **sampler_kwargs)
    idata.to_netcdf(out_dir / "idata_changepoint.nc")
    summary = summarise_changepoints(idata)
    summary.to_csv(out_dir / "changepoint_summary.csv", index=False)
    print(summary.to_string())
    return json.loads(summary.to_json(orient="records"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--stage", nargs="+", default=["all"],
                        choices=["all", "data", "fit", "validate", "simulate", "changepoint"])
    parser.add_argument("--quantile", type=float, default=0.25, help="reference quantile for the terrain curve")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--sim-draws", type=int, default=750)
    parser.add_argument("--changepoint-athletes", type=int, default=1200)
    args = parser.parse_args(argv)

    stages = set(args.stage)
    if "all" in stages:
        stages = {"data", "fit", "validate", "simulate", "changepoint"}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    sampler = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    sim_sampler = {"draws": args.sim_draws, "tune": args.tune, "chains": min(args.chains, 2)}

    results: dict[str, object] = {}
    metric: pd.DataFrame | None = None
    fitted: dict[str, az.InferenceData] = {}

    if "data" in stages:
        metric = stage_data(args.data_dir, args.out_dir, args.fig_dir, args.quantile)
    elif stages & {"fit", "validate", "changepoint"}:
        metric = pd.read_csv(args.out_dir / "metric.csv")

    if "fit" in stages:
        assert metric is not None
        fitted = stage_fit(metric, args.out_dir, **sampler)
    elif "validate" in stages:
        fitted = {name: az.from_netcdf(args.out_dir / f"idata_{name}.nc") for name in _MODELS}

    if "validate" in stages:
        assert metric is not None
        results["validate"] = stage_validate(metric, fitted, args.out_dir, args.fig_dir)

    if "simulate" in stages:
        results["simulate"] = stage_simulate(args.out_dir, **sim_sampler)

    if "changepoint" in stages:
        assert metric is not None
        results["changepoint"] = stage_changepoint(
            metric, args.out_dir, max_athletes=args.changepoint_athletes, **sim_sampler
        )

    if results:
        (args.out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
