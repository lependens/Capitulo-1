"""Offline sensitivity experiment; NOT Dataset v1 or a calibrated baseline run.

Stdlib only. Reads immutable snapshots, writes aggregate diagnostics to a NEW
directory. No network, no interpolation, no source writes. Run from any cwd:
python units_offline.py --repo REPO --output NEW_DIRECTORY
"""
import argparse
import ast
import csv
import hashlib
import json
import logging
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HYPOTHESES = {
    "Rad1_Wind1": (1.0, 1.0),
    "Rad1_Wind2": (1.0, 1 / 3.6),
    "Rad2_Wind1": (0.0864, 1.0),
    "Rad2_Wind2": (0.0864, 1 / 3.6),
}
INPUTS = ("TempMedia", "TempMin", "TempMax", "HumedadMin", "HumedadMax",
          "VelViento", "Radiacion", "EtPMon")
FUNCTIONS = {
    "saturation_vapor_pressure", "slope_vapor_pressure", "psychrometric_constant",
    "atmospheric_pressure", "latent_heat_vaporization", "extraterrestrial_radiation",
    "clear_sky_radiation", "net_longwave_radiation", "penman_monteith_ET0",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy_functions(path):
    """Extract named pure functions only; do not import or execute legacy I/O."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in FUNCTIONS]
    if {n.name for n in functions} != FUNCTIONS:
        raise ValueError("Legacy function set changed")
    namespace = {"math": math, "logging": logging}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def latitude(dms):
    return (int(dms[:2]) + int(dms[2:4]) / 60
            + float(dms[4:-1]) / 1000 / 3600) * (-1 if dms[-1] == "S" else 1)


def parse_row(row):
    try:
        d = date.fromisoformat(row["Fecha"][:10])
        values = {key: float(row[key]) for key in INPUTS}
    except (ValueError, TypeError, KeyError):
        return None, "missing_or_unparseable"
    if not all(math.isfinite(v) for v in values.values()):
        return None, "nonfinite"
    if not values["TempMin"] <= values["TempMedia"] <= values["TempMax"]:
        return None, "temperature_order"
    if not 0 <= values["HumedadMin"] <= values["HumedadMax"] <= 100:
        return None, "humidity_domain"
    if values["VelViento"] < 0 or values["Radiacion"] < 0:
        return None, "negative_wind_or_radiation"
    if values["TempMin"] <= -100 or values["TempMax"] >= 100:
        return None, "temperature_domain"
    return (d, values), None


def calculate(values, day, lat, z, rad_factor, wind_factor, kernel, legacy):
    """Hold all inputs/conventions fixed within a kernel except the two factors.

    Both kernels intentionally use SIAR TempMedia to isolate unit hypotheses.
    This is NOT the final pipeline Tmean convention. Legacy matches PM2 exactly;
    fao56_sensitivity uses standard coefficients, without its lower cloud floor.
    Wind height remains an ASSUMPTION, identical under all hypotheses.
    """
    t, lo, hi = (values[k] for k in ("TempMedia", "TempMin", "TempMax"))
    rs, wind = values["Radiacion"] * rad_factor, values["VelViento"] * wind_factor
    es0 = legacy["saturation_vapor_pressure"] if kernel == "legacy_pm2" else (
        lambda x: 0.6108 * math.exp(17.27 * x / (x + 237.3)))
    es = (es0(lo) + es0(hi)) / 2
    ea = (es0(lo) * values["HumedadMax"] + es0(hi) * values["HumedadMin"]) / 200
    delta = 4098 * es0(t) / (t + 237.3)**2
    pressure = legacy["atmospheric_pressure"](z)
    gamma = (legacy["psychrometric_constant"](pressure, legacy["latent_heat_vaporization"](t))
             if kernel == "legacy_pm2" else 0.000665 * pressure)
    ra = legacy["extraterrestrial_radiation"](lat, day.timetuple().tm_yday)
    rso = legacy["clear_sky_radiation"](ra, z)
    if kernel == "legacy_pm2":
        rnl = legacy["net_longwave_radiation"](hi, lo, ea, rs, rso)
    else:
        rnl = (4.903e-9 * ((hi + 273.16)**4 + (lo + 273.16)**4) / 2
               * (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * min(rs / rso, 1) - 0.35))
    rn = 0.77 * rs - rnl
    et0 = legacy["penman_monteith_ET0"](delta, rn, 0, gamma, t, wind, es, ea)
    return dict(et0=et0, rs=rs, wind=wind, ratio=rs/rso, rn=rn, rnl=rnl)


class Summary:
    def __init__(self):
        self.n = 0
        self.error = self.absolute = self.squared = 0.0
        self.obs = self.et = self.rs = self.wind = self.ratio = 0.0
        self.negative_et = self.low_ratio = self.high_ratio = self.negative_rnl = 0
        self.min_et = math.inf
        self.max_et = -math.inf

    def add(self, observation, prediction):
        e = prediction["et0"] - observation
        self.n += 1
        self.error += e
        self.absolute += abs(e)
        self.squared += e*e
        self.obs += observation
        self.et += prediction["et0"]
        self.rs += prediction["rs"]
        self.wind += prediction["wind"]
        self.ratio += prediction["ratio"]
        self.negative_et += prediction["et0"] < 0
        self.low_ratio += prediction["ratio"] < 0.35/1.35
        self.high_ratio += prediction["ratio"] > 1
        self.negative_rnl += prediction["rnl"] < 0
        self.min_et = min(self.min_et, prediction["et0"])
        self.max_et = max(self.max_et, prediction["et0"])

    def result(self):
        return dict(n=self.n, bias=self.error/self.n, mae=self.absolute/self.n,
                    rmse=math.sqrt(self.squared/self.n), observed_mean=self.obs/self.n,
                    et0_mean=self.et/self.n, et0_min=self.min_et, et0_max=self.max_et,
                    rs_mean=self.rs/self.n, wind_mean=self.wind/self.n,
                    rs_rso_mean=self.ratio/self.n, negative_et0=self.negative_et,
                    low_rs_rso=self.low_ratio, rs_above_rso=self.high_ratio,
                    negative_rnl=self.negative_rnl)


def strata(code, d, values):
    season = "DJF" if d.month in (12, 1, 2) else "MAM" if d.month < 6 else "JJA" if d.month < 9 else "SON"
    return [("global", "all"), ("station", code), ("year", str(d.year)),
            ("station_year", f"{code}:{d.year}"), ("season", season),
            ("raw_radiation_bin", "0-5" if values["Radiacion"] <= 5 else "5-20" if values["Radiacion"] <= 20 else ">20"),
            ("raw_wind_bin", "0-1" if values["VelViento"] <= 1 else "1-4" if values["VelViento"] <= 4 else ">4"),
            ("temperature_bin", "<=10C" if values["TempMedia"] <= 10 else "10-25C" if values["TempMedia"] <= 25 else ">25C"),
            ("humidity_bin", "RHmin<=40" if values["HumedadMin"] <= 40 else "RHmin>40")]


def run(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output must be a new directory")
    if any(output.is_relative_to(repo / p) for p in ("datos_siar_baleares", "legacy", "sources", "data_quality", "siar-sync", "forecast_collector", "aemet_forecast_collector")):
        raise ValueError("Output cannot be inside reference/acquisition directories")
    legacy_path = repo / "legacy/et0/PM2.py"
    legacy = load_legacy_functions(legacy_path)
    catalog_path = repo / "estaciones_baleares.csv"
    with catalog_path.open(encoding="utf-8-sig", newline="") as stream:
        catalog = {r["Codigo"]: (latitude(r["Latitud"]), float(r["Altitud"])) for r in csv.DictReader(stream)}
    sources = ([("canonical_snapshot", p) for p in sorted((repo/"datos_siar_baleares").glob("*_datos_completos.csv"))]
               + [("legacy_calculated", p) for p in sorted((repo/"legacy/data/siar-derived/csv").glob("*_et0_calc.csv"))]
               + [("legacy_backup", p) for p in sorted((repo/"legacy/data/siar-derived/backups/csv").glob("*_datos_completos.csv"))])
    groups, records, examples = defaultdict(Summary), [], []
    for corpus, path in sources:
        code = path.name.split("_")[0].upper()
        lat, z = catalog[code]
        digest = sha256(path)
        excluded, seen, duplicate = Counter(), set(), 0
        total = valid = zero = 0
        dates, raw_rad, raw_wind = [], [], []
        cohort_hash = hashlib.sha256()
        reproduction_max = 0.0
        for line, row in enumerate(csv.DictReader(path.open(encoding="utf-8-sig", newline="")), 2):
            total += 1
            parsed, reason = parse_row(row)
            if reason:
                excluded[reason] += 1
                continue
            d, values = parsed
            if d in seen:
                duplicate += 1
                raise ValueError(f"Duplicate date: {path}:{line}")
            seen.add(d)
            if row.get("Estacion") != code:
                raise ValueError(f"Station mismatch: {path}:{line}")
            valid += 1
            zero += values["EtPMon"] == 0
            dates.append(d.isoformat())
            raw_rad.append(values["Radiacion"])
            raw_wind.append(values["VelViento"])
            cohort_hash.update(f"{path.relative_to(repo).as_posix()}:{line}:{d.isoformat()}\n".encode())
            for kernel in ("legacy_pm2", "fao56_sensitivity"):
                for hypothesis, factors in HYPOTHESES.items():
                    prediction = calculate(values, d, lat, z, *factors, kernel, legacy)
                    if not all(math.isfinite(v) for v in prediction.values()):
                        raise ValueError(f"Nonfinite prediction: {path}:{line}")
                    for kind, key in strata(code, d, values):
                        groups[corpus, kernel, hypothesis, kind, key].add(values["EtPMon"], prediction)
                    if code == "IB04" and d == date(2004, 4, 1):
                        historical = calculate(values, d, 39.5, 100, *factors, kernel, legacy)
                        examples.append(dict(corpus=corpus, kernel=kernel, hypothesis=hypothesis,
                                             source=path.relative_to(repo).as_posix(), source_line=line,
                                             observation=values["EtPMon"], inputs=values,
                                             catalog_metadata=prediction,
                                             historical_placeholder_metadata=historical))
            if corpus == "legacy_calculated":
                # A separate regression checks extracted PM2 against its preserved
                # outputs; inferred historical placeholders are NOT used for the
                # main hypothesis scores with catalog metadata.
                hist_lat, hist_z = (39.5, 100) if code in ("IB03", "IB04") else (lat, z)
                predicted = calculate(values, d, hist_lat, hist_z, 1, 1, "legacy_pm2", legacy)["et0"]
                reproduction_max = max(reproduction_max, abs(predicted - float(row["ET0_calc"])))
        def quantiles(xs):
            xs = sorted(xs)
            return {str(q): xs[round((len(xs)-1)*q)] for q in (0, .01, .5, .99, 1)} if xs else {}
        if digest != sha256(path):
            raise ValueError(f"Source changed during read: {path}")
        records.append(dict(corpus=corpus, path=path.relative_to(repo).as_posix(), sha256=digest,
                            station=code, latitude_deg=lat, altitude_m=z,
                            rows=total, eligible=valid, excluded=dict(excluded),
                            duplicates=duplicate, observed_zero_included=zero,
                            first=min(dates) if dates else None, last=max(dates) if dates else None,
                            cohort_sha256=cohort_hash.hexdigest(),
                            raw_radiation_quantiles=quantiles(raw_rad), raw_wind_quantiles=quantiles(raw_wind),
                            historical_reproduction_max_abs_error=reproduction_max if corpus == "legacy_calculated" else None))
    scores = [dict(corpus=c, kernel=k, hypothesis=h, group=g, key=v, **summary.result())
              for (c, k, h, g, v), summary in sorted(groups.items())]
    manifest = dict(experiment="UNITS-OFFLINE-FORENSIC", schema_version=1,
                    source_repo_sha=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                    source_tree_dirty=bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()),
                    python_version=sys.version,
                    tool_sha256=sha256(Path(__file__)), catalog_sha256=sha256(catalog_path),
                    legacy_pm2_sha256=sha256(legacy_path), hypotheses=HYPOTHESES,
                    assumptions=dict(wind_height="assumed 2m, not certified", tmean="SIAR TempMedia in both kernels",
                                     bias="prediction minus EtPMon", cohort="same eligible rows for all hypotheses; no sigma; zeros included",
                                     corpus="never pool overlapping canonical/legacy sources", metadata="current catalog; historical placeholders only separate regression/example"),
                    source_files=records)
    output.mkdir(parents=True)
    with (output/"scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    (output/"ib04_example.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    manifest["output_sha256"] = {name: sha256(output/name) for name in ("scores.csv", "ib04_example.json")}
    for record in records:
        if sha256(repo/record["path"]) != record["sha256"]:
            raise ValueError("Source changed before experiment completion")
    for name, value in (("manifest.json", manifest),):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(files=len(records), score_rows=len(scores), output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.output)
