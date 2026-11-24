# SPDX-License-Identifier: MIT
"""DK1 reserve-market inputs (FCR, aFRR, mFRR).

* :func:`preprocess_reserves` downloads each market once and caches it under
  the year's ``reserve_markets/`` folder. A valid existing CSV is not rewritten.
* :func:`load_reserve_inputs` reads the CSVs back for the network build.

All CSVs are indexed by tz-naive UTC.
"""

from pathlib import Path

import numpy as np
import pandas as pd

_FMT = "%Y-%m-%dT%H:%M"
_TIME_COLS = ("TimeUTC", "Time UTC", "HourUTC", "Hour UTC")
_META_COLS = {"PriceArea", "Price Area", "TimeDK", "Time DK", "HourDK", "Hour DK", *_TIME_COLS}
_MFRR_CUTOFF = pd.Timestamp("2023-06-20 16:00")  # MfrrReservesDK1 -> mFRRCapacityMarket

AFRR_ACTIVATED_COLS = ["aFRR_DownActivated", "aFRR_UpActivated"]
AFRR_CAP_COLS = ["UpDemandMW", "UpProcuredMW", "UpPriceEUR",
                 "DownDemandMW", "DownProcuredMW", "DownPriceEUR"]
MFRR_CAP_COLS = ["UpDemandMW", "UpProcuredMW", "UpPriceEUR", "UpPriceDKK",
                 "DownDemandMW", "DownProcuredMW", "DownPriceEUR", "DownPriceDKK"]
MFRR_OLD_COLS = [f"mFRR_{d}{s}" for d in ("Down", "Up")
                 for s in ("Expected", "Purchased", "PriceDKK", "PriceEUR",
                           "ExpectedXtra", "PurchasedXtra", "PriceXtraDKK", "PriceXtraEUR")]
REG_BAL_COLS = ["mFRRUpActBal", "mFRRDownActBal", "mFRRUpActSpec", "mFRRDownActSpec",
                "ImbalanceMWh", "ImbalancePriceEUR", "ImbalancePriceDKK",
                "BalancingPowerPriceUpEUR", "BalancingPowerPriceUpDKK",
                "BalancingPowerPriceDownEUR", "BalancingPowerPriceDownDKK"]
IMBALANCE_COLS = ["SatisfiedDemand", "ImbalancePriceEUR", "ImbalancePriceDKK", "SpotPriceEUR",
                  "DominatingDirection", "aFRRUpMW", "aFRRVWAUpEUR", "aFRRVWAUpDKK",
                  "aFRRDownMW", "aFRRVWADownEUR", "aFRRVWADownDKK",
                  "mFRRMarginalPriceUpEUR", "mFRRMarginalPriceUpDKK",
                  "mFRRMarginalPriceDownEUR", "mFRRMarginalPriceDownDKK"]
FCR_COLS = ["FCRdomestic_MW", "FCRabroad_MW", "FCRcross_EUR", "FCRcross_DKK",
            "FCRdk_EUR", "FCRdk_DKK"]

# ── Normalisation ────────────────────────────────────────────────────────────

def reserve_frame(raw, name, hours=None, cols=None, empty_cols=None, area=True):
    """Raw Energi Data Service records -> UTC-indexed DataFrame (DK1).

    cols        keep, validate and convert these columns (None keeps all as delivered).
    empty_cols  if given, no data gives zeros with these columns on ``hours``;
                otherwise an empty result raises.
    hours       hourly data is reindexed onto this grid (gaps = 0); sub-hourly data
                keeps its native resolution; None never reindexes.
    area        filter on PriceArea == DK1 (False for datasets without that column).
    """
    if area and not raw.empty:
        col = next((c for c in ("PriceArea", "Price Area") if c in raw), None)
        if col is None:
            raise RuntimeError(f"{name}: no price-area column in {sorted(raw.columns)}")
        raw = raw[raw[col].eq("DK1")]
    if raw.empty:
        if empty_cols is None:
            raise RuntimeError(f"{name} returned no DK1 records.")
        print(f"[preprocess] WARNING: {name} returned no DK1 records; filling with 0.")
        return pd.DataFrame(0.0, index=hours, columns=empty_cols)

    tcol = next((c for c in _TIME_COLS if c in raw), None)
    if tcol is None:
        raise RuntimeError(f"{name}: no UTC timestamp column in {sorted(raw.columns)}")
    idx = pd.DatetimeIndex(pd.to_datetime(raw[tcol], utc=True).dt.tz_localize(None))
    df = raw.drop(columns=[c for c in raw if c in _META_COLS]).set_axis(idx).sort_index()
    df.index.name = None
    if df.index.has_duplicates:
        raise RuntimeError(f"{name} contains duplicate UTC timestamps.")
    if df.shape[1] == 0:
        raise RuntimeError(f"{name} returned timestamps but no market-data columns.")

    if cols is not None:
        missing = set(cols) - set(df.columns)
        if missing:
            raise RuntimeError(f"{name} is missing columns: {sorted(missing)}")
        df = df[cols].apply(pd.to_numeric, errors="raise")

    if hours is not None:
        if (df.index != df.index.floor("h")).any():
            print(f"[preprocess] {name} has sub-hourly timestamps; keeping native resolution.")
        else:
            n_missing = len(hours.difference(df.index))
            if n_missing:
                print(f"[preprocess] WARNING: {name} misses {n_missing} hours; filling with 0.")
            df = df.reindex(hours)
    return df.fillna(0.0)

def _cache_ok(path, cols=(), hours=None, nonzero_cols=None, nonzero_from=None):
    """True if an existing reserve CSV is usable and must not be downloaded again.

    cols          columns that must be present.
    hours         if given, the index must equal it and contain no NaN.
    nonzero_cols / nonzero_from
                  require a non-zero value in these columns / from this timestamp on
                  (guards against zero-filled files written after a failed download).
    """
    try:
        df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    if df.empty or not set(cols) <= set(df.columns):
        return False
    if hours is not None and (not df.index.equals(hours) or df.isna().to_numpy().any()):
        return False
    if nonzero_cols is None and nonzero_from is None:
        return bool(df.notna().to_numpy().any())
    probe = df if nonzero_from is None else df.loc[nonzero_from:]
    probe = probe if nonzero_cols is None else probe[list(nonzero_cols)]
    return bool(probe.ne(0.0).to_numpy().any())

def _build(path, label, cached, make):
    """Skip if cached, otherwise build the frame and write it."""
    if cached:
        print(f"[preprocess] Skipping {label} download — {path} already exists.")
    else:
        make().to_csv(path, sep=";")

# ── FCR activation from grid frequency ───────────────────────────────────────

def _frequency_day_15min(day, deadband, full):
    """One day of German-local frequency samples -> UTC 15-min mean activation fractions."""
    local = pd.to_datetime(day["DATE"] + " " + day["TIME"], format="%d.%m.%Y %H:%M:%S")
    utc = (local.dt.tz_localize("Europe/Berlin", ambiguous="infer")
                .dt.tz_convert("UTC").dt.tz_localize(None))
    if utc.duplicated().any() or not utc.is_monotonic_increasing:
        raise RuntimeError("German frequency timestamps are not a unique, increasing UTC index.")
    freq = pd.Series(pd.to_numeric(day["FREQUENCY_[HZ]"], errors="raise").to_numpy(),
                     index=pd.DatetimeIndex(utc))
    dev = 50.0 - freq
    act = np.sign(dev) * ((dev.abs() - deadband) / (full - deadband)).clip(0, 1)
    return pd.DataFrame({"act_frac_up": act.clip(lower=0),
                         "act_frac_down": (-act).clip(lower=0)}).resample("15min").mean()

def read_german_frequency_15min(path, deadband=0.010, full=0.200):
    """Stream the frequency CSV one calendar day at a time (DST-safe per day)."""
    parts, pending = [], pd.DataFrame()

    def flush(df):
        parts.extend(_frequency_day_15min(d, deadband, full) for _, d in df.groupby("DATE", sort=False))

    for chunk in pd.read_csv(path, sep=";", decimal=",", usecols=["DATE", "TIME", "FREQUENCY_[HZ]"],
                             dtype={"DATE": str, "TIME": str}, chunksize=86_400):
        df = pd.concat((pending, chunk), ignore_index=True)
        last = df["DATE"].iloc[-1]
        flush(df[df["DATE"].ne(last)])      # complete days
        pending = df[df["DATE"].eq(last)]   # day may continue in the next chunk
    flush(pending)

    if not parts:
        raise RuntimeError(f"No frequency readings found in {path}.")
    out = pd.concat(parts).sort_index()
    if out.index.has_duplicates:
        raise RuntimeError(f"Duplicate UTC 15-minute frequency periods in {path}.")
    return out

def fcr_activation_15min(fractions, fcr_mw):
    """Scale 15-min activation fractions by the hourly FCR capacity (MW)."""
    out = fractions[["act_frac_up", "act_frac_down"]].copy()
    out["fcr_mw"] = fcr_mw.resample("15min").ffill().reindex(out.index)
    for d in ("up", "down"):
        out[f"act_mw_{d}"] = out[f"act_frac_{d}"] * out["fcr_mw"]
    for d in ("up", "down"):
        out[f"act_mwh_{d}"] = out[f"act_mw_{d}"] * 0.25
    return out.fillna(0.0)

# ── Download / cache ─────────────────────────────────────────────────────────

def preprocess_reserves(year, folder, hours, start_date, end_date, fetch):
    """Download and cache all DK1 reserve markets for one year.

    ``fetch`` is ``download_energidata``; ``hours`` is the UTC hourly snapshot index.

    Sources
    -------
    aFRR capacity   AfrrReservesNordic (from 2022-12-08)
    aFRR activation AfrrActivatedAutomatic (<2025), ImbalancePrice (>=2025)
    mFRR capacity   MfrrReservesDK1 (before 2023-06-20 16:00), mFRRCapacityMarket (after)
    mFRR activation RegulatingBalancePowerdata (<2025), MfrrEnergyActivationMarket (from 2025-03-04)
    FCR capacity    FcrDK1
    FCR activation  estimated from German grid frequency (manually supplied CSV)
    """
    f = Path(folder) / "reserve_markets"
    f.mkdir(parents=True, exist_ok=True)
    # the API filters on local HourDK (UTC+1): extend the exclusive boundary by 1 h
    end_1h = (pd.Timestamp(end_date) + pd.Timedelta(hours=1)).strftime(_FMT)

    def get(dataset, start=start_date, end=end_date, area="DK1", sort=None):
        return fetch(dataset_name=dataset, start_date=start, end_date=end,
                     sort_val=sort, price_area=area, limit=0)

    # aFRR capacity -- first local hour 2022-12-08 00:00 DK is 2022-12-07 23:00 UTC
    path = f / "reserves_aFRR_capacity.csv"
    _build(path, "aFRR capacity",
           _cache_ok(path, AFRR_CAP_COLS, hours=hours,
                     nonzero_from="2022-12-07 23:00" if year >= 2022 else None),
           lambda: reserve_frame(
               get("AfrrReservesNordic", max(start_date, "2022-12-07"), sort="TimeUTC asc")
               if year >= 2022 else pd.DataFrame(),
               "AfrrReservesNordic", hours, AFRR_CAP_COLS, AFRR_CAP_COLS))

    # mFRR capacity -- two sources, each keeps its own column names
    path = f / "reserves_mFRR_capacity.csv"
    old, new = year <= 2023, year >= 2023
    need = (MFRR_OLD_COLS if old else []) + (MFRR_CAP_COLS if new else [])

    def mfrr_capacity():
        parts = []
        if pd.Timestamp(start_date) < _MFRR_CUTOFF:
            end = min(pd.Timestamp(end_date) + pd.Timedelta(hours=1), _MFRR_CUTOFF).strftime(_FMT)
            df = reserve_frame(get("MfrrReservesDK1", end=end, area=None), "MfrrReservesDK1",
                               hours, MFRR_OLD_COLS, MFRR_OLD_COLS, area=False)
            parts.append(df[df.index < _MFRR_CUTOFF])
        if pd.Timestamp(end_date) > _MFRR_CUTOFF:
            df = reserve_frame(get("mFRRCapacityMarket", max(start_date, _MFRR_CUTOFF.strftime(_FMT))),
                               "mFRRCapacityMarket", hours, MFRR_CAP_COLS, MFRR_CAP_COLS)
            parts.append(df[df.index >= _MFRR_CUTOFF])
        # concat forms the union of columns; the other source's columns become 0
        return pd.concat(parts).sort_index().reindex(hours).fillna(0.0)

    _build(path, "mFRR capacity",
           _cache_ok(path, need, nonzero_cols=MFRR_OLD_COLS if year in (2022, 2023) else None),
           mfrr_capacity)

    # FCR capacity
    fcr_path = f / "reserves_FCR_DK1.csv"
    _build(fcr_path, "FCR DK1", fcr_path.exists(),
           lambda: reserve_frame(get("FcrDK1", end=end_1h, area=None, sort="HourUTC asc"),
                                 "FcrDK1", hours, FCR_COLS, area=False))

    # FCR activation (15 min) from German frequency
    path = f / "reserves_FCR_DK1_activation_15min.csv"
    freq_path = f / f"{year}_Germany_frequency_grid_measurements.csv"
    if path.exists() and path.stat().st_size > 0:
        print(f"[preprocess] Skipping FCR activation resampling — {path} already exists.")
    elif freq_path.exists():
        cap = pd.read_csv(fcr_path, sep=";", index_col=0, parse_dates=True)
        fcr_mw = cap[["FCRdomestic_MW", "FCRabroad_MW"]].apply(pd.to_numeric, errors="raise").sum(axis=1)
        act = fcr_activation_15min(read_german_frequency_15min(freq_path), fcr_mw)
        act = act[(act.index >= hours[0]) & (act.index < hours[-1] + pd.Timedelta(hours=1))]
        act.to_csv(path, sep=";")
    else:
        print(f"[preprocess] Skipping FCR activation resampling — no German frequency data at {freq_path}.")

    # aFRR activation
    path = f / "reserves_aFRR_activation.csv"
    if year < 2025:
        make = lambda: reserve_frame(get("AfrrActivatedAutomatic", sort="HourUTC asc"),
                                     "AfrrActivatedAutomatic", hours, AFRR_ACTIVATED_COLS, AFRR_ACTIVATED_COLS)
    else:  # all columns as delivered
        make = lambda: reserve_frame(get("ImbalancePrice"), "ImbalancePrice", hours,
                                     empty_cols=IMBALANCE_COLS)
    _build(path, "aFRR activation",
           _cache_ok(path, AFRR_ACTIVATED_COLS if year < 2025 else ()), make)

    # mFRR activation (2025 data kept at native resolution)
    path = f / "reserves_mFRR_activation.csv"
    if year < 2025:
        make = lambda: reserve_frame(get("RegulatingBalancePowerdata", sort="HourUTC asc"),
                                     "RegulatingBalancePowerdata", hours, REG_BAL_COLS, REG_BAL_COLS)
    else:
        make = lambda: reserve_frame(get("MfrrEnergyActivationMarket", max(start_date, "2025-03-04")),
                                     "MfrrEnergyActivationMarket")
    _build(path, "mFRR activation",
           _cache_ok(path, REG_BAL_COLS if year < 2025 else ()), make)

# ── Loading for the network build ────────────────────────────────────────────

def load_reserve_inputs(folder, hours):
    """Read the cached reserve CSVs; keys match the ``inputs_dict`` entries."""
    folder = Path(folder) / "reserve_markets"
    folder.mkdir(parents=True, exist_ok=True)

    def read(name, reindex=True, optional=False):
        path = folder / name
        if optional and (not path.exists() or path.stat().st_size == 0):
            return pd.DataFrame()
        df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True)
        return (df.reindex(hours) if reindex else df).fillna(0.0)

    mfrr_act = read("reserves_mFRR_activation.csv", optional=True)
    if not mfrr_act.empty:
        # ORIGINAL MESSAGE from Energinet
        # Correction: 2026-04-26 00:15-08:15 DK time (UTC+2) the SA request and
        # total mFRR values were wrong; the SA values should be 0, the totals cannot be
        # corrected. Zero all four columns for the period.
        cols = [c for c in ("mFRRSAUpReqMW", "mFRRSADownReqMW", "TotalmFRRUpMW", "TotalmFRRDownMW")
                if c in mfrr_act]
        mfrr_act.loc["2026-04-25 22:15":"2026-04-26 06:15", cols] = 0.0

    return {
        "aFRR_capacity_reserves": read("reserves_aFRR_capacity.csv"),
        "aFRR_energy_activation": read("reserves_aFRR_activation.csv"),
        "mFRR_capacity_reserves": read("reserves_mFRR_capacity.csv"),
        "mFRR_energy_activation": mfrr_act,
        "FCR_capacity_reserves": read("reserves_FCR_DK1.csv"),
        "FCR_activation_15min": read("reserves_FCR_DK1_activation_15min.csv", reindex=False, optional=True),
    }
