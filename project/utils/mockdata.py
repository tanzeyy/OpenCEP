#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic bike trip generator with data-driven distributions.

What this script does:
1) Read a real CSV (same schema) and estimate empirical distributions:
   - Trip duration (seconds/minutes), start hour, weekday/weekend, month
   - Start station / End station popularity
   - OD-pair (optional assistance), usertype, birth year, gender
   - BikeID distribution (optionally "empirical")
   - Station catalog is built from the real CSV (ID, name, lat, lon)

2) Generate synthetic data that follows these distributions while embedding
   exactly X hot-path matches of SEQ(a+, b) WITHIN 1h with constraints:
   - 'a' part is a simple path over NON-hot stations (no cycles)
   - Final 'b' ends at a hot-end station
   - Some paths can start near midnight to cross 00:00

Notes:
- You can still override hot-ends, bikeid mode, etc. via CLI.
- If hot-ends is set to "infer", the script picks top-K end stations from data.
- For NOISE trips, you can use "empirical" bikeid-mode to sample by observed frequency.
- Trip durations are sampled from empirical distribution with robust clipping.

CSV headers (exact order):
tripduration,starttime,stoptime,start station id,start station name,start station latitude,
start station longitude,end station id,end station name,end station latitude,end station longitude,
bikeid,usertype,birth year,gender
"""

import csv
import argparse
import random
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Set, Iterable, Optional

# ----------------------------- Headers -----------------------------

HEADERS = [
    "tripduration",
    "starttime",
    "stoptime",
    "start station id",
    "start station name",
    "start station latitude",
    "start station longitude",
    "end station id",
    "end station name",
    "end station latitude",
    "end station longitude",
    "bikeid",
    "usertype",
    "birth year",
    "gender",
]

USERS_FALLBACK = ["Subscriber", "Customer"]
GENDERS_FALLBACK = [0, 1, 2]
BIRTH_YEARS_FALLBACK = list(range(1950, 2006))

# ----------------------------- Helpers -----------------------------


def parse_hot_ends(s: str) -> Optional[Set[int]]:
    """Parse comma-separated station ids into a set of ints; return None if s == 'infer'."""
    if s.strip().lower() == "infer":
        return None
    return {int(tok.strip()) for tok in s.split(",") if tok.strip()}


def parse_len_dist(s: str) -> List[Tuple[int, float]]:
    """
    Parse length distribution string like "1:0.6,2:0.3,3:0.1" into a list of (length, prob).
    The probabilities will be normalized if they don't sum to 1.
    """
    pairs = []
    for part in s.split(","):
        k, v = part.split(":")
        pairs.append((int(k.strip()), float(v.strip())))
    total = sum(p for _, p in pairs)
    if total <= 0:
        raise ValueError("hot-len-dist probabilities must sum to > 0")
    return [(k, p / total) for k, p in pairs]


def sample_len(len_dist: List[Tuple[int, float]]) -> int:
    """Sample an 'a' length according to provided distribution."""
    ks = [k for k, _ in len_dist]
    ps = [p for _, p in len_dist]
    return random.choices(ks, weights=ps, k=1)[0]


def dt_parse_flex(s: str) -> Optional[datetime]:
    """
    Parse datetime robustly for common formats:
    - "YYYY-mm-dd HH:MM:SS"
    - "m/d/Y HH:MM:SS"
    - "m/d/YY HH:MM:SS"
    Return None on failure.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",  # fallback
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            pass
    return None


def quantile(values: List[float], q: float) -> float:
    """Simple quantile for a list; return a reasonable fallback if list is empty."""
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def to_prob(counter: Counter) -> List[Tuple[any, float]]:
    """Convert a Counter to a list of (key, prob) pairs."""
    total = sum(counter.values())
    if total == 0:
        return []
    return [(k, v / total) for k, v in counter.items()]


def weighted_choice(pairs: List[Tuple[any, float]]):
    """Randomly choose based on (key, prob) pairs."""
    if not pairs:
        return None
    keys = [k for k, _ in pairs]
    ps = [p for _, p in pairs]
    return random.choices(keys, weights=ps, k=1)[0]


def station_tuple_from_catalog(sid: int, catalog: Dict[int, Tuple[str, float, float]]) -> Tuple[int, str, float, float]:
    """Return station tuple by id; raise if missing."""
    if sid not in catalog:
        raise KeyError(f"Station id {sid} not found in catalog.")
    name, lat, lon = catalog[sid]
    return sid, name, lat, lon


def make_trip_row(
    start_dt: datetime,
    duration_min: int,
    start_sid: int,
    end_sid: int,
    bike_id: int,
    catalog: Dict[int, Tuple[str, float, float]],
    usertype_sampler,
    birthyear_sampler,
    gender_sampler,
    time_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> List:
    """Create a single CSV row with correct schema and integer bikeid."""
    duration_sec = max(1, duration_min * 60)
    end_dt = start_dt + timedelta(seconds=duration_sec)

    ssid, sname, slat, slon = station_tuple_from_catalog(start_sid, catalog)
    esid, ename, elat, elon = station_tuple_from_catalog(end_sid, catalog)

    usertype = usertype_sampler()
    birth_year = birthyear_sampler()
    gender = gender_sampler()

    start_str = start_dt.strftime(time_fmt)
    stop_str = end_dt.strftime(time_fmt)

    return [
        duration_sec,
        start_str,
        stop_str,
        ssid,
        sname,
        slat,
        slon,
        esid,
        ename,
        elat,
        elon,
        bike_id,
        usertype,
        birth_year,
        gender,
    ]


# ----------------------------- BikeId samplers (for noise trips) -----------------------------


class BikeIdSampler:
    """Base class for bikeid sampling for NOISE trips."""

    def next(self) -> int:
        raise NotImplementedError


class SequentialSampler(BikeIdSampler):
    """Sequential increasing integers starting from 'start'."""

    def __init__(self, start: int):
        self.cur = start

    def next(self) -> int:
        v = self.cur
        self.cur += 1
        return v


class UniformRangeSampler(BikeIdSampler):
    """Uniformly sample integers from [lo, hi], with replacement."""

    def __init__(self, lo: int, hi: int):
        if hi < lo:
            raise ValueError("bikeid-end must be >= bikeid-start")
        self.lo, self.hi = lo, hi

    def next(self) -> int:
        return random.randint(self.lo, self.hi)


class ZipfRangeSampler(BikeIdSampler):
    """
    Zipf-like sampling over [lo, hi] with exponent s (>1 recommended).
    Implements weights proportional to 1 / rank^s where rank = 1..N on the range order.
    """

    def __init__(self, lo: int, hi: int, s: float = 1.2):
        if hi < lo:
            raise ValueError("bikeid-end must be >= bikeid-start")
        if s <= 0:
            raise ValueError("bikeid-zipf-s must be > 0")
        self.ids = list(range(lo, hi + 1))
        N = len(self.ids)
        ranks = list(range(1, N + 1))
        weights = [1.0 / (r**s) for r in ranks]
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def next(self) -> int:
        return random.choices(self.ids, weights=self.weights, k=1)[0]


class EmpiricalBikeIdSampler(BikeIdSampler):
    """Sample bikeid from empirical frequency distribution of the real dataset."""

    def __init__(self, id_probs: List[Tuple[int, float]]):
        if not id_probs:
            raise ValueError("EmpiricalBikeIdSampler requires non-empty id_probs")
        self.ids = [k for k, _ in id_probs]
        self.ps = [p for _, p in id_probs]

    def next(self) -> int:
        return random.choices(self.ids, weights=self.ps, k=1)[0]


def build_noise_bike_sampler(
    mode: str, start: int, end: int, zipf_s: float, empirical_pairs: Optional[List[Tuple[int, float]]]
) -> BikeIdSampler:
    """Factory for noise bikeid sampler."""
    mode = mode.lower()
    if mode == "sequential":
        return SequentialSampler(start)
    if mode == "uniform":
        return UniformRangeSampler(start, end)
    if mode == "zipf":
        return ZipfRangeSampler(start, end, s=zipf_s)
    if mode == "empirical":
        if not empirical_pairs:
            raise ValueError("bikeid-mode=empirical requires real data bikeid distribution")
        return EmpiricalBikeIdSampler(empirical_pairs)
    raise ValueError("bikeid-mode must be one of: sequential, uniform, zipf, empirical")


# ----------------------------- Data-driven profiling -----------------------------


def profile_real_csv(path: str):
    """
    Read the real CSV and build:
    - station catalog from both start and end sides
    - empirical distributions for duration_sec, start hour, weekday, month
    - distributions for start station, end station, OD pairs
    - distributions for usertype, birth year, gender, bikeid
    """
    catalog: Dict[int, Tuple[str, float, float]] = {}
    durations_sec: List[int] = []
    hours: Counter = Counter()
    weekdays: Counter = Counter()
    months: Counter = Counter()
    start_station_counts: Counter = Counter()
    end_station_counts: Counter = Counter()
    od_counts: Counter = Counter()
    usertype_counts: Counter = Counter()
    birthyear_counts: Counter = Counter()
    gender_counts: Counter = Counter()
    bikeid_counts: Counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate header order loosely (we rely on names)
        for row in reader:
            try:
                # Station catalog (start)
                ssid = int(row["start station id"])
                sname = row["start station name"]
                slat = float(row["start station latitude"])
                slon = float(row["start station longitude"])
                catalog[ssid] = (sname, slat, slon)

                # Station catalog (end)
                esid = int(row["end station id"])
                ename = row["end station name"]
                elat = float(row["end station latitude"])
                elon = float(row["end station longitude"])
                catalog[esid] = (ename, elat, elon)

                # Datetimes
                st = dt_parse_flex(row["starttime"])
                et = dt_parse_flex(row["stoptime"])
                if not st or not et or et <= st:
                    continue

                # Duration (prefer provided, fallback to computed)
                try:
                    dsec = int(row["tripduration"])
                    if dsec <= 0:
                        dsec = int((et - st).total_seconds())
                except Exception:
                    dsec = int((et - st).total_seconds())
                if dsec <= 0:
                    continue
                durations_sec.append(dsec)

                hours[st.hour] += 1
                weekdays[st.weekday()] += 1  # 0=Mon ... 6=Sun
                months[st.month] += 1

                start_station_counts[ssid] += 1
                end_station_counts[esid] += 1
                od_counts[(ssid, esid)] += 1

                # User side
                usertype = (row.get("usertype") or "").strip() or "Unknown"
                usertype_counts[usertype] += 1

                # Birth year may be float-like text
                by_raw = row.get("birth year")
                try:
                    by = int(float(by_raw)) if by_raw not in (None, "", "NaN") else None
                except Exception:
                    by = None
                if by:
                    birthyear_counts[by] += 1

                # Gender
                g_raw = row.get("gender")
                try:
                    g = int(float(g_raw)) if g_raw not in (None, "", "NaN") else 0
                except Exception:
                    g = 0
                gender_counts[g] += 1

                # Bike id
                bid_raw = row.get("bikeid")
                try:
                    bid = int(float(bid_raw))
                    bikeid_counts[bid] += 1
                except Exception:
                    pass

            except KeyError:
                # Missing expected column names
                continue

    # Build probability tables
    duration_sec_list = sorted(durations_sec)
    # Robust clipping thresholds
    d_lo = int(quantile(duration_sec_list, 0.01)) if duration_sec_list else 60
    d_hi = int(quantile(duration_sec_list, 0.99)) if duration_sec_list else 60 * 60
    # Keep raw list for sampling (rejection sampling with clipping)
    duration_profile = {
        "values": duration_sec_list,
        "clip_lo": max(10, d_lo),
        "clip_hi": max(60, d_hi),
    }

    hour_pairs = to_prob(hours)  # (0..23, p)
    weekday_pairs = to_prob(weekdays)  # (0..6, p)
    month_pairs = to_prob(months)  # (1..12, p)

    start_station_pairs = to_prob(start_station_counts)
    end_station_pairs = to_prob(end_station_counts)
    od_pairs = to_prob(od_counts)

    usertype_pairs = to_prob(usertype_counts) or [(u, 0.5) for u in USERS_FALLBACK]
    birthyear_pairs = to_prob(birthyear_counts) or [(y, 1.0 / len(BIRTH_YEARS_FALLBACK)) for y in BIRTH_YEARS_FALLBACK]
    gender_pairs = to_prob(gender_counts) or [(g, 1.0 / len(GENDERS_FALLBACK)) for g in GENDERS_FALLBACK]
    bikeid_pairs = to_prob(bikeid_counts)

    prof = {
        "catalog": catalog,
        "duration_profile": duration_profile,
        "hour_pairs": hour_pairs,
        "weekday_pairs": weekday_pairs,
        "month_pairs": month_pairs,
        "start_station_pairs": start_station_pairs,
        "end_station_pairs": end_station_pairs,
        "od_pairs": od_pairs,
        "usertype_pairs": usertype_pairs,
        "birthyear_pairs": birthyear_pairs,
        "gender_pairs": gender_pairs,
        "bikeid_pairs": bikeid_pairs,
    }
    return prof


# ----------------------------- Sampling utilities from profile -----------------------------


def sample_duration_min_from_profile(duration_profile: Dict) -> int:
    """Sample a duration in minutes using empirical seconds list with robust clipping."""
    vals = duration_profile["values"]
    if not vals:
        # Fallback: 6..35 min
        return random.randint(6, 35)
    lo = duration_profile["clip_lo"]
    hi = duration_profile["clip_hi"]
    # Rejection sampling on seconds, then convert to minutes (at least 1 minute)
    for _ in range(50):
        s = random.choice(vals)
        if lo <= s <= hi:
            return max(1, int(round(s / 60.0)))
    # Fallback if repeated rejections
    s = max(lo, min(random.choice(vals), hi))
    return max(1, int(round(s / 60.0)))


def build_simple_sampler(
    pairs: List[Tuple[any, float]], fallback_choices: List, fallback_p: Optional[List[float]] = None
):
    """Return a zero-argument function that samples from given pairs or fallback."""
    if pairs:
        keys = [k for k, _ in pairs]
        ps = [p for _, p in pairs]

        def _f():
            return random.choices(keys, weights=ps, k=1)[0]

        return _f
    else:
        if not fallback_choices:

            def _g():
                return None

            return _g
        if fallback_p is None:

            def _f():
                return random.choice(fallback_choices)

            return _f
        else:

            def _f():
                return random.choices(fallback_choices, weights=fallback_p, k=1)[0]

            return _f


# ----------------------------- Hot-path construction (data-driven) -----------------------------


def build_hot_sequence(
    station_ids: List[int],
    hot_ends: Set[int],
    a_len: int,
    base_time: datetime,
    bike_id: int,
    cross_midnight_prob: float,
    catalog: Dict[int, Tuple[str, float, float]],
    duration_profile: Dict,
    non_hot_station_sampler,
) -> List[List]:
    """
    Build one hot-path sequence using empirical duration distribution:
    - 'a_len' hops for the 'a+' part (no cycles, non-hot stations only).
    - Final 'b' ends at a station in 'hot_ends'.
    - Entire sequence within ~60 minutes (best-effort using shorter durations).
    - Random small gaps to mimic reality (1..4 min).
    """
    rows: List[List] = []

    # Choose start time; some sequences biased to start near midnight
    t = base_time
    if random.random() < cross_midnight_prob:
        t = base_time.replace(hour=23, minute=random.choice([40, 45, 50, 55]), second=0)

    non_hot = [sid for sid in station_ids if sid not in hot_ends]
    if len(set(non_hot)) < a_len + 1:
        raise ValueError("Not enough non-hot stations to build a no-cycle path of requested length.")

    visited = set()
    # Choose first non-hot start
    current_sid = non_hot_station_sampler()
    while current_sid in hot_ends:
        current_sid = non_hot_station_sampler()
    visited.add(current_sid)

    total_elapsed = 0
    for _ in range(a_len):
        # Next must be different and not visited
        candidates = [sid for sid in non_hot if sid != current_sid and sid not in visited]
        if not candidates:
            candidates = [sid for sid in non_hot if sid not in visited] or [sid for sid in non_hot]
        next_sid = random.choice(candidates)
        visited.add(next_sid)

        hop_min = max(5, min(12, sample_duration_min_from_profile(duration_profile)))
        total_elapsed += hop_min
        rows.append(
            make_trip_row(
                t,
                hop_min,
                current_sid,
                next_sid,
                bike_id,
                catalog,
                usertype_sampler=lambda: "Subscriber",  # hot paths often look like commuter flows; tweak if needed
                birthyear_sampler=lambda: 1985,
                gender_sampler=lambda: 1,
            )
        )

        gap = random.randint(1, 4)
        t = t + timedelta(minutes=hop_min + gap)
        total_elapsed += gap
        current_sid = next_sid

    # Final 'b' hop: end in a hot-end station
    remaining = max(5, 55 - total_elapsed)
    b_duration = min(12, remaining)
    end_b = random.choice(list(hot_ends))
    rows.append(
        make_trip_row(
            t,
            b_duration,
            current_sid,
            end_b,
            bike_id,
            catalog,
            usertype_sampler=lambda: "Subscriber",
            birthyear_sampler=lambda: 1985,
            gender_sampler=lambda: 1,
        )
    )

    return rows


# ----------------------------- Noise trips (data-driven) -----------------------------


def build_noise_trip(
    station_ids: List[int],
    base_time: datetime,
    bike_id: int,
    hot_ends: Set[int],
    last_by_bike: Dict[int, Tuple[datetime, int]],
    catalog: Dict[int, Tuple[str, float, float]],
    duration_profile: Dict,
    start_station_sampler,
    end_station_sampler,
    usertype_sampler,
    birthyear_sampler,
    gender_sampler,
) -> List:
    """
    Build a single noise trip that avoids triggering the hot pattern:
    - If the same bikeid is re-used, break adjacency and window semantics:
      * Ensure start station != last end station for that bike.
      * Ensure time gap > 70 minutes so it cannot join a SEQ WITHIN 1h.
    - Prefer end station NOT in hot_ends.
    - Durations sampled from empirical distribution (robust clipped).
    """
    # Compute base time with enforced >70min gap if bike seen before
    if bike_id in last_by_bike:
        last_time, last_end_sid = last_by_bike[bike_id]
        base_time = last_time + timedelta(minutes=71)
        # Start at a station different from the last end to break chaining
        start_sid = start_station_sampler()
        if start_sid == last_end_sid:
            # Pick alternative if possible
            alternatives = [sid for sid in station_ids if sid != last_end_sid]
            if alternatives:
                start_sid = random.choice(alternatives)
    else:
        start_sid = start_station_sampler()

    # Ensure start_sid exists
    if start_sid not in station_ids:
        start_sid = random.choice(station_ids)

    # Choose end station; avoid hot-ends if possible
    tries = 0
    end_sid = end_station_sampler()
    while end_sid in hot_ends and tries < 5:
        end_sid = end_station_sampler()
        tries += 1
    if end_sid == start_sid:
        # Change end to a different station
        alternatives = [sid for sid in station_ids if sid != start_sid and sid not in hot_ends]
        if alternatives:
            end_sid = random.choice(alternatives)

    duration = sample_duration_min_from_profile(duration_profile)
    row = make_trip_row(
        base_time,
        duration,
        start_sid,
        end_sid,
        bike_id,
        catalog,
        usertype_sampler=usertype_sampler,
        birthyear_sampler=birthyear_sampler,
        gender_sampler=gender_sampler,
    )

    # Update last_by_bike
    start_dt = dt_parse_flex(row[1])
    stop_dt = dt_parse_flex(row[2])
    last_by_bike[bike_id] = (stop_dt, end_sid)

    return row


# ----------------------------- Main -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)  # total rows to generate
    ap.add_argument("--x", type=int, required=True)  # hot-path sequences to generate
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="scripts/bike_events.csv")
    ap.add_argument("--base-date", type=str, default="2014-01-31")
    args = ap.parse_args()
    random.seed(args.seed)

    # 1) Profile real data
    prof = profile_real_csv(args.source_csv)
    catalog = prof["catalog"]
    if not catalog:
        raise ValueError("No stations found in source CSV; cannot proceed.")
    station_ids = list(catalog.keys())

    # 2) Infer hot ends if required
    hot_ends_cfg = parse_hot_ends(args.hot_ends)
    if hot_ends_cfg is None:
        # infer from top-K end stations
        end_pairs_sorted = sorted(prof["end_station_pairs"], key=lambda x: x[1], reverse=True)
        top_k = [sid for sid, _ in end_pairs_sorted[: max(1, args.hot_ends_topk)]]
        hot_ends: Set[int] = set(top_k)
    else:
        hot_ends: Set[int] = hot_ends_cfg
        missing = [sid for sid in hot_ends if sid not in catalog]
        if missing:
            raise ValueError(f"Hot-end station ids not found in source stations: {missing}")

    # 3) Prepare 'a' length distribution
    len_dist = parse_len_dist(args.hot_len_dist)

    # 4) Build samplers from profile
    duration_profile = prof["duration_profile"]

    # Hour sampler: override or empirical
    if args.override_hour:
        hour_override = []
        for part in args.override_hour.split(","):
            k, v = part.split(":")
            hour_override.append((int(k.strip()), float(v.strip())))
        # Normalize
        total = sum(p for _, p in hour_override)
        hour_pairs = [(h, p / total) for h, p in hour_override]
    else:
        hour_pairs = prof["hour_pairs"]
    hour_sampler = build_simple_sampler(hour_pairs, list(range(24)))

    weekday_sampler = build_simple_sampler(prof["weekday_pairs"], list(range(7)))
    month_sampler = build_simple_sampler(prof["month_pairs"], list(range(1, 13)))

    # Station samplers
    start_station_sampler = build_simple_sampler(prof["start_station_pairs"], station_ids)
    end_station_sampler = build_simple_sampler(prof["end_station_pairs"], station_ids)

    # Non-hot station sampler (bias towards popular start stations)
    non_hot_station_ids = [sid for sid in station_ids if sid not in hot_ends]
    # Build a reduced pair list removing hot ends
    start_pairs_non_hot = [(sid, p) for sid, p in prof["start_station_pairs"] if sid in non_hot_station_ids]
    non_hot_station_sampler = build_simple_sampler(start_pairs_non_hot, non_hot_station_ids)

    # User attribute samplers
    usertype_sampler = build_simple_sampler(prof["usertype_pairs"], USERS_FALLBACK)
    birthyear_sampler = build_simple_sampler(prof["birthyear_pairs"], BIRTH_YEARS_FALLBACK)
    gender_sampler = build_simple_sampler(prof["gender_pairs"], GENDERS_FALLBACK)

    # BikeID sampler
    bikeid_mode = args.bikeid_mode.lower()
    bikeid_sampler = build_noise_bike_sampler(
        mode=bikeid_mode,
        start=args.bikeid_start,
        end=args.bikeid_end,
        zipf_s=args.bikeid_zipf_s,
        empirical_pairs=prof["bikeid_pairs"],
    )

    base_day = datetime.strptime(args.base_date, "%Y-%m-%d")

    rows: List[List] = []

    # 5) Assign UNIQUE bikeids for each hot-path sequence to avoid cross-sequence interference.
    hot_bike_base = 20000
    hot_bike_ids = [hot_bike_base + i for i in range(args.x)]

    # 6) Build X hot-path sequences (no cycles), using data-driven durations/stations
    for i in range(args.x):
        # Choose a plausible start hour from empirical distribution
        h = hour_sampler()
        seq_start = base_day.replace(hour=int(h) if isinstance(h, int) else 8, minute=0, second=0)
        a_len = sample_len(len_dist)
        rows.extend(
            build_hot_sequence(
                station_ids=station_ids,
                hot_ends=hot_ends,
                a_len=a_len,
                base_time=seq_start,
                bike_id=hot_bike_ids[i],
                cross_midnight_prob=args.cross_midnight_prob,
                catalog=catalog,
                duration_profile=duration_profile,
                non_hot_station_sampler=non_hot_station_sampler,
            )
        )

    used = len(rows)
    remaining = args.n - used
    if remaining < 0:
        raise ValueError(f"Requested n={args.n} smaller than hot-path rows {used}.")

    # 7) Generate remaining noise trips
    last_by_bike: Dict[int, Tuple[datetime, int]] = {}

    for j in range(remaining):
        # Distribute noise trips by empirical hour + random minute/second
        hour = hour_sampler()
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        base_time = base_day.replace(hour=int(hour) if isinstance(hour, int) else 9, minute=minute, second=second)
        bike_id = bikeid_sampler.next()

        row = build_noise_trip(
            station_ids=station_ids,
            base_time=base_time,
            bike_id=bike_id,
            hot_ends=hot_ends,
            last_by_bike=last_by_bike,
            catalog=catalog,
            duration_profile=duration_profile,
            start_station_sampler=start_station_sampler,
            end_station_sampler=end_station_sampler,
            usertype_sampler=usertype_sampler,
            birthyear_sampler=birthyear_sampler,
            gender_sampler=gender_sampler,
        )
        rows.append(row)

    # 8) Shuffle for realism
    random.shuffle(rows)

    # 9) Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)

    # 10) Report
    print(f"Wrote {len(rows)} rows to {args.out} with exactly {args.x} hot-path matches.")
    print(f"Hot-ends used: {sorted(list(hot_ends))}")


if __name__ == "__main__":
    main()
