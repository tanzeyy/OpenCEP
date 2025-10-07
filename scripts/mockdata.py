#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate synthetic bike trip CSV data without external station file.

Features:
- Built-in station catalog (10+ fake stations with id/name/coords).
- Integer bikeid.
- Hot-path sequences that satisfy SEQ(a+, b) pattern WITHIN 1h.
- Some hot paths cross midnight.
- Exact column header/order as CitiBike-style dataset.
"""

import csv
import argparse
import random
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Set

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

# Built-in stations
STATIONS: Dict[int, Tuple[str, float, float]] = {
    7: ("Washington Square E", 40.73049, -73.99572),
    28: ("Stanton St & Chrystie St", 40.72229, -73.99147),
    8: ("Hudson St & Reade St", 40.71534, -74.00915),
    33: ("Greenwich Ave & 8 Ave", 40.73902, -74.00264),
    9: ("W 52 St & 9 Ave", 40.76440, -73.98780),
    41: ("Grand St & Elizabeth St", 40.71850, -73.99720),
    44: ("St James Pl & Pearl St", 40.71117, -74.00016),
    47: ("E 20 St & Park Ave", 40.73818, -73.98433),
    52: ("E 33 St & 1 Ave", 40.74402, -73.97196),
    60: ("Columbus Ave & W 72 St", 40.77813, -73.97371),
}

# hot end station ids
HOT_ENDS: Set[int] = {7, 8, 9}

USERS = ["Subscriber", "Customer"]
GENDERS = [0, 1, 2]
BIRTH_YEARS = list(range(1950, 2006))


def station_tuple(sid: int) -> Tuple[int, str, float, float]:
    name, lat, lon = STATIONS[sid]
    return sid, name, lat, lon


def make_trip_row(
    start_dt: datetime,
    duration_min: int,
    start_sid: int,
    end_sid: int,
    bike_id: int,
) -> List:
    duration_sec = max(1, duration_min * 60)
    end_dt = start_dt + timedelta(seconds=duration_sec)

    ssid, sname, slat, slon = station_tuple(start_sid)
    esid, ename, elat, elon = station_tuple(end_sid)

    usertype = random.choice(USERS)
    birth_year = random.choice(BIRTH_YEARS)
    gender = random.choice(GENDERS)

    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    stop_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

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


def pick_different(station_ids: List[int], current: int) -> int:
    choices = [sid for sid in station_ids if sid != current]
    return random.choice(choices) if choices else current


def build_hot_sequence(
    station_ids: List[int],
    base_time: datetime,
    bike_id: int,
    hot_ends: Set[int],
) -> List[List]:
    rows: List[List] = []
    a_len = random.randint(1, 3)

    t = base_time
    if random.random() < 0.4:  # some sequences start near midnight
        t = base_time.replace(hour=23, minute=random.choice([40, 45, 50, 55]), second=0)

    current_sid = random.choice(station_ids)
    total_elapsed = 0

    for _ in range(a_len):
        next_sid = pick_different(station_ids, current_sid)
        hop_min = random.randint(5, 12)
        total_elapsed += hop_min
        rows.append(make_trip_row(t, hop_min, current_sid, next_sid, bike_id))
        gap = random.randint(1, 4)
        t = t + timedelta(minutes=hop_min + gap)
        total_elapsed += gap
        current_sid = next_sid

    remaining = max(5, 55 - total_elapsed)
    b_duration = min(12, remaining)
    end_b = random.choice(list(hot_ends))
    rows.append(make_trip_row(t, b_duration, current_sid, end_b, bike_id))
    return rows


def build_noise_trip(
    station_ids: List[int],
    base_time: datetime,
    bike_id: int,
    hot_ends: Set[int],
) -> List:
    start_sid = random.choice(station_ids)
    end_candidates = [sid for sid in station_ids if sid != start_sid and sid not in hot_ends]
    end_sid = random.choice(end_candidates) if end_candidates else pick_different(station_ids, start_sid)
    duration = random.randint(6, 35)
    return make_trip_row(base_time, duration, start_sid, end_sid, bike_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="bike_events.csv")
    ap.add_argument("--base-date", type=str, default="2014-01-31")
    args = ap.parse_args()

    random.seed(args.seed)
    station_ids = list(STATIONS.keys())

    if args.n < args.x * 2:
        raise ValueError(f"n={args.n} too small for x={args.x}. Need at least {args.x*2} rows.")

    base_day = datetime.strptime(args.base_date, "%Y-%m-%d")

    rows: List[List] = []
    bike_seed = 20000

    for i in range(args.x):
        bike_id = bike_seed + i
        seq_start = base_day.replace(hour=8 + (i % 10), minute=0, second=0)
        rows.extend(build_hot_sequence(station_ids, seq_start, bike_id, HOT_ENDS))

    used = len(rows)
    remaining = args.n - used

    for j in range(remaining):
        hour = 7 + (j % 12)
        minute = random.randint(0, 59)
        base_time = base_day.replace(hour=hour, minute=minute, second=random.randint(0, 59))
        bike_id = bike_seed + args.x + j
        rows.append(build_noise_trip(station_ids, base_time, bike_id, HOT_ENDS))

    random.shuffle(rows)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out} with exactly {args.x} hot-path matches.")


if __name__ == "__main__":
    main()
