# -*- encoding: utf-8 -*-
# File: main.py

import pathlib
import pandas as pd
import sys
import time

# Ensure the project root is in sys.path for imports
nb_dir = pathlib.Path(__file__).parent if "__file__" in globals() else pathlib.Path.cwd()
project_root = str((nb_dir / "..").resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import timedelta
from base.Pattern import Pattern
from base.PatternStructure import KleeneClosureOperator, PrimitiveEventStructure, SeqOperator
from CEP import CEP
from condition.CompositeCondition import AndCondition
from condition.Condition import SimpleCondition, Variable
from condition.KCCondition import KCIndexCondition
from stream.FileStream import FileOutputStream
from scripts.BikeTripUtils import DataFrameInputStream, BikeTripDataFormatter, BikeTripEventTypeClassifier

# PATTERN SEQ (BikeTrip+ a[], BikeTrip b)
# WHERE a[i+1].bike = a[i].bike AND b.end in {7,8,9}
# AND a[last].bike = b.bike AND a[i+1].start = a[i].end
# WITHIN 1h
# RETURN (a[1].start, a[i].end, b.end)

# --- Structure: SEQ( (BikeTrip a)+ , (BikeTrip b) ) ---
a_prim = PrimitiveEventStructure("BikeTrip", "a")
b_prim = PrimitiveEventStructure("BikeTrip", "b")
a_plus = KleeneClosureOperator(arg=a_prim, min_size=1, max_size=10)
structure = SeqOperator(a_plus, b_prim)

# --- Variables bound by the engine: {"a": List[dict], "b": dict} ---
var_a_seq = Variable("a", getattr_func=lambda seq: seq)  # the whole sequence of 'a'
var_a_last_bike = Variable(
    "a", getattr_func=lambda seq: (seq[-1].get("bikeid") if isinstance(seq, list) and len(seq) > 0 else None)
)
var_b_bike = Variable("b", getattr_func=lambda ev: ev.get("bikeid"))
var_b_end_station_id = Variable("b", getattr_func=lambda ev: ev.get("end station id"))

# --- Conditions ---
# 1) For all i: a[i+1].bikeid = a[i].bikeid AND a[i+1].starttime = a[i].stoptime

def kc_compare_op(a, b):
    print(f"Comparing: a={a}, b={b}")
    return a["bikeid"] == b["bikeid"] and a["end station id"] == b["start station id"]

chain_cond = KCIndexCondition(
    names={"a"},
    getattr_func=lambda elem: {
        "bikeid": elem.get("bikeid"),
        "start station id": elem.get("start station id"),
        "end station id": elem.get("end station id"),
    },
    relation_op=kc_compare_op,
    offset=1,
)

# 2) a[last].bikeid = b.bikeid
last_matches_bike = SimpleCondition(
    var_a_last_bike, var_b_bike, relation_op=lambda last_bike, b_bike: last_bike == b_bike
)

# 3) b.end station id in {265}
b_end_station_in_set = SimpleCondition(var_b_end_station_id, relation_op=lambda sid: sid in {265})


bike_trip_pattern = Pattern(
    structure,
    # AndCondition(chain_cond),
    AndCondition(chain_cond, last_matches_bike, b_end_station_in_set),
    timedelta(hours=1),
)

print(bike_trip_pattern)

cep = CEP([bike_trip_pattern])
cur_path = pathlib.Path(__file__).parent.resolve()

data_path = cur_path / "demo_data.csv"

df = pd.read_csv(data_path)
events = DataFrameInputStream(df)

start = time.monotonic_ns()
cep.run(events, FileOutputStream("./", "output.txt"), BikeTripDataFormatter(BikeTripEventTypeClassifier()))
end = time.monotonic_ns()
print(f"Time taken: {(end - start) / 1e9} seconds")
