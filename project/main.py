# -*- encoding: utf-8 -*-
# File: main.py
import argparse
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
from misc.ConsumptionPolicy import ConsumptionPolicy
from misc.SelectionStrategies import SelectionStrategies
from condition.KCCondition import KCIndexCondition
from misc import DefaultConfig
from parallel.ParallelExecutionParameters import DataParallelExecutionParameters
from parallel.ParallelExecutionParameters import DataParallelExecutionParametersHirzelAlgorithm
from stream.FileStream import FileOutputStream
from project.utils.BikeTripUtils import DataFrameInputStream, BikeTripDataFormatter, BikeTripEventTypeClassifier

# -------------------- ARGUMENT PARSER --------------------
parser = argparse.ArgumentParser(description="OpenCEP Pattern Matcher")
parser.add_argument(
    "--policy",
    type=str,
    default="none",
    choices=["none", "match_any", "match_next", "match_single"],
    help="Consumption policy",
)
parser.add_argument(
    "--freeze",
    type=str,
    default="none",
    choices=["a", "b"],
    help="Prohibit creation of new partial matches from the point a new chosen event (a or b) is accepted "
    "and until it is either matched or expired.",
)
parser.add_argument("--threads", type=int, default=12, help="Number of execution threads")
parser.add_argument("--input", type=str, default="bike_events_2x.csv", help="Path to input CSV file")
parser.add_argument("--output", type=str, default="output.csv", help="Path for emitted text output")
args = parser.parse_args()

# -------------------- LOAD INPUT DATA --------------------


def build_events_stream(data_path: pathlib.Path) -> DataFrameInputStream:
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")
    df = pd.read_csv(data_path)
    events = DataFrameInputStream(df)

    return events


# -------------------- DEFINE PATTERN --------------------
# PATTERN SEQ (BikeTrip+ a[], BikeTrip b)
# WHERE a[i+1].bike = a[i].bike AND b.end in {7,8,9}
# AND a[last].bike = b.bike AND a[i+1].start = a[i].end
# WITHIN 1h
# RETURN (a[1].start, a[i].end, b.end)

# --- Structure: SEQ( (BikeTrip a)+ , (BikeTrip b) ) ---
a_prim = PrimitiveEventStructure(event_type="BikeTrip", name="a")
b_prim = PrimitiveEventStructure(event_type="BikeTrip", name="b")
a_plus = KleeneClosureOperator(arg=a_prim, min_size=1, max_size=10)  # limit the sequence length
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
chain_cond = KCIndexCondition(
    names={"a"},
    getattr_func=lambda elem: {
        "bikeid": elem.get("bikeid"),
        "start station id": elem.get("start station id"),
        "end station id": elem.get("end station id"),
    },
    relation_op=lambda a, b: a["bikeid"] == b["bikeid"] and a["end station id"] == b["start station id"],
    offset=1,
)

# 2) a[last].bikeid = b.bikeid
last_matches_bike = SimpleCondition(
    var_a_last_bike, var_b_bike, relation_op=lambda last_bike, b_bike: last_bike == b_bike
)

# 3) b.end station id in {7,8,9}
b_end_station_in_set = SimpleCondition(var_b_end_station_id, relation_op=lambda sid: sid in {293, 497, 521})

# -------------------- LOAD SHEDDING POLICY --------------------
if args.policy == "match_any":
    policy = ConsumptionPolicy(primary_selection_strategy=SelectionStrategies.MATCH_ANY, freeze=args.freeze)
elif args.policy == "match_single":
    # MATCH_ANY (the default) as the primary strategy.
    # MATCH_SINGLE forces each primitive event into at most one full match.
    policy = ConsumptionPolicy(
        primary_selection_strategy=SelectionStrategies.MATCH_ANY,
        secondary_selection_strategy=SelectionStrategies.MATCH_SINGLE,
        freeze=args.freeze,
    )
elif args.policy == "match_next":
    # MATCH_ANY (the default) as the primary strategy.
    # MATCH_NEXT limits all primitive events to only appear in the next possible match
    policy = ConsumptionPolicy(
        primary_selection_strategy=SelectionStrategies.MATCH_ANY,
        secondary_selection_strategy=SelectionStrategies.MATCH_NEXT,
        freeze=args.freeze,
    )
else:
    policy = None  # no consumption policy

bike_trip_pattern = Pattern(
    structure,
    AndCondition(chain_cond, last_matches_bike, b_end_station_in_set),
    timedelta(hours=1),
    consumption_policy=policy,
)

# -------------------- PARALLELISM SETUP --------------------
# thread parallelism via Hirzel et al. algorithm
eval_mechanism_params = None
pattern_preprocessing_params = None
parallel_execution_params = DataParallelExecutionParametersHirzelAlgorithm(
    platform=DefaultConfig.ParallelExecutionPlatforms.THREADING,
    units_number=args.threads,
    key="bikeid",
)


# -------------------- CEP ENGINE SETUP --------------------
def build_cep() -> CEP:
    return CEP(
        patterns=[bike_trip_pattern],
        eval_mechanism_params=eval_mechanism_params,
        parallel_execution_params=parallel_execution_params,
        pattern_preprocessing_params=pattern_preprocessing_params,
    )


cep = build_cep()
# -------------------- RUN & MEASURE --------------------

# Define paths relative to this script's location
cur_path = pathlib.Path(__file__).parent.resolve()
data_path = cur_path / "data" / args.input
events = build_events_stream(data_path)

fmt = BikeTripDataFormatter(BikeTripEventTypeClassifier())

start = time.monotonic_ns()
cep.run(
    events,
    FileOutputStream("./", args.output, is_async=True),
    fmt,
)
end = time.monotonic_ns()

# print(f"Pattern detection on dataset {args.input}.")
# print(f"Load shedding strategy: {args.policy}.")
# print(f"Text output written to {args.output}.")
# print(f"Time taken: {(end - start) / 1e9} seconds")
