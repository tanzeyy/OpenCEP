# -*- encoding: utf-8 -*-
# File: main.py
import argparse
import pathlib
import sys
import time

import pandas as pd

# Ensure the project root is in sys.path for imports
nb_dir = pathlib.Path(__file__).parent if "__file__" in globals() else pathlib.Path.cwd()
project_root = str((nb_dir / "..").resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import timedelta

from base.Pattern import Pattern
from base.PatternStructure import KleeneClosureOperator
from base.PatternStructure import PrimitiveEventStructure
from base.PatternStructure import SeqOperator
from CEP import CEP
from condition.CompositeCondition import AndCondition
from condition.Condition import SimpleCondition
from condition.Condition import Variable
from condition.KCCondition import KCIndexCondition
from misc import DefaultConfig
from misc.ConsumptionPolicy import ConsumptionPolicy
from misc.SelectionStrategies import SelectionStrategies
from parallel.ParallelExecutionParameters import DataParallelExecutionParameters
from parallel.ParallelExecutionParameters import DataParallelExecutionParametersHirzelAlgorithm
from parallel.ParallelExecutionParameters import DataParallelExecutionParametersHyperCubeAlgorithm
from parallel.ParallelExecutionParameters import DataParallelExecutionParametersRIPAlgorithm
from project.utils.BikeTripUtils import BikeTripDataFormatter
from project.utils.BikeTripUtils import BikeTripEventTypeClassifier
from project.utils.BikeTripUtils import DataFrameInputStream
from stream.FileStream import FileOutputStream

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
parser.add_argument("--threads", type=int, default=8, help="Number of execution threads")
parser.add_argument("--input", type=str, default="bike_events_2x.csv", help="Path to input CSV file")
parser.add_argument("--output", type=str, default="output.csv", help="Path for emitted text output")
parser.add_argument(
    "--parallel_algorithm",
    type=str,
    default="hirzel",
    choices=["hirzel", "hyper_cube", "rip", "none"],
    help="Data parallel algorithm to use",
)
parser.add_argument("--key", type=str, default="bikeid", help="Hirzel algorithm parameter: key for grouping events")
parser.add_argument("--multiple", type=float, default=2.0, help="RIP algorithm parameter: multiple")
parser.add_argument(
    "--attributes_dict",
    type=str,
    default=None,
    help="Attributes dictionary, in format: key1:type1,key2:type2,...",
)
parser.add_argument("--hot_ends", type=str, default=None, help="Comma-separated list of hot end station ids")
args = parser.parse_args()

# -------------------- LOAD INPUT DATA --------------------
cur_path = pathlib.Path(__file__).parent.resolve()
data_path = cur_path / "data" / args.input
if not data_path.exists():
    raise FileNotFoundError(f"Input file not found: {data_path}")
df = pd.read_csv(data_path)
events = DataFrameInputStream(df)

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
if args.hot_ends:
    try:
        hot_ends_set = {int(sid.strip()) for sid in args.hot_ends.split(",") if sid.strip()}
        if len(hot_ends_set) == 0:
            raise ValueError("Empty hot_ends")
    except Exception as e:
        raise ValueError(f"Invalid hot_ends format: {args.hot_ends}") from e
else:
    hot_ends_set = {293, 497, 521}  # default hot end station ids
b_end_station_in_set = SimpleCondition(var_b_end_station_id, relation_op=lambda sid: sid in hot_ends_set)

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

if args.parallel_algorithm == "hirzel":
    parallel_execution_params = DataParallelExecutionParametersHirzelAlgorithm(
        platform=DefaultConfig.ParallelExecutionPlatforms.THREADING,
        units_number=args.threads,
        key=args.key,
    )
elif args.parallel_algorithm == "rip":
    parallel_execution_params = DataParallelExecutionParametersRIPAlgorithm(
        platform=DefaultConfig.ParallelExecutionPlatforms.THREADING,
        units_number=args.threads,
        multiple=args.multiple,
    )
elif args.parallel_algorithm == "hyper_cube":
    if args.attributes_dict:
        try:
            attr_dict = {}
            for item in args.attributes_dict.split(","):
                k, v = item.split(":")
                attr_dict[k.strip()] = int(v.strip())
            if len(attr_dict) == 0:
                raise ValueError("Empty attributes_dict")
        except Exception as e:
            raise ValueError(f"Invalid attributes_dict format: {args.attributes_dict}") from e
    else:
        attr_dict = {"bikeid": 4, "start station id": 8, "end station id": 8}  # default: 256 partitions
    parallel_execution_params = DataParallelExecutionParametersHyperCubeAlgorithm(
        platform=DefaultConfig.ParallelExecutionPlatforms.THREADING,
        units_number=int(args.threads),
        attributes_dict=attr_dict,
    )
else:
    parallel_execution_params = None  # no parallelism

# -------------------- CEP ENGINE SETUP --------------------
cep = CEP(
    patterns=[bike_trip_pattern],
    eval_mechanism_params=eval_mechanism_params,
    parallel_execution_params=parallel_execution_params,
    pattern_preprocessing_params=pattern_preprocessing_params,
)

# -------------------- RUN & MEASURE --------------------
start = time.monotonic_ns()
cep.run(events, FileOutputStream("./", args.output), BikeTripDataFormatter(BikeTripEventTypeClassifier()))
end = time.monotonic_ns()

print(f"Pattern detection on dataset {args.input}.")
print(f"Load shedding strategy: {args.policy}.")
print(f"Text output written to {args.output}.")
print(f"Time taken: {(end - start) / 1e9} seconds")
