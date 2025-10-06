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
from misc import DefaultConfig
from parallel.ParallelExecutionParameters import DataParallelExecutionParameters
from parallel.ParallelExecutionParameters import DataParallelExecutionParametersHirzelAlgorithm
from parallel.ParallelExecutionParameters import DataParallelExecutionParametersHyperCubeAlgorithm
from parallel.ParallelExecutionParameters import DataParallelExecutionParametersRIPAlgorithm
from parallel.ParallelExecutionParameters import ParallelExecutionParameters
from stream.FileStream import FileOutputStream
from scripts.BikeTripUtils import DataFrameInputStream, BikeTripDataFormatter, BikeTripEventTypeClassifier
from stream.Stream import InputStream

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

# 3) b.end station id in {265}
b_end_station_in_set = SimpleCondition(var_b_end_station_id, relation_op=lambda sid: sid in {265})


bike_trip_pattern = Pattern(
    structure,
    # AndCondition(chain_cond),
    AndCondition(chain_cond, last_matches_bike, b_end_station_in_set),
    timedelta(hours=1),
)

print(bike_trip_pattern)

eval_mechanism_params = None
pattern_preprocessing_params = None
parallel_execution_params = DataParallelExecutionParametersHirzelAlgorithm(
    platform=DefaultConfig.ParallelExecutionPlatforms.THREADING,
    units_number=2,
    key='start station id',
)
# parallel_execution_params = None

cep = CEP(
    [bike_trip_pattern],
    eval_mechanism_params,
    parallel_execution_params,
    pattern_preprocessing_params,
)


class DataFrameInputStream(InputStream):
    """Emit each DataFrame row as a dict payload (not a CSV string)."""

    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        # Ensure all columns are string-key accessible and datetimes are stringified
        df = dataframe.copy()
        # If your CSV loader already gives strings for time columns, the following is safe/no-op.
        if pd.api.types.is_datetime64_any_dtype(df.get("starttime", pd.Series([], dtype="datetime64[ns]"))):
            df["starttime"] = df["starttime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        if pd.api.types.is_datetime64_any_dtype(df.get("stoptime", pd.Series([], dtype="datetime64[ns]"))):
            df["stoptime"] = df["stoptime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        for _, row in df.iterrows():
            self._stream.put(row.to_dict())
        self.close()


cur_path = pathlib.Path(__file__).parent.resolve()

# data_path = "/Users/zheyue/Workspace/Projects/cs-e4780/2014-citibike-tripdata/3_March/201403-citibike-tripdata_1.csv"
data_path = cur_path / "demo_data.csv"

df = pd.read_csv(data_path).iloc[:20]
events = DataFrameInputStream(df)

start = time.monotonic_ns()
cep.run(events, FileOutputStream("./", "output.txt"), BikeTripDataFormatter(BikeTripEventTypeClassifier()))
end = time.monotonic_ns()
print(f"Time taken: {(end - start) / 1e9} seconds")
