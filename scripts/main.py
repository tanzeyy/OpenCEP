# -*- encoding: utf-8 -*-
# File: main.py

# Insert package path
import pathlib
import sys

nb_dir = pathlib.Path(__file__).parent if "__file__" in globals() else pathlib.Path.cwd()
project_root = str((nb_dir / "..").resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime
from datetime import timedelta

import pandas as pd
from base.DataFormatter import DataFormatter
from base.DataFormatter import EventTypeClassifier
from base.Pattern import Pattern
from base.PatternStructure import KleeneClosureOperator
from base.PatternStructure import PrimitiveEventStructure
from base.PatternStructure import SeqOperator
from CEP import CEP
from condition.CompositeCondition import AndCondition
from condition.Condition import SimpleCondition
from condition.Condition import Variable
from stream.FileStream import FileOutputStream
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


def chain_eq_op(*seq):
    seq = list(seq)
    # print(f"type: seq={type(seq)}, len={len(seq)}")
    # print(f"seq={seq}")

    if len(seq) == 1:  # only one element in the sequence, no chaining to check
        return True

    if not (isinstance(seq, list) and all(isinstance(e, dict) for e in seq) and len(seq) >= 2):
        return False
    return all(
        (seq[i + 1].get("bikeid") == seq[i].get("bikeid"))
        and (seq[i + 1].get("start station id") == seq[i].get("end station id"))
        for i in range(len(seq) - 1)
    )


chain_cond = SimpleCondition(
    var_a_seq,
    relation_op=chain_eq_op,
)

# 2) a[last].bikeid = b.bikeid
last_matches_bike = SimpleCondition(
    var_a_last_bike, var_b_bike, relation_op=lambda last_bike, b_bike: last_bike == b_bike
)

# 3) b.end station id in {7,8,9}
b_end_station_in_set = SimpleCondition(var_b_end_station_id, relation_op=lambda sid: sid in {265})


bike_trip_pattern = Pattern(
    structure,
    AndCondition(chain_cond, last_matches_bike, b_end_station_in_set),
    # AndCondition(last_matches_bike, b_end_station_in_set),
    timedelta(hours=1),
)

print(bike_trip_pattern)

cep = CEP([bike_trip_pattern])


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

data_path = cur_path / "demo_data.csv"

df = pd.read_csv(data_path)
events = DataFrameInputStream(df)


class BikeTripDataFormatter(DataFormatter):
    """
    Data formatter for CitiBike-like CSV rows.
    It expects a dict payload with keys such as:
      'starttime', 'stoptime', 'bikeid', etc.
    """

    _TS_FMT = "%Y-%m-%d %H:%M:%S"

    def get_event_timestamp(self, event_payload: dict):
        ts_str = event_payload.get("starttime") or event_payload.get("stoptime")
        if not ts_str:
            raise KeyError("Neither 'starttime' nor 'stoptime' found in payload")
        return datetime.strptime(ts_str, self._TS_FMT)

    def get_event_type(self, event_payload: dict):
        return "BikeTrip"

    def parse_event(self, raw_data):
        return raw_data


class BikeTripEventTypeClassifier(EventTypeClassifier):
    def get_event_type(self, event_payload: dict):
        """
        The type of a bike trip event is equal to the bike ID.
        """
        return "BikeTrip"  # All events are of type "BikeTrip"


import time

start = time.monotonic_ns()
cep.run(events, FileOutputStream("./", "output.txt"), BikeTripDataFormatter(BikeTripEventTypeClassifier()))
end = time.monotonic_ns()
print(f"Time taken: {(end - start) / 1e9} seconds")
