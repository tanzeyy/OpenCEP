# scripts/BikeTripUtils.py

import pandas as pd
from datetime import datetime
from base.DataFormatter import DataFormatter, EventTypeClassifier
from stream.Stream import InputStream

class DataFrameInputStream(InputStream):
    """Emit each DataFrame row as a dict payload (not a CSV string)."""

    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        df = dataframe.copy()
        if pd.api.types.is_datetime64_any_dtype(df.get("starttime", pd.Series([], dtype="datetime64[ns]"))):
            df["starttime"] = df["starttime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        if pd.api.types.is_datetime64_any_dtype(df.get("stoptime", pd.Series([], dtype="datetime64[ns]"))):
            df["stoptime"] = df["stoptime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        for _, row in df.iterrows():
            self._stream.put(row.to_dict())
        self.close()

class BikeTripDataFormatter(DataFormatter):
    """
    Format bike trip event payloads.
    """
    _TS_FMT = "%Y-%m-%d %H:%M:%S"

    def get_event_timestamp(self, event_payload: dict) -> datetime:
        ts_str = event_payload.get("starttime") or event_payload.get("stoptime")
        if not ts_str:
            raise KeyError("Neither 'starttime' nor 'stoptime' found in payload")
        return datetime.strptime(ts_str, self._TS_FMT)

    def get_event_type(self, event_payload: dict) -> str:
        return "BikeTrip"

    def parse_event(self, raw_data):
        return raw_data

class BikeTripEventTypeClassifier(EventTypeClassifier):
    """
    Classify all events as "BikeTrip".
    """
    def get_event_type(self, event_payload: dict) -> str:
        return "BikeTrip"