# CS-E4780 Scalable Systems and Data Management Course Project
## Efficient Pattern Detection over Data Streams

## Pattern
```
# PATTERN SEQ (BikeTrip+ a[], BikeTrip b) -> Look for one or more (+) BikeTrip events (a[]), followed by another BikeTrip (b).
# WHERE a[i+1].bike = a[i].bike AND b.end in {7,8,9}
# AND a[last].bike = b.bike AND a[i+1].start = a[i].end
# WITHIN 1h
# RETURN (a[1].start, a[i].end, b.end)
```

## Structure of the Directory `project`

- `project/main.py` — the application under test (CLI).
- `project/performance_eval/`:
  - `measure_recall.py` — run baseline and latency-targeted experiments and compute recall.
- `project/utils/` — utility scripts for data generation, preprocessing, and postprocessing.'

- Result files produced by the scripts (examples): `baseline_matches.csv`, `recall_summary.csv`. 

## 
We use the Kleene closure in OpenCEP, which is implemented in base/PatternStructure.py → class KleeneClosureOperator.

## Usage

Run the main script from the root directory:

```bash
python3 project/main.py --threads 32 --policy "match_next" --input "bike_events_10x.csv" 
```

