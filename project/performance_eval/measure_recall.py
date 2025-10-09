# File: `scripts/evaluator.py`
"""
Automates:
 - baseline run (no shedding) to collect matches_baseline.csv
 - compute latency thresholds (10,30,50,70,90 percentiles)
 - compute recall under each threshold
 - write summary CSV recall_summary.csv
"""

import subprocess
import time
import pathlib
import psutil
import csv
import os
import sys
import logging

cur_path = pathlib.Path(__file__).parent.resolve()
MAIN_PY = "project/main.py"
DEFAULT_INPUT = "data/bike_events_1x.csv"
DEFAULT_OUTPUT = "performance_eval/recall_summary.csv"

def run_main(cmd_args) -> tuple[float, int, str]:
    """Run main.py with given arguments and measure time and CPU usage."""
    # Launch the process
    start = time.time()
    proc = subprocess.Popen(["python3", MAIN_PY] + cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Monitor CPU usage while running
    while proc.poll() is None:
        cpu_percent = psutil.cpu_percent(interval=1)
        # Record or log cpu usage if desired
        mem = psutil.virtual_memory()
        print("CPU usage:", cpu_percent, "% RAM usage:", mem.percent, "%")
    proc.wait()  # ensure process has terminated
    stdout, stderr = proc.communicate()
    end = time.time()
    elapsed = end - start
    return elapsed, proc.returncode, (stderr or b"").decode(errors="replace")

if __name__ == "__main__":

    # 1) Baseline run (no shedding)
    baseline_args = ["--threads", "8", "--input", "bike_events_2x.csv", "--output", "matches_baseline.out"]
    T0, rc, stderr = run_main(baseline_args)
    if rc != 0:
        logging.error("Baseline run failed (rc=%s). stderr:\n%s", rc, stderr)
        sys.exit(1)

    baseline_path = "matches_baseline.out"
    if not os.path.exists(baseline_path):
        logging.error("Expected output file `%s` not found after baseline run.", baseline_path)
        sys.exit(1)

    with open(baseline_path) as f:
        baseline_matches = sum(1 for _ in f)  # Count baseline matches for recall calculation

    if baseline_matches == 0:
        logging.warning("Baseline produced zero matches; subsequent recall computations will be zero or undefined.")

    # 2) Latency-targeted runs (no shedding)
    latency_bounds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for l in latency_bounds:
        target = l * T0
        out_file = f"run_{int(l * 100)}.out"
        args = ["--threads", "8",
                "--input", "bike_events_2x.csv",
                "--latency_target", str(target),
                "--output", out_file]
        Ta, rc, stderr = run_main(args)
        if rc != 0:
            logging.error("Run for latency %s failed (rc=%s). stderr:\n%s", l, rc, stderr)
            continue
        if not os.path.exists(out_file):
            logging.error("Expected output file `%s` not found after run (latency %s).", out_file, l)
            continue
        with open(out_file) as f:
            matches = sum(1 for _ in f)
        recall = (matches / baseline_matches) if baseline_matches else 0.0
        print(f"Latency {int(l * 100)}% target: time={Ta:.2f}s, matches={matches}, recall={recall:.2f}")
