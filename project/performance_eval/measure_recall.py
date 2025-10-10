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
import statistics
import sys
import logging

# Define paths relative to this script's location
cur_path = pathlib.Path(__file__).parent.resolve()
project_root = cur_path.parent.resolve()
MAIN_PY = str(project_root / "main.py")
RESULT_CSV = cur_path / "results/recall_summary.csv"
DEFAULT_INPUT = project_root / "data/bike_events_1x.csv"
SAMPLE_INTERVAL = 0.2  # seconds

def run_main(cmd_args, timeout_s: float or None):
    """
    Run main.py with given arguments, enforce an optional timeout, and measure time and CPU usage.
    End the process after timeout_s if still alive.

    Args:
        cmd_args: List of command-line arguments to pass to main.py
        timeout_s: Timeout in seconds; if None, wait indefinitely

    """

    start = time.time()
    cpu_samples, mem_samples = [], []
    # Start the process
    proc = subprocess.Popen(["python3", MAIN_PY] + cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p = None
    samples = []
    if psutil:
        try:
            p = psutil.Process(proc.pid)
            p.cpu_percent(interval=None)  # Prime the measurement
        except psutil.NoSuchProcess:
            p = None  # Process ended before monitoring could start

    try:
        deadline = start + timeout_s if timeout_s is not None else None
        while proc.poll() is None:
            now = time.time()
            if deadline and now >= deadline:  # Timeout reached
                if p:
                    p.kill()
                else:
                    proc.kill()
                break

            if p:
                try:
                    # Record process CPU % and Resident Set Size (RSS) memory
                    samples.append((p.cpu_percent(interval=None), p.memory_info().rss))
                except psutil.NoSuchProcess:
                    break  # Process ended

            time.sleep(SAMPLE_INTERVAL)

        stdout, stderr = proc.communicate(timeout=5)

    except (subprocess.TimeoutExpired, Exception) as e:
        proc.kill()
        stdout, stderr = proc.communicate()
        if isinstance(e, subprocess.TimeoutExpired):
            stderr = (stderr or b"") + b"\nTimeoutExpired during communicate()"
        else:
            stderr = (stderr or b"") + f"\nException during monitoring: {e}".encode()

    elapsed = time.time() - start
    returncode = proc.returncode

    summary = {"cpu_avg_pct": 0.0, "cpu_peak_pct": 0.0, "ram_avg_bytes": 0, "ram_peak_bytes": 0}
    if samples:
        cpu_vals = [s[0] for s in samples if s[0] is not None]
        ram_vals = [s[1] for s in samples]
        if cpu_vals:
            summary["cpu_avg_pct"] = sum(cpu_vals) / len(cpu_vals)
            summary["cpu_peak_pct"] = max(cpu_vals)
        if ram_vals:
            summary["ram_avg_bytes"] = int(sum(ram_vals) / len(ram_vals))
            summary["ram_peak_bytes"] = max(ram_vals)

    return elapsed, returncode, (stderr or b"").decode(errors="ignore"), summary


def count_matches(output_path):
    """Count matches emitted by main.py textual output file (line count)."""
    if not output_path.exists():
        return 0
    with output_path.open("r") as f:
        return sum(1 for _ in f)

def main():
    # 1) Baseline run (no shedding, no timeout)
    baseline_out = cur_path / "results/baseline_matches.csv"

    baseline_args = ["--output", str(baseline_out), "--input", DEFAULT_INPUT, "--policy", "none"]
    T0, rc, stderr, _ = run_main(baseline_args, timeout_s=None)
    if rc != 0:
        logging.error(f"Baseline run failed with exit code {rc}. Stderr:\n{stderr}")
        sys.exit(1)

    baseline_matches = count_matches(baseline_out)
    logging.info(f"Baseline finished in {T0:.2f}s with {baseline_matches} matches.")
    if baseline_matches == 0:
        logging.warning("Baseline run produced 0 matches. Recall will be 0 or undefined.")

    # 2. Prepare CSV output file
    header = [
        "latency_pct_target", "latency_target_s", "elapsed_s", "matches", "recall",
        "cpu_avg_pct", "cpu_peak_pct", "ram_avg_bytes", "ram_peak_bytes", "return_code", "stderr_snippet"
    ]
    with RESULT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # 3. Latency-constrained runs
    latency_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    for percent in latency_percentages:
        target_latency = percent * T0
        print(f"Running with {int(percent * 100)}% latency target ({target_latency:.2f}s)...")

        out_file = cur_path / f"run_{int(percent * 100)}.out"
        # Assuming a policy that supports shedding is needed for these runs
        run_args = [
            "--output", str(out_file),
            "--input", DEFAULT_INPUT,
            "--policy", "none",
            "--latency-target", str(target_latency)  # Pass to main.py if it can use it
        ]

        Ta, rc, stderr, summary = run_main(run_args, timeout_s=target_latency)

        matches = count_matches(out_file)
        recall = (matches / baseline_matches) if baseline_matches > 0 else 0.0

        logging.info(f"  -> Finished in {Ta:.2f}s. Matches: {matches}, Recall: {recall:.3f}")

        # 4. Write results to CSV
        row = [
            int(percent * 100),
            f"{target_latency:.4f}",
            f"{Ta:.4f}",
            matches,
            f"{recall:.4f}",
            f"{summary['cpu_avg_pct']:.2f}",
            f"{summary['cpu_peak_pct']:.2f}",
            summary['ram_avg_bytes'],
            summary['ram_peak_bytes'],
            rc,
            stderr.replace("\n", " ").strip()[:200]
        ]
        with RESULT_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    logging.info(f"Evaluation complete. Results saved to '{RESULT_CSV}'")

if __name__ == "__main__":
    main()