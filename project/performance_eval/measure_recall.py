# File: project/performance_eval/measure_recall.py
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

# Ensure the project root is in sys.path for imports
nb_dir = pathlib.Path(__file__).parent if "__file__" in globals() else pathlib.Path.cwd()
project_root = str((nb_dir / "../..").resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define paths relative to this script's location
cur_path = pathlib.Path(__file__).parent.resolve()
project_root = cur_path.parent.resolve()

print(f"Project root: {project_root}")
MAIN_PY = str(project_root / "main.py")
RESULT_CSV = cur_path / "results/recall_summary.csv"
DEFAULT_INPUT = project_root / "data/bike_events_synth_250.csv"
SAMPLE_INTERVAL = 0.01  # seconds

from project.main import (
    cep,
    build_events_stream,
    FileOutputStream,
    BikeTripDataFormatter,
    BikeTripEventTypeClassifier,
    build_cep,
)

import signal


# Define timeout handler
def handler(signum, frame):
    raise TimeoutError("Function execution timed out!")


# def run_with_timeout(func, timeout, *args, **kwargs):
#     # Register signal handler
#     signal.signal(signal.SIGALRM, handler)
#     signal.alarm(timeout)  # Set alarm
#     try:
#         result = func(*args, **kwargs)
#     finally:
#         signal.alarm(0)  # Cancel alarm
#     return result
from multiprocessing import Process, Queue
import time


def long_task(q):
    args = q.get()
    print(f"Starting long_task with args: {args}")
    input_stream = build_events_stream(args[0])
    output_stream = FileOutputStream("./", args[1], is_async=True, _print=False, q=q)
    fmt = BikeTripDataFormatter(BikeTripEventTypeClassifier())
    cep = build_cep()
    cep.run(input_stream, output_stream, fmt)
    q.put("done")


def run_with_timeout(args, timeout, q):
    p = Process(target=long_task, args=(q,))
    p.start()
    q.put(args)
    print(f"Process started with PID {p.pid}, waiting for {timeout} seconds")
    p.join(timeout)
    print(f"Process join returned. Alive: {p.is_alive()}")
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError("Function execution timed out!")
    return q.get()


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

    input_path = project_root / "data/bike_events_synth_250.csv"
    input_stream = build_events_stream(input_path)
    output_path = cmd_args[cmd_args.index("--output") + 1] if "--output" in cmd_args else "output.csv"

    print(f"Running CEP with args: {cmd_args}, timeout: {timeout_s}s")
    start = time.time()
    results = []
    try:
        # if timeout_s is not None:
        print(f"Running with timeout of {timeout_s} seconds")
        q = Queue()
        run_with_timeout((input_path, output_path), timeout_s, q)
        # else:
            # q = Queue()
            # output_stream = FileOutputStream("./", output_path, is_async=True, q=q)
            # fmt = BikeTripDataFormatter(BikeTripEventTypeClassifier())
            # cep.run(input_stream, output_stream, fmt)
        returncode = 0
        stderr = b""
        print("CEP run completed successfully")
    except TimeoutError:
        returncode = -1
        stderr = b"TimeoutExpired during cep.run()"
    except Exception as e:
        returncode = -2
        stderr = f"Exception during cep.run(): {e}".encode()
        raise
    while not q.empty():
        results.append(q.get())
    # output_stream.close()

    print(f"CEP run finished, returncode: {returncode}, results: {results}")

    elapsed = time.time() - start
    # returncode = proc.returncode

    summary = {
        "cpu_avg_pct": 0.0,
        "cpu_peak_pct": 0.0,
        "ram_avg_bytes": 0,
        "ram_peak_bytes": 0,
        "num_matches": len(results),
    }
    # if samples:
    #     cpu_vals = [s[0] for s in samples if s[0] is not None]
    #     ram_vals = [s[1] for s in samples]
    #     if cpu_vals:
    #         summary["cpu_avg_pct"] = sum(cpu_vals) / len(cpu_vals)
    #         summary["cpu_peak_pct"] = max(cpu_vals)
    #     if ram_vals:
    #         summary["ram_avg_bytes"] = int(sum(ram_vals) / len(ram_vals))
    #         summary["ram_peak_bytes"] = max(ram_vals)

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

    baseline_args = ["--output", str(baseline_out), "--input", str(DEFAULT_INPUT), "--policy", "none"]
    T0, rc, stderr, _ = run_main(baseline_args, timeout_s=None)
    if rc != 0:
        logging.error(f"Baseline run failed with exit code {rc}. Stderr:\n{stderr}")
        sys.exit(1)

    baseline_matches = count_matches(baseline_out)
    T0 += 5
    logging.info(f"Baseline finished in {T0:.2f}s with {baseline_matches} matches.")
    if baseline_matches == 0:
        logging.warning("Baseline run produced 0 matches. Recall will be 0 or undefined.")

    # 2. Prepare CSV output file
    header = [
        "latency_pct_target",
        "latency_target_s",
        "elapsed_s",
        "matches",
        "recall",
        "cpu_avg_pct",
        "cpu_peak_pct",
        "ram_avg_bytes",
        "ram_peak_bytes",
        "return_code",
        "stderr_snippet",
    ]
    with RESULT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # 3. Latency-constrained runs
    latency_percentages = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99]
    for percent in latency_percentages:
        target_latency = percent * T0
        print(f"Running with {int(percent * 100)}% latency target ({target_latency:.2f}s)...")

        out_file = cur_path / f"run_{int(percent * 100)}.out"
        # The timeout is enforced by run_main, not by an argument to main.py
        run_args = [
            "--output",
            str(out_file),
            "--input",
            str(DEFAULT_INPUT),
            "--policy",
            "none",
        ]

        Ta, rc, stderr, summary = run_main(run_args, timeout_s=target_latency)

        matches = summary["num_matches"]
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
            summary["ram_avg_bytes"],
            summary["ram_peak_bytes"],
            rc,
            stderr.replace("\n", " ").strip()[:200],
        ]
        with RESULT_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    logging.info(f"Evaluation complete. Results saved to '{RESULT_CSV}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
