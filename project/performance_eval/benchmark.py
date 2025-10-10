import pathlib
import psutil
import subprocess
import sys

cur_path = pathlib.Path(__file__).parent.resolve()
data_path = cur_path / "data"

for policy in ["none","match_single","match_next"]:
    for scale in [1,2,5,10]:
        scaled_input = f"bike_events_{scale}x.csv"
        out_text = f"out_{policy}_{scale}x.txt"
        out_matches = f"matches_{policy}_{scale}x.csv"
        subprocess.run(["python", "scripts/main.py", f"--policy={policy}", "--threads=16",
                        f"--input={scaled_input}", f"--output={out_text}"])
