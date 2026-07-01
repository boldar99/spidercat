import subprocess
import re
import sys
import os
import time
import tempfile
import uuid
import shutil
import stim
import multiprocessing
import matplotlib.pyplot as plt
from spiderstate.utils import count_operations, get_project_root

CODES = ["16_6_4"]
HEURISTICS = ["overlap"]
FIRST_LAYERS = ["X"]
NUM_RUNS_PER_LAYER = 1
MAX_COL_OPS_LIST = list(range(20))

def run_test(code, heuristic, first_layer, seed, max_col_ops):
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.stim")
    
    cmd = [
        "conda", "run", "-n", "zxlive", "python", "-m", "spiderstate.fault_tolerance_verification",
        "--code", code,
        "--heuristic", heuristic,
        "--num_circuits", "10",
        "--first_layer", first_layer,
        "--seed", str(seed),
        "--max_col_ops", str(max_col_ops),
        "--save_circuit", tmp_path
    ]
    
    start_time = time.time()
    TIMEOUT = 1200
    try:
        # Use a timeout of 1200 seconds per run just in case
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1, -1, float(TIMEOUT), None
    end_time = time.time()
    elapsed = end_time - start_time
    
    output = result.stdout + "\n" + result.stderr

    cx_match = re.search(r"#CX:\s*(\d+)", output)
    ms_match = re.search(r"#Meas:\s*(\d+)", output)

    if not cx_match or not ms_match:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return "UNKNOWN", -1, -1, elapsed, None

    num_cx = int(cx_match.group(1))
    num_meas = int(ms_match.group(1))

    # Check if non-FT or Verification failed
    if "Final Verification Result: False" in output or "[FAIL]" in output:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return "NON-FT", num_cx, num_meas, elapsed, None
    if "Final Verification Result: True" not in output:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return "ERROR", num_cx, num_meas, elapsed, None

    return "FT", num_cx, num_meas, elapsed, tmp_path


def process_task(args):
    code, heuristic, first_layer, seed, max_col_ops = args
    status, cnots, measurements, elapsed, tmp_path = run_test(code, heuristic, first_layer, seed, max_col_ops)
    return args, status, cnots, measurements, elapsed, tmp_path

def main():
    print(f"{'Code':<10} | {'Heur':<10} | {'Layer':<5} | {'Run':<3} | {'MaxOps':<6} | {'Status':<10} | {'CNOTs':<5} | {'Meas':<5} | {'Time (s)':<8}")
    print("-" * 88)
    
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    
    code = CODES[0]
    heuristic = HEURISTICS[0]
    first_layer = FIRST_LAYERS[0]
    
    tasks = []
    for max_col_ops in MAX_COL_OPS_LIST:
        for i in range(NUM_RUNS_PER_LAYER):
            seed = 100 + i
            tasks.append((code, heuristic, first_layer, seed, max_col_ops))
            
    results_by_col_ops = {}
            
    with multiprocessing.Pool(processes=num_cores) as pool:
        for (c, heur, f_layer, seed, max_col_ops), status, cnots, measurements, elapsed, tmp_path in pool.imap(process_task, tasks):
            run_idx = seed - 100
            print(f"{c:<10} | {heur[:10]:<10} | {f_layer:<5} | {run_idx:<3} | {max_col_ops:<6} | {status:<10} | {cnots if cnots != -1 else '-':<5} | {measurements if measurements != -1 else '-':<5} | {elapsed:.2f}")
            
            if max_col_ops not in results_by_col_ops:
                results_by_col_ops[max_col_ops] = []
            results_by_col_ops[max_col_ops].append((status, cnots, measurements, elapsed))
            
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    # Plotting
    ops_vals = []
    cnot_vals = []
    for max_col_ops in sorted(results_by_col_ops.keys()):
        runs = results_by_col_ops[max_col_ops]
        ft_runs = [cnots for stat, cnots, meas, elapsed in runs if stat == "FT"]
        if ft_runs:
            min_cnots = min(ft_runs)
            ops_vals.append(max_col_ops)
            cnot_vals.append(min_cnots)
            
    if ops_vals:
        plt.figure(figsize=(8, 6))
        plt.plot(ops_vals, cnot_vals, marker='o', linestyle='-')
        plt.xlabel('max_col_ops')
        plt.ylabel('Number of CNOT gates (Best FT)')
        plt.title(f'CNOT gates vs max_col_ops for {code}')
        plt.grid(True)
        plot_path = get_project_root().joinpath("figures", f"{code}_cnot_vs_max_col_ops.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f"\nPlot saved to {plot_path}")
    else:
        print("\nNo FT circuits found to plot.")

if __name__ == "__main__":
    main()
