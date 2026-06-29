import subprocess
import re
import sys
import os
import time

CODES = ["7_1_3", "9_1_3", "15_7_3", "12_2_4", "16_6_4", "17_1_5", "19_1_5", "32_20_4", "24_4_5", "20_2_6", "23_1_7"]
CODES = ["12_2_4", "16_6_4", "17_1_5", "19_1_5"]
HEURISTICS = ["overlap", "zero_tolerance", "weighted_syndrome"]
FIRST_LAYERS = ["X"]
NUM_RUNS_PER_LAYER = 30

def run_test(code, heuristic, first_layer, seed):
    cmd = [
        "conda", "run", "-n", "zxlive", "python", "-m", "spiderstate.fault_tolerance_verification",
        "--code", code,
        "--heuristic", heuristic,
        "--first_layer", first_layer,
        "--seed", str(seed)
    ]
    
    start_time = time.time()
    TIMEOUT = 1200
    try:
        # Use a timeout of 120 seconds per run just in case
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1, -1, float(TIMEOUT)
    end_time = time.time()
    elapsed = end_time - start_time
    
    output = result.stdout + "\n" + result.stderr

    cx_match = re.search(r"#CX:\s*(\d+)", output)
    ms_match = re.search(r"#Meas:\s*(\d+)", output)

    if not cx_match or not ms_match:
        return "UNKNOWN", -1, -1, elapsed

    num_cx = int(cx_match.group(1))
    num_meas = int(ms_match.group(1))

    # Check if non-FT or Verification failed
    if "Final Verification Result: False" in output or "[FAIL]" in output:
        return "NON-FT", num_cx, num_meas, elapsed
    if "Final Verification Result: True" not in output:
        return "ERROR", num_cx, num_meas, elapsed

    return "FT", num_cx, num_meas, elapsed

import multiprocessing

def process_task(args):
    code, heuristic, first_layer, seed = args
    status, cnots, measurements, elapsed = run_test(code, heuristic, first_layer, seed)
    return args, status, cnots, measurements, elapsed

def main():
    print(f"{'Code':<10} | {'Heuristic':<20} | {'Layer':<5} | {'Run':<5} | {'Status':<10} | {'CNOTs':<5} | {'Meas':<5} | {'Time (s)':<8}")
    print("-" * 88)
    
    results = {}
    
    tasks = []
    for code in CODES:
        for heuristic in HEURISTICS:
            for first_layer in FIRST_LAYERS:
                for i in range(NUM_RUNS_PER_LAYER):
                    seed = 100 + i
                    tasks.append((code, heuristic, first_layer, seed))
                    
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        for (code, heuristic, first_layer, seed), status, cnots, measurements, elapsed in pool.imap(process_task, tasks):
            run_idx = seed - 100
            print(f"{code:<10} | {heuristic:<20} | {first_layer:<5} | {run_idx:<5} | {status:<10} | {cnots if cnots != -1 else '-':<5} | {measurements if measurements != -1 else '-':<5} | {elapsed:.2f}")
            
            key = (code, heuristic, first_layer)
            if key not in results:
                results[key] = []
            results[key].append((status, cnots, elapsed))
                    
    print("\n--- Summary (Min CNOTs per configuration) ---")
    for code in CODES:
        for heuristic in HEURISTICS:
            for first_layer in FIRST_LAYERS:
                key = (code, heuristic, first_layer)
                runs = results.get(key, [])
                if not runs:
                    continue
                ft_runs = [(cnots, elapsed) for stat, cnots, elapsed in runs if stat == "FT"]
                if ft_runs:
                    min_cnots = min(r[0] for r in ft_runs)
                    avg_time = sum(r[1] for r in runs) / len(runs)
                    print(f"{code:<10} | {heuristic:<20} | {first_layer:<5} -> Best FT CNOTs: {min_cnots:<5} (Avg Time: {avg_time:.2f}s)")
                else:
                    avg_time = sum(r[2] for r in runs) / len(runs)
                    print(f"{code:<10} | {heuristic:<20} | {first_layer:<5} -> No FT circuits found (Avg Time: {avg_time:.2f}s)")

if __name__ == "__main__":
    main()
