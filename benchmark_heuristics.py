import subprocess
import re
import sys
import os
import time

CODES = ["7_1_3", "9_1_3", "15_7_3", "12_2_4", "16_6_4", "17_1_5", "19_1_5"]
HEURISTICS = ["overlap", "zero_tolerance", "weighted_syndrome"]
FIRST_LAYERS = ["Z", "X"]
NUM_RUNS_PER_LAYER = 4

def run_test(code, heuristic, first_layer, seed):
    cmd = [
        "conda", "run", "-n", "zxlive", "python", "-m", "spiderstate.fault_tolerance_verification",
        "--code", code,
        "--heuristic", heuristic,
        "--first_layer", first_layer,
        "--seed", str(seed)
    ]
    
    start_time = time.time()
    try:
        # Use a timeout of 120 seconds per run just in case
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1, 120.0
    end_time = time.time()
    elapsed = end_time - start_time
    
    output = result.stdout + "\n" + result.stderr
    
    # Check if non-FT or Verification failed
    if "Final Verification Result: False" in output or "[FAIL]" in output:
        return "NON-FT", -1, elapsed
    if "Final Verification Result: True" not in output:
        return "ERROR", -1, elapsed
        
    # Extract CNOT count
    match = re.search(r"#CX:\s*(\d+)", output)
    if match:
        return "FT", int(match.group(1)), elapsed
    
    return "UNKNOWN", -1, elapsed

import multiprocessing

def process_task(args):
    code, heuristic, first_layer, seed = args
    status, cnots, elapsed = run_test(code, heuristic, first_layer, seed)
    return args, status, cnots, elapsed

def main():
    print(f"{'Code':<10} | {'Heuristic':<20} | {'Layer':<5} | {'Run':<5} | {'Status':<10} | {'CNOTs':<5} | {'Time (s)':<8}")
    print("-" * 75)
    
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
        for (code, heuristic, first_layer, seed), status, cnots, elapsed in pool.imap(process_task, tasks):
            run_idx = seed - 100
            print(f"{code:<10} | {heuristic:<20} | {first_layer:<5} | {run_idx:<5} | {status:<10} | {cnots if cnots != -1 else '-':<5} | {elapsed:.2f}")
            
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
