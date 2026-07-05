import subprocess
import re
import sys
import os
import time
import tempfile
import uuid
import shutil
import stim
from spiderstate.utils import count_operations, get_project_root

CODES = ["7_1_3", "9_1_3", "15_1_3", "15_7_3", "12_2_4", "16_6_4", "17_1_5", "19_1_5", "25_1_5", "24_4_5", "20_2_6", "23_1_7"]
# CODES = ["23_1_7"]
CODES = ["37_1_7"]
HEURISTICS = ["overlap", "zero_tolerance", "weighted_syndrome"]
# HEURISTICS = ["overlap"]
FIRST_LAYERS = ["X"]
NUM_RUNS_PER_LAYER = 1
NUM_CIRCUITS = 1

def get_best_count(code):
    path = get_project_root().joinpath("good_circuits", f"{code}.stim")
    if not os.path.exists(path):
        return float('inf')
    try:
        circ = stim.Circuit.from_file(str(path))
        cx, meas = count_operations(circ)
        return cx + meas
    except Exception:
        return float('inf')

def run_test(code, heuristic, first_layer, seed):
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.stim")
    
    cmd = [
        "conda", "run", "-n", "zxlive", "python", "-m", "spiderstate.fault_tolerance_verification",
        "--code", code,
        "--heuristic", heuristic,
        "--num_circuits", str(NUM_CIRCUITS),
        "--first_layer", first_layer,
        "--seed", str(seed),
        "--save_circuit", tmp_path
    ]
    
    start_time = time.time()
    TIMEOUT = 25000
    try:
        # Use a timeout of 120 seconds per run just in case
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT * NUM_RUNS_PER_LAYER)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1, -1, float(TIMEOUT * NUM_CIRCUITS), None
    end_time = time.time()
    elapsed = end_time - start_time
    
    output = result.stdout + "\n" + result.stderr

    cx_match = re.search(r"#CX:\s*(\d+)", output)
    ms_match = re.search(r"#Meas:\s*(\d+)", output)

    if not cx_match or not ms_match:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        print(output, file=sys.stderr)
        return "UNKNOWN", -1, -1, elapsed, None

    num_cx = int(cx_match.group(1))
    num_meas = int(ms_match.group(1))

    # Check if non-FT or Verification failed
    if "Final Verification Result: False" in output or "[FAIL]" in output:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return "NON-FT", num_cx, num_meas, elapsed, None
    if "Final Verification Result: True" not in output:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        print(output, file=sys.stderr)
        return "ERROR", num_cx, num_meas, elapsed, None

    return "FT", num_cx, num_meas, elapsed, tmp_path

import multiprocessing

def process_task(args):
    code, heuristic, first_layer, seed = args
    status, cnots, measurements, elapsed, tmp_path = run_test(code, heuristic, first_layer, seed)
    return args, status, cnots, measurements, elapsed, tmp_path

def main():
    print(f"{'Code':<10} | {'Heuristic':<20} | {'Layer':<5} | {'Run':<5} | {'Status':<10} | {'CNOTs':<5} | {'Meas':<5} | {'Time (s)':<8}")
    print("-" * 88)
    
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    
    for code in CODES:
        results = {}
        best_path = None
        best_cost = float('inf')
        
        tasks = []
        for heuristic in HEURISTICS:
            for first_layer in FIRST_LAYERS:
                for i in range(NUM_RUNS_PER_LAYER):
                    seed = 100 + i
                    tasks.append((code, heuristic, first_layer, seed))
                    
        with multiprocessing.Pool(processes=num_cores) as pool:
            for (c, heuristic, first_layer, seed), status, cnots, measurements, elapsed, tmp_path in pool.imap(process_task, tasks):
                run_idx = seed - 100
                print(f"{c:<10} | {heuristic:<20} | {first_layer:<5} | {run_idx:<5} | {status:<10} | {cnots if cnots != -1 else '-':<5} | {measurements if measurements != -1 else '-':<5} | {elapsed:.2f}")
                
                key = (heuristic, first_layer)
                if key not in results:
                    results[key] = []
                results[key].append((status, cnots, measurements, elapsed))
                
                if status == "FT" and tmp_path and os.path.exists(tmp_path):
                    total_cost = cnots + measurements
                    if total_cost < best_cost:
                        if best_path and os.path.exists(best_path):
                            os.remove(best_path)
                        best_cost = total_cost
                        best_path = tmp_path
                    else:
                        os.remove(tmp_path)
                        
        print(f"\n--- Summary for {code} ---")
        for heuristic in HEURISTICS:
            for first_layer in FIRST_LAYERS:
                key = (heuristic, first_layer)
                runs = results.get(key, [])
                if not runs:
                    continue
                ft_runs = [(cnots, elapsed) for stat, cnots, meas, elapsed in runs if stat == "FT"]
                if ft_runs:
                    min_cnots = min(r[0] for r in ft_runs)
                    avg_time = sum(r[1] for r in runs) / len(runs)
                    print(f"{code:<10} | {heuristic:<20} | {first_layer:<5} -> Best FT CNOTs: {min_cnots:<5} (Avg Time: {avg_time:.2f}s)")
                else:
                    avg_time = sum(r[3] for r in runs) / len(runs)
                    print(f"{code:<10} | {heuristic:<20} | {first_layer:<5} -> No FT circuits found (Avg Time: {avg_time:.2f}s)")
                    
        if best_path:
            existing_cost = get_best_count(code)
            if best_cost < existing_cost:
                dest_dir = get_project_root().joinpath("good_circuits")
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir.joinpath(f"{code}.stim")
                shutil.move(best_path, str(dest_path))
                if existing_cost == float('inf'):
                    print(f"[{code}] Saved new FT circuit! (Total cost: {best_cost})\n")
                else:
                    print(f"[{code}] Improved FT circuit! Cost reduced from {existing_cost} to {best_cost}. Overwrote existing file.\n")
            else:
                print(f"[{code}] Could not beat existing best. (Best found: {best_cost}, Existing: {existing_cost})\n")
                if os.path.exists(best_path):
                    os.remove(best_path)
        else:
            print(f"[{code}] No valid FT circuits generated to save.\n")

if __name__ == "__main__":
    main()
