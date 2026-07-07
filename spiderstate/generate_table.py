import glob
import json
import os
import itertools

from spiderstate.utils import load_qecc, load_qecc_data

def get_state(code):
    if code in ("49_1_5", "95_1_7"):
        return r"$\ket{\overline{+}}$"
    return r"$\ket{\overline{0}}$"

def main():
    results_dir = "simulation_results"
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    data = []
    for f in json_files:
        with open(f, 'r') as file:
            try:
                stats = json.load(file)
                code_raw = stats.get("code", "")
                code_data = load_qecc_data(code_raw, "FAO")
                stats["n"] = code_data["n"]
                stats["k"] = code_data["k"]
                stats["d"] = code_data["d"]
                stats["label"] = code_data.get("abbr_name", "")
                data.append(stats)
            except Exception:
                continue
                
    # Sort by d then n
    data.sort(key=lambda x: (x["d"], x["n"], x.get("code", ""), x.get("strategy", "")))
    
    grouped_data = []
    for code, group in itertools.groupby(data, key=lambda x: x.get("code", "")):
        grouped_data.append((code, list(group)))
    
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\small")
    # Using 9 columns since CXs is its own column
    print("\\begin{tabular}{l c c c c c c c c}")
    print("\\toprule")
    print("\\makecell[l]{QEC Code \\\\ \\& State} & \\makecell{CX \\\\ Count} & \\makecell{Flag \\\\ Count} & \\makecell{Reuse \\\\ Strategy} & \\makecell{Sim. \\\\ Qubits} & Depth & \\makecell{Expected \\\\ Circ. Volume} & \\makecell{Log. Error \\\\ Rate} & \\makecell{Acceptance \\\\ Rate} \\\\")
    print("\\midrule")
    
    for code_raw, group in grouped_data:
        num_rows = len(group)
        first_row = group[0]
        n, k, d = first_row["n"], first_row["k"], first_row["d"]
        label = first_row["label"]
        code_tex = f"$\\code{{{n}, {k}, {d}}}$"
        state = get_state(code_raw)
        # Add space only if label exists
        code_state = f"{code_tex} {label} {state}" if label else f"{code_tex} {state}"
        cxs = first_row.get("num_cx", 0)
        flags = first_row.get("num_flags", 0)
        
        multirow_code = f"\\multirow{{{num_rows}}}{{*}}{{\\makecell[l]{{{code_state}}}}}"
        multirow_cx = f"\\multirow{{{num_rows}}}{{*}}{{{cxs}}}"
        multirow_flags = f"\\multirow{{{num_rows}}}{{*}}{{{flags}}}"
        
        for i, row in enumerate(group):
            strategy = row.get("strategy", "Unknown").replace("Strategy", "")
            if strategy in ("AggressiveDepthAware", "PureAggressive"):
                strategy = "Aggressive"
            elif strategy == "DepthPreserving":
                strategy = r"Naive"
            elif strategy == "VolumeOptimizingReuse":
                strategy = "Volume Optimizing"
                
            sim_qubits = row.get("num_sim_qubits", 0)
            depth = row.get("depth", 0)
            
            ler = row.get("logical_error_rate", 0.0)
            ler_str = f"{ler:.2e}".split("e")
            if len(ler_str) == 2:
                base = ler_str[0]
                exp = int(ler_str[1])
                ler_latex = f"${base} \\times 10^{{{exp}}}$"
            else:
                ler_latex = f"${ler}$"
                
            ar = row.get("acceptance_rate", 0.0)
            vol = row.get("expected_circuit_volume", 0)
            
            first_col = multirow_code if i == 0 else ""
            second_col = multirow_cx if i == 0 else ""
            third_col = multirow_flags if i == 0 else ""
            
            print(f"{first_col} & {second_col} & {third_col} & {strategy} & {sim_qubits} & {depth} & {vol} & {ler_latex} & {ar:.4f} \\\\")
            
        # Optional line between different codes for clean grouping
        print("\\midrule")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Simulation results across various QEC codes and qubit reuse strategies.}")
    print("\\label{tab:sim_results}")
    print("\\end{table*}")

if __name__ == "__main__":
    main()
