import glob
import json
import os
import itertools

from spiderstate.utils import load_qecc, load_qecc_data

import math

BASELINE_DATA = {
    "7_1_3": {"cx": 15, "flags": 3, "sim_qubits": "8", "depth": "10", "ler": 2.8e-5, "ar": 0.97835},
    "9_1_3": {"cx": 26, "flags": 9, "sim_qubits": "12", "depth": "9", "ler": 2.5e-5, "ar": 0.97154},
    "17_1_5": {"cx": 74, "flags": 21, "sim_qubits": "23", "depth": "25", "ler": 1.295e-6, "ar": 0.89465},
    "25_1_5": {"cx": 92, "flags": 28, "sim_qubits": "32", "depth": "23", "ler": 1.545e-6, "ar": 0.8982},
    "49_1_5": {"cx": 361, "flags": 105, "sim_qubits": "95", "depth": "59", "ler": 4.45e-5, "ar": 0.5845},
    "20_2_6": {"cx": 145, "flags": 47, "sim_qubits": "36", "depth": "54", "ler": 6.0e-8, "ar": 0.82345},
    "23_1_7": {"cx": 237, "flags": 80, "sim_qubits": "44", "depth": "33", "ler": 2.45e-7, "ar": 0.7097},
    "31_1_7": {"cx": 211, "flags": 69, "sim_qubits": "55", "depth": "58", "ler": 3.75e-7, "ar": 0.7505},
    "49_1_7": {"cx": 262, "flags": 85, "sim_qubits": "64", "depth": "46", "ler": 2.8e-7, "ar": 0.7025},
    "95_1_7": {"cx": 1175, "flags": 380, "sim_qubits": "258", "depth": "389", "ler": 5.35e-5, "ar": 0.2405},
    "49_1_9": {"cx": 408, "flags": 136, "sim_qubits": "93", "depth": "123", "ler": 3.45e-7, "ar": 0.5315},
    "81_1_9": {"cx": 614, "flags": 206, "sim_qubits": "141", "depth": "129", "ler": 6.5e-7, "ar": 0.3555},
    "47_1_11": {"cx": 1033, "flags": 388, "sim_qubits": "186", "depth": "292", "ler": 1.03e-6, "ar": 0.1225},
    "71_1_11": {"cx": 829, "flags": 268, "sim_qubits": "177", "depth": "282", "ler": 1.67e-7, "ar": 0.2145},
}

def get_state(code, k):
    if code in ("49_1_5", "95_1_7"):
        return r"$\ket{\overline{+}}$"
    zeros = "0" * k
    return f"$\\ket{{\\overline{{{zeros}}}}}$"

def format_float(val, digits=1):
    return f"{val + 1e-9:.{digits}f}"

def escape_percentage(p):
    return f"{p * 100 + 1e-9:.1f}\\%"

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
    
    print("\\begin{table*}[ht]")
    print("\\centering")
    print("\\scriptsize")
    print("\\begin{tabular}{l l c c c c c c c}")
    print("\\toprule")
    print("\\makecell[l]{QEC Code \\\\ \\& State} & Method & \\makecell{CX \\\\ Count} & \\makecell{Flag \\\\ Count} & \\makecell{Qubit Reuse \\\\ Opt.\\@ Target} & \\makecell{Sim.\\@ \\\\ Qubits} & Depth & LER & AR \\\\")
    print("\\midrule")
    
    for code_raw, group in grouped_data:
        num_strategy_rows = len(group)
        num_rows = num_strategy_rows + 1  # 1 for Flag at Origin + N for CSSCat
        
        first_row = group[0]
        n, k, d = first_row["n"], first_row["k"], first_row["d"]
        label = first_row["label"]
        code_tex = f"$\\code{{{n}, {k}, {d}}}$"
        state = get_state(code_raw, k)
        code_state = f"{code_tex} {label} {state}" if label else f"{code_tex} {state}"
        
        cxs = first_row.get("num_cx", 0)
        flags = first_row.get("num_flags", 0)
        
        multirow_code = f"\\multirow{{{num_rows}}}{{*}}{{\\makecell[l]{{{code_state}}}}}"
        
        base_cx = BASELINE_DATA.get(code_raw, {}).get("cx", "-")
        base_flags = BASELINE_DATA.get(code_raw, {}).get("flags", "-")
        base_sim = BASELINE_DATA.get(code_raw, {}).get("sim_qubits", "-")
        base_depth = BASELINE_DATA.get(code_raw, {}).get("depth", "-")
        
        our_first_ler = next((r.get("logical_error_rate") for r in group if r.get("logical_error_rate") and r.get("logical_error_rate") > 0), None)
        shared_exp = math.floor(math.log10(our_first_ler)) if our_first_ler is not None else 0
            
        base_ler_val = BASELINE_DATA.get(code_raw, {}).get("ler", None)
        if base_ler_val is not None:
            exp_to_use = shared_exp if shared_exp != 0 else (math.floor(math.log10(base_ler_val)) if base_ler_val > 0 else 0)
            if exp_to_use != 0:
                b_val = base_ler_val / (10**exp_to_use)
                base_ler = f"${format_float(b_val, 1)}\\! \\times\\! 10^{{{exp_to_use}}}$"
            else:
                base_ler = f"${base_ler_val}$"
        else:
            base_ler = "-"
            
        base_ar_val = BASELINE_DATA.get(code_raw, {}).get("ar", None)
        if base_ar_val is not None:
            base_ar = escape_percentage(base_ar_val)
        else:
            base_ar = "-"
        
        # Print the Flag at Origin row
        print(f"{multirow_code} & Flag at Origin & {base_cx} & {base_flags} & - & {base_sim} & {base_depth} & {base_ler} & {base_ar} \\\\")
        
        multirow_method = f"\\multirow{{{num_strategy_rows}}}{{*}}{{CSSCat}}"
        multirow_cx = f"\\multirow{{{num_strategy_rows}}}{{*}}{{{cxs}}}"
        multirow_flags = f"\\multirow{{{num_strategy_rows}}}{{*}}{{{flags}}}"
        
        for i, row in enumerate(group):
            strategy = row.get("strategy", "Unknown").replace("Strategy", "")
            if strategy in ("AggressiveDepthAware", "PureAggressive"):
                strategy = r"Sim.\@ Qubit"
            elif strategy == "DepthPreserving":
                strategy = r"Depth"
            elif strategy == "VolumeOptimizingReuse":
                strategy = r"Volume"
                
            sim_qubits = row.get("num_sim_qubits", 0)
            depth = row.get("depth", 0)
            
            ler = row.get("logical_error_rate", None)
            if ler is None or ler <= 0:
                ler_latex = "-"
            else:
                exp_to_use = shared_exp if shared_exp != 0 else math.floor(math.log10(ler))
                if exp_to_use != 0:
                    our_val = ler / (10**exp_to_use)
                    ler_latex = f"${format_float(our_val, 1)}\\! \\times\\! 10^{{{exp_to_use}}}$"
                else:
                    ler_latex = f"${ler}$"
                
            ar = row.get("acceptance_rate", None)
            ar_latex = escape_percentage(ar) if ar is not None else "-"
            
            method_col = multirow_method if i == 0 else ""
            cx_col = multirow_cx if i == 0 else ""
            flag_col = multirow_flags if i == 0 else ""
            
            print(f" & {method_col} & {cx_col} & {flag_col} & {strategy} & {sim_qubits} & {depth} & {ler_latex} & {ar_latex} \\\\")
            
        # Optional line between different codes for clean grouping
        print("\\midrule")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{")
    print("\tResource overhead, logical error rate, and acceptance rate for different CSS QECCs.")
    print("\tColumns from left to right: QEC code and state, Method (CSSCat or Flag at Origin), number of CNOT gates in the circuit, number of flag measurements, qubit reuse policy used, maximum simultaneous number of qubits necessary, circuit depth, logical error rate, and acceptance rate.")
    print("}")
    print("\\label{tab:sim_results}")
    print("\\end{table*}")

if __name__ == "__main__":
    main()
