import glob
import json
import os
import itertools

from spiderstate.utils import load_qecc, load_qecc_data

import math

BASELINE_DATA = {
    "7_1_3": {"cx": 15, "flags": 3, "sim_qubits": "8", "depth": "10", "ler_bounds": (2.7, 2.9, -5), "ar_bounds": (0.9783, 0.9784)},
    "9_1_3": {"cx": 26, "flags": 9, "sim_qubits": "12", "depth": "9", "ler_bounds": (2.4, 2.6, -5), "ar_bounds": (0.9715, 0.9716)},
    "17_1_5": {"cx": 74, "flags": 21, "sim_qubits": "23", "depth": "25", "ler_bounds": (7.7, 18.2, -7), "ar_bounds": (0.8945, 0.8948)},
    "25_1_5": {"cx": 92, "flags": 28, "sim_qubits": "32", "depth": "23", "ler_bounds": (6.7, 24.2, -7), "ar_bounds": (0.8980, 0.8984)},
    "49_1_5": {"cx": 361, "flags": 105, "sim_qubits": "95", "depth": "59", "ler_bounds": (4.2, 4.7, -5), "ar_bounds": (0.5840, 0.5850)},
    "20_2_6": {"cx": 145, "flags": 47, "sim_qubits": "36", "depth": "54", "ler_bounds": (2.3, 9.7, -8), "ar_bounds": (0.8234, 0.8235)},
    "23_1_7": {"cx": 237, "flags": 80, "sim_qubits": "44", "depth": "33", "ler_bounds": (1.8, 3.1, -7), "ar_bounds": (0.7095, 0.7099)},
    "31_1_7": {"cx": 211, "flags": 69, "sim_qubits": "55", "depth": "58", "ler_bounds": (2.1, 5.4, -7), "ar_bounds": (0.7500, 0.7510)},
    "49_1_7": {"cx": 262, "flags": 85, "sim_qubits": "64", "depth": "46", "ler_bounds": (1.2, 4.4, -7), "ar_bounds": (0.7020, 0.7030)},
    "95_1_7": {"cx": 1175, "flags": 380, "sim_qubits": "258", "depth": "389", "ler_bounds": (4.4, 6.3, -5), "ar_bounds": (0.2400, 0.2410)},
    "49_1_9": {"cx": 408, "flags": 136, "sim_qubits": "93", "depth": "123", "ler_bounds": (1.1, 5.8, -7), "ar_bounds": (0.5310, 0.5320)},
    "81_1_9": {"cx": 614, "flags": 206, "sim_qubits": "141", "depth": "129", "ler_bounds": (2.0, 11.0, -7), "ar_bounds": (0.3550, 0.3560)},
    "47_1_11": {"cx": 1033, "flags": 388, "sim_qubits": "186", "depth": "292", "ler_bounds": (3.6, 17.0, -7), "ar_bounds": (0.1220, 0.1230)},
    "71_1_11": {"cx": 829, "flags": 268, "sim_qubits": "177", "depth": "282", "ler_bounds": (4.4, 29.0, -8), "ar_bounds": (0.2140, 0.2150)},
}

def get_state(code, k):
    if code in ("49_1_5", "95_1_7"):
        return r"$\ket{\overline{+}}$"
    if k > 3:
        return f"$\\ket{{\\overline{0}}}^{{\\otimes {k}}}$"
    zeros = "0" * k
    return f"$\\ket{{\\overline{{{zeros}}}}}$"

def format_float(val, digits=1):
    return f"{val + 1e-9:.{digits}f}"

def escape_percentage(p):
    return f"{p * 100 + 1e-9:.1f}\\%"

def wilson_score_interval(p, n, z=1.95996):
    if n <= 0:
        return p, p
    denominator = 1 + z**2/n
    center = p + z**2 / (2*n)
    spread = z * math.sqrt(p*(1-p)/n + z**2 / (4*n**2))
    return (center - spread) / denominator, (center + spread) / denominator

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
    data.sort(key=lambda x: (x["d"], x["n"], x.get("code", ""), len(x.get("strategy", ""))))
    
    grouped_data = []
    for code, group in itertools.groupby(data, key=lambda x: x.get("code", "")):
        grouped_data.append((code, list(group)))
    
    print("\\begin{table*}[ht]")
    print("\\centering")
    print("\\scriptsize")
    print("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l l c c c c c c c}")
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
        
        base_ler_bounds = BASELINE_DATA.get(code_raw, {}).get("ler_bounds", None)
        base_ar_bounds = BASELINE_DATA.get(code_raw, {}).get("ar_bounds", None)
        
        all_lers = [r.get("logical_error_rate") for r in group if r.get("logical_error_rate") and r.get("logical_error_rate") > 0]
        if base_ler_bounds:
            all_lers.append(base_ler_bounds[0] * 10**base_ler_bounds[2])
            
        if all_lers:
            min_ler = min(all_lers)
            shared_exp = math.floor(math.log10(min_ler))
        else:
            shared_exp = 0
            
        if base_ler_bounds:
            low, high, orig_exp = base_ler_bounds
            factor = 10**(orig_exp - shared_exp)
            adj_low = low * factor
            adj_high = high * factor
            base_ler = f"$[{format_float(adj_low, 1)}, \\,\\, {format_float(adj_high, 1)}]\\! \\times\\! 10^{{{shared_exp}}}$"
        else:
            base_ler = "-"
            
        if base_ar_bounds:
            base_ar = f"$[{base_ar_bounds[0]:.4f}, \\,\\, {base_ar_bounds[1]:.4f}]$"
        else:
            base_ar = "-"
        
        # Print the Flag at Origin row
        print(f"{multirow_code} & Flag at Origin & {base_cx} & {base_flags} &   & {base_sim} & {base_depth} & {base_ler} & {base_ar} \\\\")
        
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
            
            n_samples = row.get("num_samples", 0)
            ler = row.get("logical_error_rate", None)
            if ler is None or ler <= 0:
                ler_latex = "-"
            else:
                ar_temp = row.get("acceptance_rate", 1.0)
                n_ler = n_samples * ar_temp
                low, high = wilson_score_interval(ler, n_ler)
                
                # Determine shared exponent if we didn't calculate one globally
                exp_to_use = shared_exp if shared_exp != 0 else math.floor(math.log10(ler))
                
                if exp_to_use != 0:
                    low_val = low / (10**exp_to_use)
                    high_val = high / (10**exp_to_use)
                    ler_latex = f"$[{format_float(low_val, 1)}, \\,\\, {format_float(high_val, 1)}]\\! \\times\\! 10^{{{exp_to_use}}}$"
                else:
                    ler_latex = f"$[{format_float(low, 5)}, \\,\\, {format_float(high, 5)}]$"
                
            ar = row.get("acceptance_rate", None)
            if ar is None:
                ar_latex = "-"
            else:
                ar_low, ar_high = wilson_score_interval(ar, n_samples)
                ar_latex = f"$[{ar_low:.4f}, \\,\\, {ar_high:.4f}]$"
            
            method_col = multirow_method if i == 0 else ""
            cx_col = multirow_cx if i == 0 else ""
            flag_col = multirow_flags if i == 0 else ""
            
            print(f" & {method_col} & {cx_col} & {flag_col} & {strategy} & {sim_qubits} & {depth} & {ler_latex} & {ar_latex} \\\\")
            
        # Optional line between different codes for clean grouping
        print("\\midrule")
        
    print("\\bottomrule")
    print("\\end{tabular*}")
    print("\\caption{")
    print("\tResource overhead, logical error rate, and acceptance rate for different CSS QECCs.")
    print("\tColumns from left to right: QEC code and state, Method (CSSCat or Flag at Origin~\\cite{forlivesi2025flag}), number of CNOT gates in the circuit, number of flag measurements, optimization target of qubit reuse strategy, maximum simultaneous number of qubits necessary, circuit depth, and finally logical error rate and acceptance rates using Wilson confidence intervals of 95\\%.")
    print("\tThe logical error rates of some codes were not estimated (marked $-$) as the lookup table was too large to store in memory.")
    print("}")
    print("\\label{tab:sim_results}")
    print("\\end{table*}")

if __name__ == "__main__":
    main()
