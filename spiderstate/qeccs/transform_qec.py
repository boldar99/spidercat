import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Transform a QEC code JSON to project format.")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        data = json.load(f)
        
    out_data = {}
    
    out_data["full_name"] = data.get("name", "Unknown Code")
    out_data["abbr_name"] = data.get("name", "Unknown")
    
    out_data["n"] = data.get("n", 0)
    out_data["k"] = data.get("k", 0)
    out_data["d"] = data.get("d", 0)
    
    H = data.get("H", "")
    if isinstance(H, str):
        H = H.split()
    
    L = data.get("L", "")
    if isinstance(L, str):
        L = L.split()
        
    # Check if the code is CSS by inspecting if all stabilizers are purely X or purely Z
    is_actually_css = True
    for h in H:
        has_x = any(c in 'XY' for c in h)
        has_z = any(c in 'ZY' for c in h)
        if has_x and has_z:
            is_actually_css = False
            break
            
    if is_actually_css:
        H_x = [[1 if c in 'XY' else 0 for c in h] for h in H if any(c in 'XY' for c in h) and not any(c in 'ZY' for c in h)]
        H_z = [[1 if c in 'ZY' else 0 for c in h] for h in H if any(c in 'ZY' for c in h) and not any(c in 'XY' for c in h)]
        
        L_x = [[1 if c in 'XY' else 0 for c in l] for l in L if any(c in 'XY' for c in l) and not any(c in 'ZY' for c in l)]
        L_z = [[1 if c in 'ZY' else 0 for c in l] for l in L if any(c in 'ZY' for c in l) and not any(c in 'XY' for c in l)]
    else:
        H_x = [[1 if c in 'XY' else 0 for c in h] for h in H]
        H_z = [[1 if c in 'ZY' else 0 for c in h] for h in H]
        L_x = [[1 if c in 'XY' else 0 for c in l] for l in L]
        L_z = [[1 if c in 'ZY' else 0 for c in l] for l in L]
        
    out_data["is_self_dual"] = (H_x == H_z) and (L_x == L_z)
    
    out_data["H_x"] = H_x
    out_data["H_z"] = H_z
    out_data["L_x"] = L_x
    out_data["L_z"] = L_z
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("{\n")
        f.write(f'  "full_name": "{out_data["full_name"]}",\n')
        f.write(f'  "abbr_name": "{out_data["abbr_name"]}",\n')
        f.write(f'  "n": {out_data["n"]},\n')
        f.write(f'  "k": {out_data["k"]},\n')
        f.write(f'  "d": {out_data["d"]},\n')
        f.write(f'  "is_self_dual": {"true" if out_data["is_self_dual"] else "false"},\n')
        
        for key in ["H_x", "H_z", "L_x", "L_z"]:
            f.write(f'  "{key}": [\n')
            for i, row in enumerate(out_data[key]):
                f.write("    " + json.dumps(row))
                if i < len(out_data[key]) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("  ]" + (",\n" if key != "L_z" else "\n"))
        f.write("}\n")
        
    print(f"Successfully transformed {args.input} -> {args.output}")

if __name__ == "__main__":
    main()
