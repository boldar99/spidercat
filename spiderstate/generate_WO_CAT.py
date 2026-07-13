import json
import random
import time
from pathlib import Path
import networkx as nx

from spiderstate.well_ordered_cat_state import well_ordered_ft_cat_state_data

cwd = Path(__file__).parent

def init_states_folder():
    Path(f"{cwd}/cat_states_data").mkdir(parents=True, exist_ok=True)

def save_state_data(G, F, roots, D, edge, n, t):
    file_name = f"{cwd}/cat_states_data/well_ordered_state_t{t}_n{n}.json"
    with open(file_name, "w") as f:
        data = {
            "n": n,
            "t": t,
            "G": nx.node_link_data(G, edges="links"),
            "F": nx.node_link_data(F, edges="links"),
            "roots": roots,
            "D": nx.node_link_data(D, edges="links"),
            "edge": edge,
        }
        json.dump(data, f)

def process_cell(n, t, replace=False, regenerate_graph=False):
    file_name = f"{cwd}/cat_states_data/well_ordered_state_t{t}_n{n}.json"
    
    if not replace and Path(file_name).is_file():
        return " X "
        
    try:
        G_alt, F_alt, roots, dependency_graph, edge = well_ordered_ft_cat_state_data(n, t, force_generate=True, regenerate_graph=regenerate_graph)
        save_state_data(G_alt, F_alt, roots, dependency_graph, edge, n, t)
        return " V "
    except Exception as e:
        return " - "

if __name__ == "__main__":
    start_time = time.time()
    random.seed(4092)

    init_states_folder()

    N = 100
    TS = [5]

    print("Generating well-ordered CAT state data structures for given n and t")
    print()
    
    ns = range(1, N + 1)
    
    print('t\\n |', end=' ')
    for f in ns:
        print(f if f > 9 else f' {f}', end=' ')
    print()
    print("-" * 3 * (len(ns) + 2))
    
    for t in TS:
        print(f"t={t} |", end=' ', flush=True)
        results_generator = (process_cell(n, t, replace=False, regenerate_graph=True) for n in ns)
        
        for cell_str in results_generator:
            print(cell_str, end='', flush=True)
        print()
    print()
    print(f"Files saved to: {cwd}/cat_states_data")
    print()
    print("--- %s seconds ---" % (time.time() - start_time))
