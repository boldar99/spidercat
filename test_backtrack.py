import spiderstate.cat_at_origin as cao
import random

# Monkeypatch random shuffle into matcher to force backtracking
import spiderstate.spider_leg_matcher as slm
original_backtrack = None

is_self_dual, H_x, H_z, L_x, L_z, d = cao.load_qecc("12_2_4")
cao.row_optimized_cat_at_origin(H_z, d=d, record=True)
