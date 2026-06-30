import stim
from spiderstate.cat_at_origin import cat_at_origin_with_verification
from spiderstate.qec_utils import parse_qec_code

code = parse_qec_code('12_2_4')
try:
    circ = cat_at_origin_with_verification(code.H_x, code.H_z, d=code.d, t=2)
    print("Circuit depth:", len(circ))
except Exception as e:
    import traceback
    traceback.print_exc()
