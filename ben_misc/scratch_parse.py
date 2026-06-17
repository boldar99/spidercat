import re
import sys

def parse_tikz(filepath):
    lines = open(filepath).read().splitlines()
    
    wires = {} # wire -> list of (x, op_string)
    
    current_op = None
    
    for i, line in enumerate(lines):
        m = re.match(r'% Line \d+: (.*)', line)
        if m:
            op_text = m.group(1)
            # now we need the x coordinate from the next few lines
            x_coord = None
            for j in range(i+1, min(i+5, len(lines))):
                # match \draw ... (x, y) ...
                m_coord = re.search(r'\(\s*([\d\.]+)\s*,\s*[-]?[\d\.]+\s*\)', lines[j])
                if m_coord:
                    x_coord = float(m_coord.group(1))
                    break
            
            if x_coord is not None:
                # wire start
                m_w = re.match(r'(\d+) W \\ket\{([0+])\}', op_text)
                if m_w:
                    wire = int(m_w.group(1))
                    state = m_w.group(2)
                    if wire not in wires: wires[wire] = []
                    wires[wire].append((x_coord, f"INIT_{state} {wire}"))
                    continue
                    
                m_start = re.match(r'(\d+) START', op_text)
                if m_start:
                    wire = int(m_start.group(1))
                    # need to find the state
                    state = '+' # default
                    for j in range(i+1, min(i+5, len(lines))):
                        m_ket = re.search(r'\\ket\{([0+])\}', lines[j])
                        if m_ket:
                            state = m_ket.group(1)
                            break
                    if wire not in wires: wires[wire] = []
                    wires[wire].append((x_coord, f"INIT_{state} {wire}"))
                    continue
                    
                m_cnot = re.match(r'(\d+) \+(\d+)', op_text)
                if m_cnot:
                    ctrl = int(m_cnot.group(1))
                    targ = int(m_cnot.group(2))
                    # Add to an operations list? Or we just collect all ops globally?
                    # Since operations can span wires, global is better.
                    pass
                
                m_meas = re.match(r'(\d+) M \{\\scriptsize \$([XZ])\$\}', op_text)
                if m_meas:
                    pass

# Actually, let's collect ALL operations in a single global list and sort by x.
def parse_all(filepath):
    lines = open(filepath).read().splitlines()
    ops = []
    
    for i, line in enumerate(lines):
        m = re.match(r'% Line \d+: (.*)', line)
        if m:
            op_text = m.group(1)
            x_coord = None
            for j in range(i+1, min(i+5, len(lines))):
                # match \draw ... (x, y) ...
                m_coord = re.search(r'\(\s*([\d\.]+)\s*,\s*[-]?[\d\.]+\s*\)', lines[j])
                if m_coord:
                    x_coord = float(m_coord.group(1))
                    break
            
            if x_coord is not None:
                m_w = re.match(r'(\d+) W \\ket\{([0+])\}', op_text)
                if m_w:
                    ops.append((x_coord, i, f"INIT_{m_w.group(2)} {m_w.group(1)}"))
                    continue
                    
                m_start = re.match(r'(\d+) START', op_text)
                if m_start:
                    state = '+'
                    for j in range(i+1, min(i+5, len(lines))):
                        m_ket = re.search(r'\\ket\{([0+])\}', lines[j])
                        if m_ket:
                            state = m_ket.group(1)
                            break
                    ops.append((x_coord, i, f"INIT_{state} {m_start.group(1)}"))
                    continue
                    
                m_cnot = re.match(r'(\d+) \+(\d+)', op_text)
                if m_cnot:
                    ops.append((x_coord, i, f"CX {m_cnot.group(1)} {m_cnot.group(2)}"))
                    continue
                
                m_meas = re.match(r'(\d+) M \{\\scriptsize \$([XZ])\$\}', op_text)
                if m_meas:
                    ops.append((x_coord, i, f"M{m_meas.group(2)} {m_meas.group(1)}"))
                    continue
                    
                m_end = re.match(r'(\d+) END', op_text)
                if m_end:
                    continue
                    
                m_end2 = re.match(r'(\d+) (\d+) END', op_text)
                if m_end2:
                    continue
                    
                m_start2 = re.match(r'(\d+) (\d+) START', op_text)
                if m_start2:
                    w1 = m_start2.group(1)
                    w2 = m_start2.group(2)
                    state1 = '0'
                    state2 = '+'
                    # Hardcode based on tikz
                    for j in range(i+1, min(i+10, len(lines))):
                        m_ket = re.search(r'node\[left\] \{\\ket\{([0+])\}\}', lines[j])
                        if m_ket:
                            if state1 is None:
                                state1 = m_ket.group(1)
                            else:
                                state2 = m_ket.group(1)
                                break
                    # Let's just hardcode what we saw: 21 is 0, 23 is +, 22 is 0, 24 is +
                    if "21 23 START" in op_text:
                        ops.append((x_coord, i, f"INIT_0 21"))
                        ops.append((x_coord, i, f"INIT_+ 23"))
                    elif "22 24 START" in op_text:
                        ops.append((x_coord, i, f"INIT_0 22"))
                        ops.append((x_coord, i, f"INIT_+ 24"))
                    continue

    ops.sort(key=lambda x: (x[0], x[1]))
    for op in ops:
        print(f"{op[0]:.1f}\t{op[2]}")

parse_all('17_1_5_prep.tikz')
