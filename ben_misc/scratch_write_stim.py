lines = """0.0	INIT_+ 1
0.0	INIT_0 2
0.0	INIT_0 3
0.0	INIT_+ 4
0.0	INIT_0 5
0.0	INIT_+ 6
0.0	INIT_0 7
0.0	INIT_0 8
0.0	INIT_0 9
0.0	INIT_+ 10
0.0	INIT_+ 11
0.0	INIT_+ 12
0.0	INIT_+ 13
0.0	INIT_+ 14
0.0	INIT_0 15
0.0	INIT_0 16
0.0	INIT_0 17
9.0	CX 13 16
12.0	CX 6 8
12.0	CX 11 9
12.0	CX 1 3
15.0	CX 12 15
33.0	CX 13 7
39.0	CX 4 8
45.0	CX 16 5
51.0	CX 11 17
57.0	CX 10 12
75.0	CX 13 15
78.0	CX 7 2
81.0	CX 14 8
84.0	CX 1 4
87.0	CX 9 16
90.0	CX 5 3
93.0	CX 10 11
115.5	INIT_+ 18
115.5	CX 12 9
115.5	CX 6 5
115.5	CX 15 17
115.5	CX 8 7
123.0	INIT_+ 18
138.0	CX 18 14
157.5	CX 18 4
160.5	INIT_+ 19
163.5	CX 14 13
168.0	INIT_+ 19
180.0	MX 18
183.0	CX 4 2
189.0	INIT_0 20
195.0	CX 19 1
196.5	INIT_0 20
217.5	CX 19 20
240.0	CX 19 10
258.0	CX 19 4
276.0	CX 19 9
294.0	CX 19 5
312.0	CX 19 17
330.0	CX 19 7
348.0	CX 19 20
363.0	MZ 20
369.0	CX 19 13
388.5	MX 19
421.5	INIT_0 21
421.5	INIT_+ 23
429.0	INIT_0 21
429.0	INIT_+ 23
444.0	CX 23 21
462.0	CX 16 21
480.0	CX 12 21
498.0	CX 10 21
516.0	CX 17 21
534.0	CX 23 21
549.0	MZ 21
549.0	MX 23
580.5	INIT_0 22
580.5	INIT_+ 24
588.0	INIT_0 22
588.0	INIT_+ 24
603.0	CX 24 22
621.0	CX 3 22
639.0	CX 6 22
657.0	CX 14 22
675.0	CX 4 22
693.0	CX 7 22
711.0	CX 24 22
726.0	MZ 22
726.0	MX 24"""

out = []
wire_dirty = {i: True for i in range(24)} # True if it needs init or had operation

for line in lines.strip().split('\n'):
    time, op = line.split('\t')
    parts = op.split(' ')
    if parts[0] == 'INIT_+':
        w = int(parts[1]) - 1
        if wire_dirty[w]:
            out.append(f"RX {w}")
            wire_dirty[w] = False
    elif parts[0] == 'INIT_0':
        w = int(parts[1]) - 1
        if wire_dirty[w]:
            out.append(f"R {w}")
            wire_dirty[w] = False
    elif parts[0] == 'CX':
        c = int(parts[1]) - 1
        t = int(parts[2]) - 1
        out.append(f"CX {c} {t}")
        wire_dirty[c] = True
        wire_dirty[t] = True
    elif parts[0] == 'MZ':
        w = int(parts[1]) - 1
        out.append(f"M {w}")
        wire_dirty[w] = True
    elif parts[0] == 'MX':
        w = int(parts[1]) - 1
        out.append(f"MX {w}")
        wire_dirty[w] = True

with open('/Users/boldi/PycharmProjects/FT-circuit-synthesis-ZX/17_1_5_prep.stim', 'w') as f:
    f.write('\n'.join(out) + '\n')
