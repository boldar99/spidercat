import re

tex = """
    $[[7,1,3]]$ Steane $\ket{\overline{0}}$ & $\textbf{11}$ \cite{Got16:SteaneCodeFT} & $15$ & $8$ & $3$ & $10$ & $[2.7 , \,\, 2.9] \times 10^{-5}$ & $[0.9783, \,\, 0.9784]$\\
        \addlinespace[0.8mm]
    $[[9,1,3]]$ rot. surface $\ket{\overline{0}}$ & $\textbf{8}$ \cite{got23:sur3} & $26$ & $12$ & $9$ & $9$ & $[2.4, \,\, 2.6] \times 10^{-5}$ & $[0.97150,\,\,0.97158] $\\
        \addlinespace[0.8mm]
        $[[17,1,5]]$ color code $\ket{\overline{0}}$ & $\textbf{71}$ \cite{Peh25:automatedSynthesis} & $74$ & $23$ & $21$ & $25$ & $[7.7 , \,\, 18.2] \times 10^{-7}$ & $[0.8945,\,\,0.8948]$\\
        \addlinespace[0.8mm]
        $[[25,1,5]]$ rot. surface $\ket{\overline{0}}$ & $120$ \cite{Den02:Topological} & $\textbf{92}$ & $32$ & $28$ & $23$ & $[6.7 , \,\, 24.2] \times 10^{-7}$ & $[0.8980,\,\,0.8984]$\\
        \addlinespace[0.8mm]
        $[[49,1,5]]$ triorthogonal $\ket{\overline{+}}$ & $936$ & $\textbf{361}$ & $95$ & $105$ & $59$ & $[4.2 , \,\, 4.7]\times 10^{-5}$ & $[0.585,\,\,0.584]$\\
        \addlinespace[0.8mm]
        $[[20,2,6]]$ self-dual $\ket{\overline{00}}$& $376$ & $\textbf{145}$ & $36$ & $47$ & $54$ & $[2.3, \,\, 9.7] \times 10^{-8}$ & $[0.8234,\,\,0.8235$]\\
        \addlinespace[0.8mm]
        $[[23,1,7]]$ Golay $\ket{\overline{0}}$ & $297$ \cite{Pae11:GolayCode} & $\textbf{237}$ & $44$ & $80$ & $33$ & $[1.8, \,\, 3.1] \times 10^{-7}$ & $[0.7095,\,\,0.7099]$\\
        \addlinespace[0.8mm]
        $[[31,1,7]]$ color code $\ket{\overline{0}}$ & $421$  \cite{Peh25:automatedSynthesis} & $\textbf{211}$ & $55$ & $69$ & $58$ & $[2.1, \,\, 5.4] \times 10^{-7}$ & $[0.750,\,\,0.751]$\\
        \addlinespace[0.8mm]
        $[[49,1,7]]$ rot. surface $\ket{\overline{0}}$ & $336$  \cite{Den02:Topological} & $\textbf{262}$ & $64$ & $85$ & $46$ & $[1.2 , \,\, 4.4]\times 10^{-7}$ & $[0.702,\,\,0.703]$\\
        \addlinespace[0.8mm]
        $[[95,1,7]]$ triorthogonal $\ket{\overline{+}}$ & $ 4792 $ & $\textbf{1175}$ & $258$ & $380$ & $389$ & $ [4.4, \,\, 6.3] \times 10^{-5} $ & $ [0.240, \,\, 0.241] $\\
        \addlinespace[0.8mm]
        $[[49,1,9]]$ color code $\ket{\overline{0}}$ & $1020$ & $\textbf{408}$ & $93$ & $136$ & $123$ & $[1.1, \,\, 5.8]\times 10^{-7} $ & $[0.531,\,\,0.532]$\\
        \addlinespace[0.8mm]
        $[[81,1,9]]$ rot. surface $\ket{\overline{0}}$ & $720$  \cite{Den02:Topological} & $\textbf{614}$ & $141$ & $206$ & $129$ & $[2.0, \,\, 11] \times 10^{-7}$ & $[0.355,\,\,0.356]$\\
        \addlinespace[0.8mm]
        $[[47,1,11]]$ self-dual $\ket{\overline{0}}$ & $4140$ & $\textbf{1033}$ & $186$ & $388$ & $292$ & $[3.6 , \,\, 17] \times 10^{-7}$ & $[0.122, \,\, 0.123]$\\
        \addlinespace[0.8mm]
        $[[71,1,11]]$ color code $\ket{\overline{0}}$ & $1860$ & $\textbf{829}$ & $177$ & $268$ & $282$ & $[4.4 , \,\, 29] \times 10^{-8}$ & $[0.214, \,\, 0.215]$\\
"""

code_mapping = {
    "7,1,3": "7_1_3", "9,1,3": "9_1_3", "17,1,5": "17_1_5", "25,1,5": "25_1_5",
    "49,1,5": "49_1_5", "20,2,6": "20_2_6", "23,1,7": "23_1_7", "31,1,7": "31_1_7",
    "49,1,7": "49_1_7", "95,1,7": "95_1_7", "49,1,9": "49_1_9", "81,1,9": "81_1_9",
    "47,1,11": "47_1_11", "71,1,11": "71_1_11"
}

for line in tex.strip().split('\n'):
    if not line.strip() or line.strip().startswith(r'\addlinespace'): continue
    # extract code
    m = re.search(r'\[\[(\d+,\d+,\d+)\]\]', line)
    if not m: continue
    code = code_mapping[m.group(1)]
    
    parts = line.split('&')
    cxs_part = parts[2].strip()
    sim_part = parts[3].strip()
    flags_part = parts[4].strip()
    depth_part = parts[5].strip()
    ler_part = parts[6].strip()
    ar_part = parts[7].split('\\\\')[0].strip()
    
    cx = re.sub(r'[^0-9]', '', cxs_part)
    flags = re.sub(r'[^0-9]', '', flags_part)
    sim = re.sub(r'[^0-9]', '', sim_part)
    depth = re.sub(r'[^0-9]', '', depth_part)
    
    print(f'    "{code}": {{"cx": {cx}, "flags": {flags}, "sim_qubits": "{sim}", "depth": "{depth}", "ler_str": r"${ler_part}$", "ar_str": r"${ar_part}$"}},')

