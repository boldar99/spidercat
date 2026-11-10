OPENQASM 2.0;
include "qelib1.inc";
qreg f[1];
qreg q[2];
h q[0];
cx q[0],f[0];
cx q[0],q[1];
cx q[0],f[0];
