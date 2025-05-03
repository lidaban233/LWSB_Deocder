from helper_functions_LDPC import load_code
import numpy as np
H_filename = 'codes/LDPC_chk_mat_270_150.txt'
G_filename = 'codes/LDPC_gen_mat_270_120.txt'
code = load_code(H_filename, G_filename)
H = code.H
G = code.G
var_degrees = code.var_degrees
chk_degrees = code.chk_degrees
num_edges = sum(code.num_edges)
u = code.u
d = code.d
n = code.n
m = code.m
k = code.k
np.set_printoptions(threshold=np.inf)
print(H.shape)
