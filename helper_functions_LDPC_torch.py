import numpy as np
import tensorflow as tf
import torch


class Code:
    def __init__(self):
        self.num_edges = 0
# self.n = n
# self.k = k


def load_code(H_filename, G_filename):
    # parity-check matrix; Tanner graph parameters
    fileH = np.loadtxt(open(H_filename, "rb"), dtype=np.int32) - 1
    m = max(fileH[:, 0]) + 1
    n = max(fileH[:, 1]) + 1
    H_matrix = np.zeros([m, n], dtype=np.int32)
    H_matrix[fileH[:, 0], fileH[:, 1]] = 1

    var_degrees = []
    var_edges = []
    chk_degrees = []
    chk_edges = []
    d = [];

    i = 0

    for jj in range(n):
        isi2 = []
        edges = []
        el1 = np.where(H_matrix[:, jj] != 0)[0]
        for k in range(len(el1)):
            edges.append(el1[k])
            isi2.append(i)
            i = i + 1
        var_degrees.append(len(el1))
        var_edges.append(edges)
        d.append(isi2)

    for jj in range(m):
        el2 = np.where(H_matrix[jj, :] != 0)[0]
        chk_degrees.append(len(el2))
        edges = []
        for k in range(len(el2)):
            edges.append(el2[k])
        chk_edges.append(edges)

    num_edges = sum(H_matrix)

    u = [[] for _ in range(0, m)]

    # Matriks G
    fileG = np.loadtxt(open(G_filename, "rb"), dtype=np.int32) - 1
    m_G = max(fileG[:, 0]) + 1
    n_G = max(fileG[:, 1]) + 1
    G_matrix = np.zeros([m_G, n_G], dtype=np.int32)
    G_matrix[fileG[:, 0], fileG[:, 1]] = 1

    edge = 0
    for i in range(0, m):
        for j in range(0, chk_degrees[i]):
            v = chk_edges[i][j]
            for e in range(0, var_degrees[v]):
                if (i == var_edges[v][e]):
                    u[i].append(d[v][e])

    code = Code()
    code.H = H_matrix
    code.G = G_matrix
    code.var_degrees = var_degrees
    code.chk_degrees = chk_degrees
    code.num_edges = num_edges
    code.u = u
    code.d = d
    code.n = n
    code.m = m
    code.k = n - m
    return code


# compute the "soft syndrome"
def syndrome(soft_output, code):
    H = torch.tensor(code.H, dtype=torch.float32)  # Ensure H is a PyTorch tensor
    m, n = H.shape  # Get dimensions of the H matrix

    soft_syndrome = []  # To store the computed syndrome for each check node
    for c in range(m):  # Iterate over each check node
        # Find the variable nodes connected to the check node
        variable_nodes = torch.where(H[c] == 1)[0]  # Indices of non-zero entries in row c of H

        # Gather the relevant soft_output values
        temp = soft_output[variable_nodes]

        # Compute the sign product
        temp1 = torch.prod(torch.sign(temp), dim=0)

        # Compute the minimum absolute value
        temp2 = torch.min(torch.abs(temp), dim=0).values

        # Append the product of the sign product and minimum absolute value
        soft_syndrome.append(temp1 * temp2)

    # Stack all syndromes into a single tensor
    soft_syndrome = torch.stack(soft_syndrome)
    return soft_syndrome
