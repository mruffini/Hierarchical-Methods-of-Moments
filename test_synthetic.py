# script to reproduce the experiment at section 4.1 of the paper "Hierarchical Methods of Moments"

import other_methods as om
import numpy as np
import pandas as pd
import random_generators as rng
import matplotlib.pyplot as pl
import engines as eg
from sklearn.metrics import adjusted_rand_score
import time

# Sample size
n = 400
# True number of clusters
k = 8

# Read a matrix of centers with a hierarchical structure
M =pd.read_csv('data/M.csv', sep=',').values.astype(np.float)
# Add random noise
M = M + np.random.random(M.shape)/5
# Project the centers on the simplex
M = M/M.sum(0)

d,k = M.shape
# Random data generation
X, M, omega, x= rng.generate_sample_single_topic_model(n, d, k, c = 100, M = M)

# Run hierarchical SIDIWO, calculates time and adjusted_rand_score
CL = []
t0 = time.time()
wCL = eg.create_cluster_graph_single_topic_model(X,3,CL,use_fast_implementation=True)
t_SIDIWO = time.time()-t0
print('hierarchical SIDIWO Time', t_SIDIWO)

AA =[CL[6][1][0],CL[6][1][1],CL[5][1][0],CL[5][1][1],CL[2][1][0],CL[2][1][1],CL[3][1][0],CL[3][1][1]]
CLusters = np.zeros(len(X))

for i,a in enumerate(AA):
    CLusters[a] = i
print('hierarchical SIDIWO Score', adjusted_rand_score(x,CLusters))

# Run TPM, calculates time and adjusted_rand_score
t0 = time.time()
M1,M2,M3 = eg.retrieve_tensors_stm(X)
M,P = om.learn_LVM_Tensor14(M2,M3,k)
CLTPM = eg.MAP_assign_clusters_stm(X, M, P)
ttpm =   time.time() - t0
print('TPM Time', ttpm)

pl.figure()
pl.imshow(X[CLTPM.argsort()].T, cmap='gray')
print('TPM Score', adjusted_rand_score(x,CLTPM))



# Run SVD method, calculates time and adjusted_rand_score
t0 = time.time()
M1,M2,M3 = eg.retrieve_tensors_stm(X)
M,P = om.learn_LVM_AnanHMM12(M1,M2,M3,k)
CLTPM = eg.MAP_assign_clusters_stm(X, M, P)
tsvd =   time.time() - t0
print('SVD Time', tsvd)

pl.figure()
pl.imshow(X[CLTPM.argsort()].T, cmap='gray')
print('SVD Score', adjusted_rand_score(x,CLTPM))


