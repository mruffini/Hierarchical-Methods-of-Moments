#######################################################################
## Imports
#######################################################################
import numpy as np
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sprs
from scipy.stats import multinomial

#######################################################################
## Main functions
#######################################################################


def MAP_assign_clusters_stm(X, M, omega):
    """
    Assign documents in a corpus X to the most likely topic,
    via MAP assignment
    @param M: the conditional expectations matrix
    @param omega: the mixing weights
    @param X: a bag-of-words documents distributed
        as a Single Topic Model, with n rows an d columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    """

    n,d = np.shape(X)

    # Project on the symplex
    M[M<=0] = 0.000001
    M[M>=1] = 0.999999
    M = M/M.sum(0)

    d, k = M.shape
    wmu = np.zeros((n, k))
    nn = X.sum(1)
    for i in range(k):
        mu = M[:, i].reshape(d)
        wmu[:, i] = multinomial.logpmf(X, n=nn, p=mu) + np.log(omega[i])

    CL = np.argmax(wmu, 1)

    return CL


def retrieve_tensors_stm(X):
    """
    Returns a the three tensors M1, M2 and M3 to be used
    to learn the Single Topic Model
    @param X: a bag-of-words documents distributed
        as a Single Topic Model, with n rows an d columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    """

    (n, d) = np.shape(X)

    M1 = np.sum(X,0)/np.sum(X)

    W = X - 1
    W[W < 0] = 0
    W2 = X - 2
    W2[W2 < 0] = 0
    Num = X * W
    Den = np.sum(X, 1)
    wDen = Den - 1
    wDen[wDen < 0] = 0
    wwDen = Den - 2
    wwDen[wwDen < 0] = 0

    Den1 = sum(Den * wDen)
    Den2 = sum(Den * wDen * wwDen)

    Diag = np.sum(Num, 0) / Den1

    M2 = np.transpose(X).dot(X) / Den1
    M2[range(d), range(d)] = Diag

    M3 = np.zeros((d, d, d))
    for j in range(d):
        Y = X[:, j].reshape((n, 1))
        Num = X * Y * W
        Diag = np.sum(Num, 0) / Den2
        wM3 = (Y * X).T.dot(X) / Den2
        wM3[range(d), range(d)] = Diag
        rr = np.sum(Y * W[:, j].reshape((n, 1)) * X,0) / Den2
        wM3[j, :] = rr
        wM3[:, j] = rr
        wM3[j, j] = np.sum(Y * W[:, j].reshape((n, 1)) * W2[:, j].reshape((n, 1))) / Den2
        M3[j] = wM3

    return M1, M2, M3


def projsplx(y):
    """
    projects a vector y onto the symplex, as in Reference [24]
    @param y: vector to be projected
    """

    m = len(y)
    bget = False
    s = y[y.argsort()[::-1]]
    tmpsum = 0
    for ii in np.arange(m-1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (1+ii)
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m-1] -1) / m

    x = y - tmax
    x[x<0] = 0

    return x


def TestFunction(a,c1,c2,c3,c4,c5):
    """
    Implements function F(a) of Theorem 3.3 to efficiently perform
    simultaneous diagonalization when l=
    @param a: the independent variable of the function
    @params c1,c2,c3,c4,c5: the parameters of the function
    """

    return a**4*c1 + a**3*np.sqrt(1-a**2)*c2 + a*np.sqrt(1-a**2)*c3 + a**2*c4 + c5


def SIDIWO_lowrank_single_topic_model(X):
    """
    Implements SIDIWO for the single topic model using l=2
    @param X: a bag-of-words documents distributed
        as a Single Topic Model, with n rows an d columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    """

    M1, M2, M3 = retrieve_tensors_stm(X)

    M, omega = SIDIWO_grid(M1,M2,M3)

    return M, omega


def SIDIWO_grid(M1,M2,M3):
    """
    Implements Algorithm 1 - SIDIWO when l=2
    @param M1,M2,M3: the symmetric moments
    """

    d, _ = M2.shape
    U, S, _ = np.linalg.svd(M2)

    # D is the pseudoinverse of the whitening matrix
    E = U[:,:2].dot(np.diag(np.sqrt(S[:2])))
    D = np.linalg.pinv(E)

    # Whiten the slices of M3
    H = np.zeros((2,2,d))
    for i in range(d):
        H[:,:,i] = D.dot(M3[i].dot(D.T))

    # Find the matrix that performs simultaneous diagonalization by griding, using thm 3.3
    h = H[0, 1, :]
    f = H[0, 0, :] - H[1, 1, :]

    c1 = np.sum(4 * h ** 2 - f ** 2)
    c2 = -4 * np.sum(f * h)
    c3 = 2 * np.sum(f * h)
    c4 = np.sum(f ** 2) - 4 * np.sum(h ** 2)
    c5 = np.sum(h ** 2)

    x = np.arange(-1, 1, 0.00001)

    Y = TestFunction(x, c1, c2, c3, c4, c5)
    a = x[np.argmin(Y)]

    O = np.array([[np.sqrt(1 - a ** 2), a], [-a, np.sqrt(1 - a ** 2)]])

    # Solves system at iteration 4 of Algorithm 1 to find the parameters of the model
    M =  E.dot(O)
    M = M/np.sign(M.sum(0))

    omega =  (np.linalg.pinv(M).dot(M1))

    M = M.dot(np.diag(1/omega))
    omega = omega**2/(omega**2).sum()

    return M, omega


def get_H(X,pu,k,N,n):
    """
    Working function for calculating the tensor H optimally.
    @param X: a bag-of-words documents distributed
    as a Single Topic Model, with n rows an d columns;
    at position (i,j) we have the number of times the word j appeared in doc. i,


    """

    H1 = np.zeros((n,k,k))
    wpuLeft = np.zeros((n, k))
    for i in range(N):
        nzX = np.nonzero(X[i, :])[0]
        Xnz = X[i, nzX]
        L = len(nzX)
        wpuRight = Xnz.dot(pu[:,nzX].T)
        wpuLeft[:,:] = 0
        wpuLeft[nzX,:] = Xnz.reshape(L,1)*(wpuRight.reshape(1,k))
        I = np.repeat(nzX,k).astype(int)
        J = np.array(range(k)*L).astype(int)
        wpuRight_ = np.broadcast_to(wpuRight.reshape(1,1,k), H1.shape)
        H1[I,J,:] += (wpuLeft[I, J, None]*wpuRight_[I, J, :])
    return H1


def SIDIWO_lowrank_single_topic_model_fast(X):
    """
    Optimized implemenatation of SIDIWO for the single topic model using l=2
    Same as SIDIWO_lowrank_single_topic_model but with sparse matrices and randomized SVD
    @param X: a bag-of-words documents distributed
        as a Single Topic Model, with n rows an d columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    """
    k=2

    spX = sprs.csr_matrix(X)
    n, d = np.shape(spX)
    M1 = np.sum(X, 0) / np.sum(X)
    W = np.zeros(np.shape(X))
    W[X > 0] = X[X > 0] - 1
    cW = np.zeros(np.shape(X))
    cW[W > 0] = W[W > 0] - 1
    Num = spX.multiply(W)
    X2 = spX.multiply(spX)
    Den = np.asarray(np.sum(spX, 1))
    wDen = Den - 1
    wDen[wDen < 0] = 0
    wwDen = Den - 2
    wwDen[wwDen < 0] = 0
    DenwDen = Den * wDen
    Den1 = np.sum(DenwDen)
    Den2 = np.sum(DenwDen * wwDen)
    Diag = np.sum(Num, 0) / Den1
    C = spX.T.dot(spX).toarray() / Den1

    C[range(d), range(d)] = Diag

    U,S,V = randomized_svd(C, n_components=k, n_iter=5, random_state=None)

    # Whitening matrix
    E = U.dot((np.diag(np.sqrt(S))))

    # Feasible solution
    D = np.linalg.pinv(E)

    # Calculates the tensor H whose slices are the whitened slices of M3
    OldDiags = ((X2).T.dot(spX)).toarray()
    NewDiags = ((Num.T).dot(spX)).toarray()
    NewDiags[range(d), range(d)] = np.sum(Num.multiply(cW),0)


    NewLateral = Num.T.dot(spX).toarray()
    NewLateral[range(d), range(d)] = 0
    OldLateral = X2.T.dot(spX).toarray()
    OldLateral[range(d), range(d)] = 0

    Lateral = (NewLateral - OldLateral) / Den2
    Diags = (NewDiags - OldDiags) / Den2

    LateralH1 = (D.dot(Lateral.T)).reshape(k, 1, d) * D.reshape(1, k, d)
    LateralH2 = np.zeros(LateralH1.shape)

    H = np.zeros(LateralH1.shape)
    Center = np.zeros(LateralH1.shape)
    wH = get_H(X, D, k, n, d) / Den2
    wCenter = (D.reshape((1, k, d)) * Diags.T.reshape((d, 1, d))).dot(D.T)

    for i in range(d):
        H[:,:,i] = wH[i,:,:]
        Center[:,:,i] = wCenter[i,:,:]
        LateralH2[:, :, i] = LateralH1[:, :, i].T

    H = H + Center + LateralH1 + LateralH2

    # Find the matrix that performs simultaneous diagonalization by griding, using thm 3.3

    h = H[0, 1, :]
    f = H[0, 0, :] - H[1, 1, :]

    c1 = np.sum(4 * h ** 2 - f ** 2)
    c2 = -4 * np.sum(f * h)
    c3 = 2 * np.sum(f * h)
    c4 = np.sum(f ** 2) - 4 * np.sum(h ** 2)
    c5 = np.sum(h ** 2)

    x = np.arange(-1, 1, 0.00001)

    Y = TestFunction(x, c1, c2, c3, c4, c5)
    a = x[np.argmin(Y)]

    O = np.array([[np.sqrt(1 - a ** 2), a], [-a, np.sqrt(1 - a ** 2)]])

    # Solves system at iteration 4 of Algorithm 1 to find the parameters of the model
    M = E.dot(O)
    M = M / np.sign(M.sum(0))

    omega = (np.linalg.pinv(M).dot(M1))

    M = M.dot(np.diag(1 / omega))
    omega = omega ** 2 / (omega ** 2).sum()

    return M, omega


def create_cluster_graph_single_topic_model(X, depth, CL_input, Y = None, use_fast_implementation = True):
    """
    Recursive implementation of Algorithm 2, to find a divisive binary tree by
    splitting a corpus into two parts.
    @param X: a bag-of-words documents distributed 
        as a Single Topic Model, with n rows an d columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    @param depth: the current depth of the tree; 
        depth =1 means that we are at a leaf
        depth >1 the algorithm perform a splits and calls himself with depth=depth-1
    @param CL_input: a list of lists containing the binary tree.
        Each entry is a node, containing: the depth of the node and the binary tree behind him
    @param Y: the list of samples contained in the current node, to be splitted at this iteration. 
    @param use_fast_implementation: whether to use the optimized implementation or not 
    """

    if Y is None:
        Y = np.arange(len(X))

    print('Current depth: ',depth)

    # If I have data, I split the dataset (the node) into two parts
    if np.sum(X):
        if use_fast_implementation:
            M,P = SIDIWO_lowrank_single_topic_model_fast(X)
        else:
            M,P = SIDIWO_lowrank_single_topic_model(X)
    else:
        return [X,np.zeros(len(X))]
    CL = MAP_assign_clusters_stm(X, M, P)

    # If I am not at a leaf data, call this function again on the childs of this node
    if depth>1:
        CL_input.append([depth,[Y[CL==0],Y[CL==1]]])
        X1 = X[CL == 0]
        wX1 = create_cluster_graph_single_topic_model(X1, depth-1,CL_input,Y[CL==0], use_fast_implementation)
        X2 = X[CL == 1]
        wX2 = create_cluster_graph_single_topic_model(X2, depth-1,CL_input,Y[CL==1], use_fast_implementation)
        return [(X1,X2),[wX1,wX2]]
    else:
        CL_input.append([depth,[Y[CL==0],Y[CL==1]]])
        X1 = X[CL == 0]
        X2 = X[CL == 1]
        return (X1,X2)
