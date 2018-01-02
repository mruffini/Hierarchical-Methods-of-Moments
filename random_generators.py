#######################################################################
## Imports
#######################################################################
import numpy as np

#######################################################################
## Main functions
#######################################################################


def generate_sample_single_topic_model(n, d, k, c = 1000, M = None):

    """
    Generates a sample of n synthetic bag-of-words documents distributed
    as a Single Topic Model.
    Return:
    the generated samples X, with n rows and d columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    the topic-word probability matrix M, with n rows an k columns;
        at position (i,j) we have the probability of the word i under topic k.
    the topic probability array omega, with k entries.
        at position (i) we have the probability of drawing topic i.
    the true topic of each text x
    @param n: The number of synthetic documents to be generated
    @param d: the size of the vocabulary
    @param k: the number of hidden topics
    @param c: the number of words appearing in each document.
    """


    omega = np.random.uniform(0, 1, k)

    omega = omega / np.sum(omega)
    x = np.random.multinomial(1, omega, n)
    x = np.argmax(x, 1)

    if M is None:
        M = np.random.uniform(0, 1, [d, k])*(d*k)+20
        M = M / np.sum(M, 0)

    X = np.zeros((n,d))
    for i in range(k):
        X[x==i,:] = np.random.multinomial(c, M[:, i],int(sum(x==i) ))

    return X.astype(float), M, omega, x
