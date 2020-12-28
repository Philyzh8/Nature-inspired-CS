from __future__ import division
from numpy import *
import random as rd
from scipy.stats import pearsonr
from dl_simulation import *
from analyze_predictions import *
from run_smaf import smaf
import spams
THREADS = 4

def compare_results(A, B):
    results = list(correlations(A, B, 0))[:-1]
    results += list(compare_distances(A, B))
    results += list(compare_distances(A.T, B.T))
    return results


# Calculate the fitness value
def calFitness_DE(X):
    n = len(X)
    fitness = 0
    for i in range(n):
        fitness += X[i] * X[i]
        #fitness += X[i]**2-10*cos(2*pi*X[i])+10
    return fitness


def calFitness(X, UW):
    n = X.shape[1]
    fitness = np.zeros((1, n))
    for i in range(n):
        fitness[0, i] = 1 - pearsonr(X[:, i], UW[:, i])[0]
    return fitness[0]


def calFitness_1(X, UW):

    return 1 - pearsonr(X, UW)[0]


def FireflyAlgorithm(sizepop, Ws_tem, fitness, params):

    population = Ws_tem

    for i in range(sizepop):
        for j in range(sizepop):
            if fitness[j] > fitness[i]:
                r = np.linalg.norm(population[i, :] - population[j, :])
                beta = params[0] * \
                       np.exp(-1 * params[1] * (r ** 2))

                population[i] += beta * (population[j] - population[i])

    return population




def selection(XTemp, XTemp1, fitnessVal, X, U):
    m, n = shape(XTemp)
    fitnessVal1 = zeros(m)
    for i in range(m):
        fitnessVal1[i] = calFitness_1(X, U.dot(XTemp1[i]))
        if (fitnessVal1[i] < fitnessVal[i]):
            for j in range(n):
                XTemp[i, j] = XTemp1[i, j]
            fitnessVal[i] = fitnessVal1[i]
    return XTemp, fitnessVal




def saveBest(fitnessVal, XTemp):
    m = shape(fitnessVal)[0]
    tmp = 0
    for i in range(1, m):
        if (fitnessVal[tmp] > fitnessVal[i]):
            tmp = i
    return fitnessVal[tmp][0], XTemp[tmp]
    #print fitnessVal[tmp][0]

if __name__ == "__main__":

    # SMAF setting
    biased_training = 0.
    composition_noise = 0.
    subset_size = 0
    biased_training = 0.
    composition_noise = 0.
    subset_size = 0
    measurements = 400
    sparsity = 10
    dictionary_size = 0.5
    training_dictionary_fraction = 0.05
    SNR = 2.0

    # data load

    xa = np.load("./Data/GSE102475_xa.npy")
    xb = np.load("./Data/GSE102475_xb.npy")


    itr = 0
    while(itr < 1):

        # Parameters setting
        NP = 40
        maxItr = 50

        # Initialization
        k = min(int(xa.shape[1] * 3), 150)
        Ws = np.zeros((NP, k, xa.shape[1]))

        UW = (np.random.random((xa.shape[0], k)), np.random.random((k, xa.shape[1])))
        UF, WF = smaf(xa, k, 5, 0.0005, maxItr=10, use_chol=True, activity_lower=0., module_lower=xa.shape[0] / 10, UW=UW,
                     donorm=True, mode=1, mink=3.)

        for i in range(NP):
            lda = np.random.randint(5, 20)
            Ws[i] = sparse_decode(xa, UF, lda, worstFit=1 - 0.0005, mink=3.)

        # Calculate the fitness value
        fitnessVal = zeros((NP, xa.shape[1]))
        for i in range(NP):
            fitnessVal[i] = calFitness(xa, UF.dot(Ws[i]))

        gen = 0
        Xnorm = np.linalg.norm(xa) ** 2 / xa.shape[1]

        while gen <= maxItr:

            for i in range(xa.shape[1]):

                Ws_tem = Ws[:, :, i]
                fmin = np.min(fitnessVal[:, i])
                fmin_arg = np.argmin(fitnessVal[:, i])
                best = Ws_tem[fmin_arg, :]

                fa = FireflyAlgorithm(NP, Ws_tem, fitnessVal[:, i], [1.0, 1.0])

                Ws_tem, fitnessVal[:, i] = selection(Ws_tem, fa, fitnessVal[:, i], xa[:, i], UF)

                WF[:, i] = Ws_tem[np.where(fitnessVal[:, i] == min(fitnessVal[:, i]))[0][0], :]
                Ws[:, :, i] = Ws_tem


            UF = spams.lasso(np.asfortranarray(xa.T), D=np.asfortranarray(WF.T),
                             lambda1=0.0005 * Xnorm, mode=1, numThreads=THREADS, cholesky=True, pos=True)
            UF = np.asarray(UF.todense()).T

            print(gen)

            gen += 1

        Results = {}

        x2a, phi, y, w, d, psi = recover_system_knownBasis(xa, measurements, sparsity, Psi=UF, snr=SNR, use_ridge=False)
        Results['FA (training)'] = compare_results(xa, x2a)
        x2b, phi, y, w, d, psi = recover_system_knownBasis(xb, measurements, sparsity, Psi=UF, snr=SNR, use_ridge=False,
                                                           nsr_pool=composition_noise, subset_size=subset_size)
        Results['FA (testing)'] = compare_results(xb, x2b)

        for k, v in sorted(Results.items()):
            print('\t'.join([k] + [str(x) for x in v]))
        itr += 1
