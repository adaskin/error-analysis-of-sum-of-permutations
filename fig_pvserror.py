from main import random_prob_vector, matrix_Apq, matrix_Ap, matrix_Aq
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize':12,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
pylab.rcParams.update(params)
nruns = 30
# error, relative error, mean squared error, normalized mse

max_bit_flip_in_oneterm = 8
num_qubits = 8 # number of qubits
matrix_size = 2**num_qubits  # matrix sizes

num_terms = 2 ** num_qubits  # number of permutations in the summation,in the paper k

alphamin = -1
alphamax = 1

one_zero_probs = False

figname = (
    "fig_p_vs_e_A_alpha{}{}_maxbit{}_terms{}".format(
        alphamin, alphamax, max_bit_flip_in_oneterm, num_terms
    )
    + ".png"
)
MSE = {}
NMSE = {}
E = {}
RE = {}

for jkey in np.arange(0.01, 0.95, 0.1):
    MSE[jkey] = np.zeros(nruns)
    NMSE[jkey] = np.zeros(nruns)
    E[jkey] = np.zeros(nruns)
    RE[jkey] = np.zeros(nruns)

    for irun in range(nruns):
        p_probs = random_prob_vector(
            num_terms,
            min_prob=0,
            max_prob=jkey,
            one_zero_probs=one_zero_probs,
            one_zero_rate=jkey,
        )

        A, Aperturbed = matrix_Ap(
            p_probs,
            num_qubits,
            num_terms,
            max_bit_flip_in_oneterm,
            alphamin=alphamin,
            alphamax=alphamax,
        )

        print("eigenvalues=====================================")
        La = np.sort(np.abs(np.linalg.eigvals(A)))
        Lap = np.sort(np.abs(np.linalg.eigvals(Aperturbed)))

        eigen_err = np.abs(La[-1] - Lap[-1])
        relative_eigen_err = eigen_err / np.abs(La[-1])
        print("largest eigenvalue============================")
        print("\t found:{}, actual:{}".format(La[-1], Lap[-1]))
        print("\t absolute error: ", eigen_err)
        print("\t relative error: ", relative_eigen_err)
        mse = np.sum(np.abs(La - Lap) ** 2) / matrix_size
        normalized_mse = mse / np.abs(La[-1])

        print("all eigenvalues=================================")
        print("\t mean squared error:", mse)
        print("\t normalized mse: ", normalized_mse)
        MSE[jkey][irun] = mse
        NMSE[jkey][irun] = normalized_mse
        E[jkey][irun] = eigen_err
        RE[jkey][irun] = relative_eigen_err


# filename = 'Ap' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
# plt.plot(range(nruns),NMSE,RE)
fig, axs = plt.subplots(2, sharex=True, sharey=False)


mean_nmse = {}

for ikey in NMSE:
    data = NMSE[ikey]
    datalen = len(data)
    axs[0].plot([ikey] * datalen, data, ".")
    mean_nmse[ikey] = np.mean(data)

axs[0].plot(mean_nmse.keys(), mean_nmse.values())
axs[0].set_ylabel("Normalized MSE")

mean_re = {}
for ikey in RE:
    data = RE[ikey]
    datalen = len(data)
    axs[1].plot([ikey] * datalen, data, ".")
    mean_re[ikey] = np.mean(data)

axs[1].plot(mean_re.keys(), mean_re.values())
axs[1].set_xlabel("The maximum value of the probability vector $p$")
axs[1].set_ylabel("Relative error")
# axs[1].set_xticks(RE.keys())
plt.show()
fig.savefig(figname)
