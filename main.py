import permutations as pm
import numpy as np
import random


def random_qubit_indices(
    num_qubits, max_group_size=1, min_group_size=1, fix_group_size=None
):
    """
    generates random qubit indices
    [rand_qubit_indices]
    """
    if fix_group_size == None:
        group_size = random.randint(min_group_size, max_group_size)
    elif fix_group_size >= 0:
        group_size = fix_group_size
    else:
        return

    return random.sample(range(num_qubits), group_size)


def random_prob_vector(
    size, min_prob=0, max_prob=1, one_zero_probs=False, one_zero_rate=0.5
):
    """
    generates random vector of probabilities..
    in 0/1 probs if generated prob < one_zero_rate, it is  set 1.
    """
    pvec = np.zeros(size)
    for i in range(size):
        rand_prob = random.uniform(min_prob, max_prob)

        # one zero probabilities
        if one_zero_probs == True:
            if rand_prob < one_zero_rate:
                rand_prob = 1
            else:
                rand_prob = 0

        pvec[i] = rand_prob

    return pvec




def unitary_for_X_on_qubits(qubits, num_qubits):
    """
    returns unitary matrix that applies X gate to qubits
    qubits can be an int or a list
    """
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    U = np.array([[1]])
    for j in range(num_qubits):
        if type(qubits) == int:
            if j == qubits:
                U = np.kron(U, X)
            else:
                U = np.kron(U, I)
        else:
            if j in qubits:
                U = np.kron(U, X)
            else:
                U = np.kron(U, I)
    return U


def unitary_for_Z_on_qubits(qubits, num_qubits):
    """
    returns unitary matrix that applies Z gate to qubits
    qubits can be an int or a list
    """
    I = np.eye(2)
    Z = np.eye(2)
    Z[1][1] = -1

    U = np.array([[1]])
    for j in range(num_qubits):
        if type(qubits) == int:
            if j == qubits:
                U = np.kron(U, Z)
            else:
                U = np.kron(U, I)
        else:
            if j in qubits:
                U = np.kron(U, Z)
            else:
                U = np.kron(U, I)
    return U


def matrix_Apq(p, q, num_qubits, num_terms, max_bit_flip_in_oneterm=1,alphamin=0, alphamax=1):
    """
    generates matrices
        A, and Apq, which applies phase and bit flips to permutations
    """

    matrix_size = 2**num_qubits
    A = np.zeros(matrix_size)
    Aperturbed = np.zeros(matrix_size)

    alphas = np.zeros(num_terms, dtype=float)

    bitflip_qubits = [0] * num_terms
    phaseflip_qubits = [0] * num_terms

    In = np.eye(matrix_size)

    for i in range(num_terms):
        bitflip_qubits[i] = random_qubit_indices(
            num_qubits, max_group_size=max_bit_flip_in_oneterm
        )
        phaseflip_qubits[i] = random_qubit_indices(
            num_qubits, max_group_size=max_bit_flip_in_oneterm
        )
        print("bit flip qubits:", bitflip_qubits[i])
        print("phase flip qubits:", phaseflip_qubits[i])
        # random coefficients
        Pi, prow = pm.rand_perm_matrix(matrix_size)
        alphas[i] = random.uniform(alphamin, alphamax)

        A = A + alphas[i] * Pi
        Ux = unitary_for_X_on_qubits(bitflip_qubits[i], num_qubits)
        Uz = unitary_for_Z_on_qubits(phaseflip_qubits[i], num_qubits)
        # bit flip and phase flip models
        pX = (1 - p[i]) * In + p[i] * Ux
        qZ = (1 - q[i]) * In + q[i] * Uz
        Aperturbed = Aperturbed + alphas[i] * ((qZ @ pX) @ Pi)
    return A, Aperturbed


def matrix_Ap(p, num_qubits, num_terms, max_bit_flip_in_oneterm=1,alphamin=0, alphamax=1):
    """
    generates matrices
        A, Ap which applies bitflips to the permutations
    """

    matrix_size = 2**num_qubits
    A = np.zeros(matrix_size)
    Aperturbed = np.zeros(matrix_size)

    alphas = np.zeros(num_terms, dtype=float)

    bitflip_qubits = [0] * num_terms

    In = np.eye(matrix_size)

    for i in range(num_terms):
        bitflip_qubits[i] = random_qubit_indices(
            num_qubits, max_group_size=max_bit_flip_in_oneterm
        )
        print("bit flip qubits:", bitflip_qubits[i])
        # random coefficients
        Pi, prow = pm.rand_perm_matrix(matrix_size)
        alphas[i] = random.uniform(alphamin, alphamax)

        A = A + alphas[i] * Pi
        Ux = unitary_for_X_on_qubits(bitflip_qubits[i], num_qubits)
        # bit flip and phase flip models
        pX = (1 - p[i]) * In + p[i] * Ux
        Aperturbed = Aperturbed + alphas[i] * (pX @ Pi)
    return A, Aperturbed


def matrix_Aq(q, num_qubits, num_terms, max_bit_flip_in_oneterm=1 , alphamin=0, alphamax=1):
    """
    generates matrices
        A and Aq-only phase flips applied to permutations
    """

    matrix_size = 2**num_qubits
    A = np.zeros(matrix_size)
    Aperturbed = np.zeros(matrix_size)

    alphas = np.zeros(num_terms, dtype=float)

    phaseflip_qubits = [0] * num_terms

    In = np.eye(matrix_size)

    for i in range(num_terms):
        phaseflip_qubits[i] = random_qubit_indices(
            num_qubits, max_group_size=max_bit_flip_in_oneterm
        )
        print("phase flip qubits:", phaseflip_qubits[i])
        # random coefficients
        Pi, prow = pm.rand_perm_matrix(matrix_size)
        alphas[i] = random.uniform(alphamin, alphamax)

        A = A + alphas[i] * Pi
        Uz = unitary_for_Z_on_qubits(phaseflip_qubits[i], num_qubits)
        # phase flip models
        qZ = (1 - q[i]) * In + q[i] * Uz
        Aperturbed = Aperturbed + alphas[i] * (qZ @ Pi)
    return A, Aperturbed


if __name__ == "__main__":
    num_qubits = 4  # number of qubits
    matrix_size = 2**num_qubits  # matrix sizes

    num_terms = 2 * num_qubits  # number of permutations in the summation,in the paper k

    p_probs = random_prob_vector(
        num_terms, min_prob=0, max_prob=0.1, one_zero_probs=False, one_zero_rate=0.15
    )
    q_probs = random_prob_vector(
        num_terms, min_prob=0, max_prob=0.1, one_zero_probs=False, one_zero_rate=0.15
    )

    A, Aperturbed = matrix_Apq(p_probs, q_probs, num_qubits, num_terms,alphamin=0, alphamax=1)

    print("eigenvalues=====================================")
    La = np.sort(np.abs(np.linalg.eigvals(A)))
    Lap = np.sort(np.abs(np.linalg.eigvals(Aperturbed)))

    eigen_err = np.abs(La[-1] - Lap[-1])
    relative_eigen_err = eigen_err / np.abs(La[-1])
    print("largest eigenvalue============================")
    print("\t found:{}, actual:{}".format(La[-1], Lap[-1]))
    print("\t absolute error: ", eigen_err)
    print("\t relative error: ", relative_eigen_err)
    MSE = np.sum(np.abs(La - Lap) ** 2) / matrix_size
    normalized_MSE = MSE / np.abs(La[-1])

    print("all eigenvalues=================================")
    print("\t mean squared error:", MSE)
    print("\t normalized MSE: ", normalized_MSE)
