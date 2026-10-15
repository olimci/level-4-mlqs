import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

rng = np.random.default_rng(42)

def meas_ops(pauli):
    return [
        (I + pauli)/2,
        (I - pauli)/2,
    ]

paulis = [X, Y, Z]

ops = [
    [np.kron(A, B)
    for A in meas_ops(P1)
    for B in meas_ops(P2)]
    for P1 in paulis
    for P2 in paulis
]

def probabilities(rho, operators):
    return np.array([
        np.trace(M @ rho).real
        for M in operators
    ])

def measure(rho, operators, shots=1000):
    probs = probabilities(rho, operators)

    return rng.multinomial(shots, probs)

def neg_log_likelyhood(rho, ops, counts):
    loss = 0.0

    for ax_ops, obs in zip(ops, counts):
        p = probabilities(rho, ax_ops)

        if np.any((p <= 0) & (obs > 0)):
            return np.inf

        loss -= np.sum(
            obs[obs > 0] * np.log(p[obs > 0])
        )
    return loss

def T_from_params(params):
    return np.array([
        [params[0], 0.0, 0.0, 0.0],
        [params[4] + 1.0j*params[5], params[1], 0.0, 0.0],
        [params[6] + 1.0j*params[7], params[8] + 1.0j*params[9], params[2], 0.0],
        [params[10] + 1.0j*params[11], params[12] + 1.0j*params[13], params[14] + 1.0j*params[15], params[3]],
    ])

def rho_from_T(T):
    herm = T.T.conj() @ T
    return herm / np.trace(herm)

def rho_from_params(params):
    return rho_from_T(T_from_params(params))

def do_mle(rho_true):
    counts = [measure(rho_true, axops) for axops in ops]

    params = np.ones(16)

    min_fn = lambda params: neg_log_likelyhood(rho_from_params(params), ops, counts)

    result = minimize(
        min_fn,
        x0=params,
        jac="3-point"
    )

    rho_meas = rho_from_params(result.x)

    return rho_meas


def main() -> None:
    # The bell state:  (|00> + |11>)/sqrt(2)
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_bell = np.outer(bell, bell.conj())

    noise_levels = np.linspace(0.0, 1.0, 500)
    negativities = []
    chsh_values = []

    for p in tqdm(noise_levels):
        rho_mixed = (1 - p) * rho_bell + p * np.eye(4) / 4
        rho_meas = do_mle(rho_mixed)

        rho_pt = rho_meas.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
        negativities.append(max(0.0, -np.linalg.eigvalsh(rho_pt)[0]))

        correlations = np.array([
            [np.trace(rho_meas @ np.kron(a, b)).real for b in paulis]
            for a in paulis
        ])
        eigenvalues = np.linalg.eigvalsh(correlations.T @ correlations)
        chsh_values.append(2 * np.sqrt(eigenvalues[-1] + eigenvalues[-2]))

    _, (ax_neg, ax_chsh) = plt.subplots(2, 1, sharex=True, layout="constrained")

    ax_neg.scatter(noise_levels, negativities, s=5, label="MLE")
    ax_neg.plot(noise_levels, np.maximum(0, (2 - 3 * noise_levels) / 4), label="True state")
    ax_neg.axvline(2 / 3, color="gray", linestyle="--", label="Entanglement threshold")
    ax_neg.set_ylabel("Negativity")
    ax_neg.legend()

    ax_chsh.scatter(noise_levels, chsh_values, s=5, label="MLE")
    ax_chsh.plot(noise_levels, 2 * np.sqrt(2) * (1 - noise_levels), label="True state")
    ax_chsh.axhline(2, color="gray", linestyle="--", label="CHSH bound")
    ax_chsh.axvline(1 - 1 / np.sqrt(2), color="gray", linestyle=":", label="Violation threshold")
    ax_chsh.set_xlabel("White-noise fraction p")
    ax_chsh.set_ylabel("Maximum CHSH value")
    ax_chsh.legend()

    plt.show()
