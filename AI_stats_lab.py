import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    if a < 0:
        a = 0
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """
    P_A = 0.3
    P_B = 0.7

    likelihood_A = np.exp(-((time - 40) ** 2) / 4)
    likelihood_B = np.exp(-((time - 45) ** 2) / 4)

    num = likelihood_B * P_B
    den = likelihood_A * P_A + num

    return num / den



def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """
    classes = np.random.choice(['A', 'B'], size=n, p=[0.3, 0.7])

    samples = np.where(
        classes == 'A',
        np.random.normal(40, 2, n),   # std = sqrt(4)
        np.random.normal(45, 2, n)
    )

    mask = np.abs(samples - time) < 0.5

    if np.sum(mask) == 0:
        return 0

    selected = classes[mask]

    return np.sum(selected == 'B') / len(selected)
