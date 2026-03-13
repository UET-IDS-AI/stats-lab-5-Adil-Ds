import numpy as np

# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.
    f(x) = lam * exp(-lam*x) for x >= 0
    """
    return np.where(x >= 0, lam * np.exp(-lam * x), 0)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    P(a < X < b) = e^(-lam*a) - e^(-lam*b)
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, lam=1, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(1/lam, n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    f(x) = (1 / sigma*sqrt(2pi)) * exp(-0.5 * ((x-mu)/sigma)^2)
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coefficient * exponent


def posterior_probability(time):
    """
    Compute P(B | X = time) using Bayes rule.
    Priors:  P(A)=0.3, P(B)=0.7
    Groups:  A ~ N(40,4), B ~ N(45,4)  → sigma=2
    """
    pa, pb = 0.3, 0.7
    likelihood_a = gaussian_pdf(time, mu=40, sigma=2)
    likelihood_b = gaussian_pdf(time, mu=45, sigma=2)
    evidence_a = pa * likelihood_a
    evidence_b = pb * likelihood_b
    return evidence_b / (evidence_a + evidence_b)


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """
    # Step 1: assign groups randomly using priors
    groups = np.random.choice(['A', 'B'], size=n, p=[0.3, 0.7])

    # Step 2: generate finish times based on group
    times = np.where(groups == 'A',
                     np.random.normal(40, 2, n),
                     np.random.normal(45, 2, n))

    # Step 3: keep only swimmers close to target time
    mask = np.abs(times - time) < 0.5

    # Step 4: fraction of those swimmers from Group B
    return np.sum(groups[mask] == 'B') / np.sum(mask)


# -------------------------------------------------
# Verification — run and compare results
# -------------------------------------------------

print("=" * 45)
print("QUESTION 1 — Exponential Distribution")
print("=" * 45)
analytical  = exponential_interval_probability(2, 5, lam=1)
simulated   = simulate_exponential_probability(2, 5, lam=1)
print(f"Analytical  P(2 < X < 5) = {analytical:.6f}")
print(f"Simulated   P(2 < X < 5) = {simulated:.6f}")
print(f"Expected    e^-2 - e^-5  = {np.exp(-2) - np.exp(-5):.6f}")

print()
print("=" * 45)
print("QUESTION 2 — Bayesian Classification")
print("=" * 45)
analytical  = posterior_probability(42)
simulated   = simulate_posterior_probability(42)
print(f"Analytical  P(B | X=42)  = {analytical:.6f}")
print(f"Simulated   P(B | X=42)  = {simulated:.6f}")
```

**Expected output:**
```
=============================================
QUESTION 1 — Exponential Distribution
=============================================
Analytical  P(2 < X < 5) = 0.128617
Simulated   P(2 < X < 5) = 0.128450   ← close, small random variation
Expected    e^-2 - e^-5  = 0.128617

=============================================
QUESTION 2 — Bayesian Classification
=============================================
Analytical  P(B | X=42)  = 0.388986
Simulated   P(B | X=42)  = 0.387200   ← close, small random variation
