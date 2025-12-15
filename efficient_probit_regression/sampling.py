import numpy as np
from numpy import ndarray
from scipy.stats import expon
from scipy.sparse import diags

# random number generator
_rng = np.random.default_rng()

def _check_sample(X, y, sample_size):
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Incompatible shapes of X and y: {X.shape[0]} != {y.shape[0]}"
        )

    if sample_size > X.shape[0]:
        raise ValueError("Sample size can't be greater than total number of samples!")

    if sample_size <= 0:
        raise ValueError("Sample size must be greater than zero!")


def uniform_sampling(X: np.ndarray, y: np.ndarray, sample_size: int):
    """
    Draw a uniform sample of X and y without replacement.

    :param X: A matrix of size (n, d).
    :param y: A vector of size (n,).
    :param sample_size: The size of the sample that should be drawn.

    :return: The reduced sample (X_reduced, y_reduced).
    """
    _check_sample(X, y, sample_size)

    sample_indices = _rng.choice(X.shape[0], size=sample_size, replace=False)

    return X[sample_indices], y[sample_indices]


def fast_QR(X, p=2.0):
    """
    Returns Q of a fast QR decomposition of X.
    """
    n, d = X.shape

    if p <= 2:    
        sketch_size = d ** 2
    else:
        sketch_size = np.maximum(d ** 2, int(np.power(n, 1 - 2 / p)))

    f = np.random.randint(sketch_size, size=n)
    g = np.random.randint(2, size=n) * 2 - 1
    if p != 2:
        lamb = expon.rvs(size=n)

    # init the sketch
    X_sketch = np.zeros((sketch_size, d))
    if p == 2:       
        for i in range(n):
            X_sketch[f[i]] += g[i] * X[i]
    else:    
        for i in range(n):
            X_sketch[f[i]] += g[i] / np.power(lamb[i], 1 / p) * X[i]  # exponential-verteile Zufallsvariable

    R = np.linalg.qr(X_sketch, mode="r")
    R_inv = np.linalg.inv(R)

    if p == 2:   # hier wird es noch verschnellert
        k = 20
        g = np.random.normal(loc=0, scale=1 / np.sqrt(k), size=(R_inv.shape[1], k))
        r = np.dot(R_inv, g)
        Q = np.dot(X, r)
    else:  # normalfall, der immer richtig ist
        Q = np.dot(X, R_inv)

    return Q


def _round_up(x: np.ndarray) -> np.ndarray:
    """
    Rounds each element in x up to the nearest power of two.
    """
    if not np.all(x >= 0):
        raise ValueError("All elements of x must be greater than zero!")

    greater_zero = x > 0

    results = x.copy()
    results[greater_zero] = np.power(2, np.ceil(np.log2(x[greater_zero])))

    return results


def to_density(X: np.ndarray):
    """
    Turns scores into a density.
    """
    return X / np.sum(X)


def to_density_X_Y(X: ndarray, Y: ndarray):
    """
    Turns the combination of two scores into a density.
    """
    return (X + Y) / (X.sum() + Y.sum())


def logit_sampling(X: np.ndarray, y: np.ndarray, sample_size: int):
    """
    Logit sampling from 2018 Paper On Coresets for Logistic Regression.

    Returns X_reduced, y_reduced, weights
    """

    # 1. Obtain fast QR approximation
    n, d = X.shape

    sketch_size = d ** 2

    f = np.random.randint(sketch_size, size=n)
    g = np.random.randint(2, size=n) * 2 - 1

    X_sketch = np.zeros((sketch_size, d))
    for i in range(n):
        X_sketch[f[i]] += g[i] * X[i]

    R = np.linalg.qr(X_sketch, mode="r")
    R_inv = np.linalg.inv(R)

    k = 20
    g = np.random.normal(loc=0, scale=1 / np.sqrt(k), size=(R_inv.shape[1], k))
    r = np.dot(R_inv, g)
    Q = np.dot(X, r)

    # 2. Obtain square roots of leverage scores and add 1/n term
    scores = np.linalg.norm(Q, axis=1) + 1 / n

    # 3. draw a random sample
    p = scores / np.sum(scores)

    w = 1 / (p * sample_size)

    sample_indices = _rng.choice(
        X.shape[0],
        size=sample_size,
        replace=False,
        p=p,
    )

    return X[sample_indices], y[sample_indices], w[sample_indices]


# leverage scores


def compute_leverage_scores(X: np.ndarray, p=2, fast_approx=False):  
    """
        Computes leverage scores.
    """
    if not len(X.shape) == 2:
        raise ValueError("X must be 2D!")

    if not fast_approx: # boolean, schnellere oder nicht die schnellere Q-R-Zerlegung
        Q, *_ = np.linalg.qr(X)
    else:
        Q = fast_QR(X, p=p)  # ein Möglichkeit wie man eine Zerlegung schneller machen kann

    leverage_scores = np.power(np.linalg.norm(Q, axis=1, ord=p), p)

    return leverage_scores


def leverage_score_sampling(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    augmented: bool = False,
    round_up: bool = False,
    precomputed_scores: np.ndarray = None,
    p: float = 2,
    fast_approx: bool = False,
):
    """
    Draw a leverage score weighted sample of X and y without replacement.

    :param X: Data matrix.
    :param y: Label vector.
    :param sample_size: Sample size.
    :param augmented: Whether to add the additive 1/W term,
        where W is the sum of all weights.
    :param round_up: Round the leverage scores up to the nearest power of two.
    :param precomputed_scores: To avoid recomputing the leverage scores every time,
        pass the precomputed scores here.
    :param p: The order of the p-generalized probit model.
    :param fast_approx: Whether to use the fast leverage score approximation algorithm.

    :return:
        :X_reduced: The reduced data matrix.
        :y_reduced: The reduced label vector.
        :w: The corresponding sample weights.
    """
    _check_sample(X, y, sample_size)

    if precomputed_scores is None:
        leverage_scores = compute_leverage_scores(X, p=p, fast_approx=fast_approx)
    else:
        leverage_scores = precomputed_scores

    if augmented:
        leverage_scores = leverage_scores + 1 / X.shape[0]    

    if round_up:
        leverage_scores = _round_up(leverage_scores)

    p = leverage_scores / np.sum(leverage_scores)

    w = 1 / (p * sample_size)  
    
    _rng = np.random.default_rng()    
    sample_indices = _rng.choice(   # _rng (see np.random)
        X.shape[0],
        size=sample_size,
        replace= False,       # replace = False (without replacement)
        p=p,
    )

    return X[sample_indices], y[sample_indices], w[sample_indices]


# lewis weights

# berechnet l_2 leverage score
def calculate_lev_score_exact(X):
    Xt = X.T
    XXinv = np.linalg.pinv(Xt.dot(X))
    lev = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        xi = X[i : i + 1, :]
        val = (xi.dot(XXinv)).dot(xi.T)    # typically (1, 1)
        lev[i] = np.asarray(val).item()    # convert 1x1 to Python scalar
    return lev


def calculate_lewis_weights_exact(X, p=1.0, T=20):
    n = X.shape[0]
    w = np.ones(n)

    for i in range(T):
        Wp = diags(np.power(w, 0.5 - 1.0 / p))
        # Q = qr(Wp.dot(X))
        # s = _calculate_sensitivities_leverage(Q)
        s = calculate_lev_score_exact(Wp.dot(X))
        w_nxt = np.power(w, 1.0 - p / 2.0) * np.power(s, p / 2.0)
        # print("|w_t - w_t+1|/|w_t| = ", np.linalg.norm(w - w_nxt) / np.linalg.norm(w))
        w = w_nxt

    # return np.array(w + 1.0 / n, dtype=float) #TODO
    return np.array(w, dtype=float)


def calculate_lewis_weights_fast(X, p=1.0, T=20):
    n = X.shape[0]
    w = np.ones(n)

    for i in range(T):
        # assert min(w) > 0, str(min(w))
        Wp = diags(np.power(w, 0.5 - 1.0 / p))

        Q = fast_QR(Wp.dot(X), p)
        s = np.power(np.linalg.norm(Q, axis=1, ord=2), 2)
        w_nxt = np.power(w, 1.0 - p / 2.0) * np.power(s, p / 2.0)
        # print("|w_t - w_t+1|/|w_t| = ", npl.norm(w - w_nxt) / npl.norm(w))
        w = w_nxt

    #return np.array(w + 1.0 / n, dtype=float)
    return np.array(w, dtype=float)


def lewis_sampling(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    p: float = 1.0,
    precomputed_weights=None,
    fast_approx=False,
):
    """
    Returns X_reduced, y_reduced, probabilities
    """
    if precomputed_weights is None:
        if fast_approx:
            s = calculate_lewis_weights_fast(X, p=p)
        else:
            s = calculate_lewis_weights_exact(X, p=p)
    else:
        s = precomputed_weights

    # calculate probabilities
    p = s / np.sum(s)

    # draw the sample
    rng = np.random.default_rng()
    sample_indices = rng.choice(X.shape[0], size=sample_size, replace=False, p=p)

    return X[sample_indices], y[sample_indices], p[sample_indices]


# Kombination von l_2 und l_p leverage scores

def calculate_l2_lp_leverage_score(X: np.ndarray, p=2, fast_approx=False):
    """
    Returns the sum of the l_2 and l_p leverage scores (total_score), and it's density (p).
    """
    l_2_leverage_score = calculate_lev_score_exact(X)
    l_p_leverage_score = compute_leverage_scores(X, p=p, fast_approx=fast_approx)
    p = to_density_X_Y(l_2_leverage_score, l_p_leverage_score)
    total_score = l_2_leverage_score + l_p_leverage_score

    return total_score, p


# random evaluations


def compute_random_evaluations_probabilities(X, m=50, p=2.0, x_dist="normal", rng=None, eps=1e-12):
    """
    Compute sampling probabilities via random evaluations:

        p_i = (1/m) * sum_{j=1..m}  |a_i^T x_j|^p / ||A x_j||_p^p

    where A = X (n x d), a_i is the i-th row of X, and x_j are random vectors in R^d.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Data matrix (A).
    m : int
        Number of random evaluation vectors x_j.
    p : float
        The p in |.|^p and ||.||_p^p.
    x_dist : {"normal", "rademacher"}
        Distribution for x_j:
        - "normal": iid N(0,1)
        - "rademacher": iid +/- 1 with prob 1/2
    rng : None, int, or np.random.Generator
        Random generator (or seed). If None, uses default generator.
    eps : float
        Small value to guard against division by zero.

    Returns
    -------
    probs : np.ndarray, shape (n,)
        Sampling probabilities, nonnegative and summing to 1.
    """
    X = np.asarray(X)
    if not len(X.shape) == 2:
        raise ValueError("X must be 2D!")
    n, d = X.shape
    if m <= 0 or not isinstance(m, int):
        raise ValueError("m must be a positive integer.")
    if p <= 0:
        raise ValueError("p must be > 0.")

    # RNG setup
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    # Draw random evaluation vectors (d x m)
    if x_dist == "normal":
        R = gen.standard_normal(size=(d, m))
    elif x_dist == "rademacher":
        R = gen.integers(0, 2, size=(d, m)) * 2 - 1  # +/-1
    else:
        raise ValueError("x_dist must be 'normal' or 'rademacher'.")

    # Compute A x_j for all j at once: (n x m)
    Y = X @ R

    # Elementwise |.|^p: (n x m)
    Z = np.abs(Y) ** p

    # Denominators: ||A x_j||_p^p = sum_i |a_i^T x_j|^p, shape (m,)
    den = Z.sum(axis=0)

    # Guard against rare degenerate cases where den == 0
    den = np.maximum(den, eps)

    # For each j, contribution is Z[:, j] / den[j]; average over j
    probs = (Z / den).mean(axis=1)

    # Numerical cleanup: enforce nonnegativity & sum to 1
    probs = np.maximum(probs, 0.0)
    s = probs.sum()
    if s <= 0:
        # Fallback (should be extremely rare): uniform
        return np.full(n, 1.0 / n)
    return probs / s


def random_evaluation_sampling(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    m: int = 50,
    p: float = 2.0,
    x_dist: str = "normal",
    rng: np.random.Generator | None = None,
):
    """
    Ziehe ein Sample ohne Zurücklegen anhand von Wahrscheinlichkeiten aus
    compute_random_evaluation_probabilities(...).

    :return: (X_reduced, y_reduced, w, p_selected)
             w sind Horvitz-Thompson-ähnliche Gewichte: w_i = 1 / (p_i * sample_size)
    """
    _check_sample(X, y, sample_size)

    rng = _rng if rng is None else rng
    prob = compute_random_evaluations_probabilities(X, m=m, p=p, rng=rng, x_dist=x_dist)

    w = 1.0 / (prob * sample_size)

    sample_indices = rng.choice(
        X.shape[0],
        size=sample_size,
        replace=False,
        p=prob,
    )

    return X[sample_indices], y[sample_indices], w[sample_indices], prob[sample_indices]