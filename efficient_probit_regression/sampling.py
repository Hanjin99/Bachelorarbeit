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
        lev[i] = (xi.dot(XXinv)).dot(xi.T)
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


def compute_random_evaluation_probabilities(
    X: np.ndarray,
    m: int = 50,
    p: float = 2.0,
    eps: float = 1e-12,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Approximiert Sampling-Wahrscheinlichkeiten über zufällige Auswertungen:

        p_i = (1/m) * sum_{j=1..m}  |(A x_j)_i|^p / ||A x_j||_p^p

    wobei x_j zufällige Vektoren sind (hier: Standardnormalverteilung).

    :param X: Datenmatrix A der Form (n, d)
    :param m: Anzahl zufälliger Auswertungen (Random Points)
    :param p: p-Norm-Exponent (p > 0)
    :param eps: numerische Stabilisierung, falls ||A x||_p^p ~ 0
    :param rng: optionaler RNG für Reproduzierbarkeit

    :return: Wahrscheinlichkeitsvektor der Länge n (Summe = 1)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D!")
    if m <= 0:
        raise ValueError("m must be greater than zero!")
    if p <= 0:
        raise ValueError("p must be greater than zero!")

    n, d = X.shape
    rng = _rng if rng is None else rng

    probs = np.zeros(n, dtype=float)

    for _ in range(m):
        x = rng.normal(loc=0.0, scale=1.0, size=d)     # x_j in R^d
        Ax = X @ x                                     # (n,)
        num = np.abs(Ax) ** p                           # |(Ax)_i|^p
        den = float(np.sum(num))                        # ||Ax||_p^p

        if not np.isfinite(den) or den <= eps:
            # Degenerierter Fall: trägt nichts Sinnvolles bei
            continue

        probs += num / den

    probs /= m

    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        # Fallback: uniform, falls alles degeneriert war
        return np.full(n, 1.0 / n, dtype=float)

    return probs / s


def random_evaluation_sampling(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    m: int = 50,
    p: float = 2.0,
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
    prob = compute_random_evaluation_probabilities(X, m=m, p=p, rng=rng)

    w = 1.0 / (prob * sample_size)

    sample_indices = rng.choice(
        X.shape[0],
        size=sample_size,
        replace=False,
        p=prob,
    )

    return X[sample_indices], y[sample_indices], w[sample_indices], prob[sample_indices]