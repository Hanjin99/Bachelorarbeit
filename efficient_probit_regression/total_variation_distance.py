import numpy as np

# total variation distance

def total_variation_distance(p, q, normalize: bool = False, tol: float = 1e-12) -> float:
    """
    Total Variation Distance (TVD) between two discrete probability distributions p and q.

    For discrete distributions:
        TVD(p, q) = 1/2 * sum_i |p_i - q_i|

    Parameters
    ----------
    p, q : array
        Probability vectors (same shape). If normalize=True, they can also be nonnegative
        scores/counts that will be normalized to sum to 1.
    normalize : bool
        If True: automatically normalizes p and q to sum to 1.
        If False: expects p and q to already (approximately) sum to 1.
    tol : float
        Numerical tolerance used for nonnegativity checks and sum checks.

    Returns
    -------
    float
        TVD in [0, 1] for valid probability vectors.
    """
    p = np.asarray(p, dtype=float).ravel()    #makes sure p is a 1D-array
    q = np.asarray(q, dtype=float).ravel()    #makes sure q is a 1D-array

    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p.shape={p.shape}, q.shape={q.shape}")

    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        raise ValueError("p and q must not contain NaN or Inf.")

    # Reject clearly negative entries
    if np.any(p < -tol) or np.any(q < -tol):
        raise ValueError("p and q must be nonnegative.")

    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)

    sp = float(p.sum())
    sq = float(q.sum())

    if normalize:
        if sp <= tol or sq <= tol:
            raise ValueError("Cannot normalize: sum of p or q is ~0.")
        p = p / sp
        q = q / sq
    else:
        if abs(sp - 1.0) > tol or abs(sq - 1.0) > tol:
            raise ValueError(
                "p and q must sum to 1. If they are scores, set normalize=True."
            )

    tvd = 0.5 * float(np.abs(p - q).sum())

    # Keep the result in [0, 1] to avoid tiny numerical overshoots
    return max(0.0, min(1.0, tvd))