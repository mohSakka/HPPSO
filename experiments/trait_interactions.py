"""Paired factorial contrasts on final error divided by matched initial error."""
import numpy as np


def normalized_response(final, initial):
    final,initial=np.asarray(final,float),np.asarray(initial,float)
    if final.shape!=initial.shape or not np.isfinite(final).all() or not np.isfinite(initial).all():
        raise ValueError("Invalid response arrays")
    if np.any(initial<0) or np.any(final < -1e-10):raise ValueError("Expected nonnegative benchmark errors")
    if not np.allclose(initial,initial[:,:1],rtol=1e-12,atol=1e-12):
        raise ValueError("Factorial paths do not have matched initial errors")
    if np.any((initial==0)&(final>1e-10)):raise ValueError("A best-so-far score worsened from a zero initial error")
    return np.divide(np.maximum(final,0),initial,out=np.zeros_like(final),where=initial>0)


def factorial_contrasts(response):
    """Return seed x 15 contrasts, averaged over all other trait states.

    Subset bit mask 1..15 identifies main, pair, triple and four-way effects.
    For a pair, this is mean_other_states(Y11-Y10-Y01+Y00).
    """
    response=np.asarray(response,float)
    if response.ndim!=2 or response.shape[1]!=16:raise ValueError("Need all 16 factorial arms per seed")
    weights=[]
    for subset in range(1,16):
        order=subset.bit_count()
        weights.append([(-1.)**(order-(mask&subset).bit_count())/(2**(4-order)) for mask in range(16)])
    return response@np.array(weights).T


def bootstrap_ci(values,draws,statistic="median"):
    operation=np.median if statistic=="median" else np.mean
    estimates=np.concatenate([operation(values[batch],axis=1) for batch in np.array_split(draws,max(1,len(draws)//50))])
    return np.quantile(estimates,[.025,.975],axis=0)
