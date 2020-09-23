import numpy as np
from sklearn.preprocessing import normalize


def replace_nan(data, a=0):
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = a
    return data


def do_powerlaw(data, powerlaw=1.0):
    data = np.sign(data) * np.power(np.abs(data), powerlaw)
    return data


def vecpostproc(data, powerlaw=1.0):
    data = do_powerlaw(data, powerlaw)
    data = normalize(data, norm='l2')
    data = replace_nan(data)
    return data


def xform_data(data,
               model,
               *,
               xform='pca',
               reduce_dims=128,
               l2normalize=True,
               keval=40,
               t=0.7):
    mn = model["mean"]
    evecs = model["eigvecs"]
    evals = model["eigvals"]

    data = data - mn

    evecs = evecs[:, :min(reduce_dims, data.shape[1])]
    evals = evals[:min(reduce_dims, data.shape[1])]

    pval = 1.0
    if xform == 'pca':
        data = data @ evecs
        pval = 0.5
    elif xform == 'whiten':
        evecs = evecs @ np.diag(np.power(evals, -0.5))
        data = data @ evecs
    elif xform == 'lw':
        data = data @ evecs
    elif xform == 'pcaws':
        alpha = evals[keval]
        evals = ((1 - alpha) * evals) + alpha
        evecs = evecs @ np.diag(np.power(evals, -0.5))
        data = data @ evecs
    elif xform == 'pcawt':
        m = -0.5 * t
        evecs = evecs @ np.diag(np.power(evals, m))
        data = data @ evecs
    else:
        raise KeyError('Unknown transform - %s' % xform)

    if l2normalize:
        data = vecpostproc(data, pval)

    return data
