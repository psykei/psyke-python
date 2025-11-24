from collections.abc import Iterable
from itertools import product

import numpy as np
import skfuzzy as skf
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def generate_membership(var, domain, thresholds, shape='tri'):
    th = [var.min()] + [min(max(t, var.min()), var.max()) for t in thresholds] + [var.max()]

    if shape == 'tri':
        mid = [(x1 + x2) / 2 for x1, x2 in zip(th[:-1], th[1:])]
        return [skf.trapmf(domain, [domain.min()] * 2 + mid[:2])] + \
            [skf.trimf(domain, [x1, x2, x3]) for x1, x2, x3 in zip(mid[:-2], mid[1:-1], mid[2:])] + \
            [skf.trapmf(domain, mid[-2:] + [domain.max()] * 2)]
    if shape == 'trap':
        beg = [None, domain.min()] + [(3 * x1 + x2) / 4 for x1, x2 in zip(th[1:-1], th[2:])] + [domain.max()]
        end = [domain.min()] + [(x1 + 3 * x2) / 4 for x1, x2 in zip(th[:-2], th[1:-1])] + [domain.max()]
        return [skf.trapmf(domain, [end[i - 1], beg[i], end[i], beg[i + 1]]) for i in range(1, len(th))]
    raise ValueError('Supported shape values are only \'tri\' and \'trap\'')

def extend_domain(x, q_low=0.05, q_high=0.95, p=0.05, k_sigma=2.0, abs_min_margin=0.0):
    ql, qh = np.quantile(x, [q_low, q_high])
    margin = max(p * (qh - ql), k_sigma * np.std(x), abs_min_margin)
    return np.linspace(ql - margin, qh + margin, 200)

def fuzzify(cuts, data, features, feature_to_idx, shape='tri'):
    cuts = dict(zip(features, cuts))
    domains = {c: extend_domain(data[c]) for c in features}
    return {c: (generate_membership(data[c], domains[c], cuts[c], shape), (min(domains[c]), max(domains[c])),
                feature_to_idx[c]) for c in features}

def fuzzy_labels(n):
    if n < 1 or n > 9:
        raise ValueError('n must be between 1 and 9')
    if n == 1:
        return ["Medium"]
    if n == 2:
        return ["Low", "High"]

    full_scale = ["Extremely Low", "Very Low", "Low", "Slightly Low", "Medium",
                  "Slightly High", "High", "Very High", "Extremely High"]
    indices = np.round(np.linspace(0, len(full_scale) - 1, n)).astype(int)

    selected = []
    for i in indices:
        if full_scale[i] not in selected:
            selected.append(full_scale[i])

    return selected

def get_activations(x, functions_domains, valid):
    levels = [np.array([skf.interp_membership(np.linspace(domain[0], domain[1], 200), mf, x[index]) for mf in mfs])
              for mfs, domain, index in functions_domains.values()]
    return np.prod(np.meshgrid(*levels, indexing='ij'), axis=0).ravel()[valid]

def crisp_or_equation(lr: float | str | LinearRegression, features=Iterable[str], decimals: int = 2) -> str | float:
    if isinstance(lr, LinearRegression):
        terms = ''.join([f"{' + ' if c >= 0 else ' - '}{abs(c):.{decimals}f} {f}" for c, f in zip(lr.coef_, features)])
        return f"{lr.intercept_:.{decimals}f}{terms}"
    return lr

def generate_fuzzy_rules(variables: dict[str, Iterable[str]], outputs: Iterable[str | float | LinearRegression],
                         features: list[str], valid: Iterable[bool]) -> list[str]:
    outputs = [crisp_or_equation(output, features) for output in outputs]
    return [f'{features[-1]} = {output} if {" and ".join(f"{v} is {lab}" for v, lab in zip(variables.keys(), combo))}'
            for combo, output in zip(np.array(list(product(*list(variables.values()))))[valid], outputs)]

def plot_membership(functions_domains):
    fig, ax = plt.subplots(nrows=len(functions_domains), figsize=(6, len(functions_domains) * 3))

    for i, (k, v) in enumerate(functions_domains.items()):
        for s, l in zip(v[0], v[2]):
            ax[i].plot(np.linspace(v[1][0], v[1][1], 200), s, linewidth=1.5, label=l)
        ax[i].set_title(k)
        ax[i].set_xlim(v[1][0], v[1][1])
        ax[i].legend()

    plt.tight_layout()
    plt.show()
