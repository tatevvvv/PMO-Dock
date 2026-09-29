"""
Shared reward helpers for hit/lead tasks.
"""
import numpy as np

def gaussian_modifier(x, mu, sigma):
    return np.exp(-0.5 * np.power((x - mu) / sigma, 2.0))

def select_sigma(prop_name: str):
    """Shared sigma for reward scaling; supports both root and genetic_chemalactica/saturn key names."""
    coef = 0.1
    prop_name2sigma = {
        "qed_score": coef * 1,
        "sa_score": coef * 7,
        "QED": coef * 1,
        "SAS": coef * 7,
        "WEIGHT": coef * 1000,
        "RINGCOUNT": coef * 5,
        "NUMAROMATICRINGS": coef * 5
    }
    if prop_name in prop_name2sigma:
        return prop_name2sigma[prop_name]
    if "docking_score" in prop_name or "DOCKING" in prop_name:
        return coef * 20
    if "SIMILAR" in prop_name:
        return coef * 1
    return None

def hit_reward(measured, sigmas, hit_ranges, prod=True, avg=False):
    rewards = []
    for m, rnge, sigma in zip(measured, hit_ranges, sigmas, strict=True):
        if m is None:
            rewards.append(0)
        elif rnge[0] <= m <= rnge[1]:
            rewards.append(1)
        else:
            dist = min(abs(m - rnge[0]), abs(m - rnge[1]))
            reward = gaussian_modifier(dist, mu=0, sigma=sigma)
            rewards.append(reward)

    if prod:
        return np.array(rewards).prod().item()
    elif avg:
        return np.array(rewards).mean().item()
    return np.array(rewards)

def compute_geam_reward(measured):
    """GEAM paper formula: (docking/20) * qed * (10-sa)/9."""
    docking_scores = measured[0]
    sa_scores = measured[1]
    qed_scores = measured[2]
    trans_sa_scores = (10 - sa_scores) / 9
    aggregated_scores = (np.clip(docking_scores, 0, 20) / 20) * qed_scores * trans_sa_scores
    return aggregated_scores


def hit_docking_score_reward(measured, sigmas, hit_ranges):
    """Lead-style: (dock/20) * hit_reward(remaining props with ranges)."""
    if measured[0] is None or any(m is None for m in measured[1:]):
        # Allow None only in docking? No — any missing prop → 0 reward.
        if measured[0] is None:
            return 0.0
        # None in constraint props → hit_reward already maps None→0
    docking_score = measured[0]
    computed = hit_reward(measured[1:], sigmas[1:], hit_ranges[1:])
    return float(docking_score) / 20.0 * float(computed)


def hit_spec_reward(measured, sigmas, hit_ranges):
    """Spec-style: gap(target, antitarget) * hit_reward(all four props)."""
    assert len(measured) == 4
    if any(m is None for m in measured):
        return 0.0
    target_docking_score = measured[0]
    antitarget_docking_score = measured[1]
    gap = np.clip(
        float(target_docking_score) / 20.0 - float(antitarget_docking_score) / 20.0,
        0.0,
        1.0,
    )
    computed = hit_reward(measured, sigmas, hit_ranges)
    return float(gap) * float(computed)
