"""Computation functions for 'Multivariate Time Series Sub-Sequence CLustering Metric'"""

# Author: Jonas Köhne <jokohonas@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd

# from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.preprocessing import LabelEncoder


def check_data_constant_values(X: np.array):
    """Checking for only constant values in the data and raising ValueError

    Args:
        X (np.array): multivariate time series data provided

    Raises:
        ValueError: "Some or all dimensions of X have only constant values over time! No clustering possible! Remove constant columns"

    Returns:
        _type_: np.array
    """
    if ((X.min(axis=0) - X.max(axis=0)) == 0).any():
        raise ValueError("Some or all dimensions of X have only constant values over time! No clustering possible! Remove constant columns")
    return X


def divide(*args, **kwargs):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(*args, **kwargs)
    return np.nan_to_num(result)


def derivative_calculation_per_feature(X: np.array, eps: float = 1e-5):
    dts = []
    for dim in range(X.shape[1]):
        dx_dt = np.gradient(X[:, dim])
        dts.append(dx_dt)
    deriv = np.stack(dts, axis=1)
    # Set absolute gradients which are lower than eps value to zero
    deriv[np.absolute(deriv) < eps] = 0
    return deriv


def compute_curvature(X: np.array, value_limit: float = 1e6) -> tuple[float, float, float, float]:
    gamma1 = derivative_calculation_per_feature(X)
    gamma2 = derivative_calculation_per_feature(gamma1)
    gamma3 = derivative_calculation_per_feature(gamma2)
    speed = np.sqrt(np.power(gamma1, 2).sum(axis=1))
    acceleration = np.sqrt(np.power(gamma2, 2).sum(axis=1))

    # Calculation from https://en.wikipedia.org/wiki/Differentiable_curve
    # E1 is the first Frenet vector also known as the unit tangent vector
    e1 = divide(gamma1, np.linalg.norm(gamma1, axis=1)[:, np.newaxis])
    # E2 is the unit normal vector
    part1 = np.einsum("ij,ij->i", gamma2, e1)
    normal = gamma2 - np.multiply(part1[:, np.newaxis], e1)
    e2 = divide(normal, np.linalg.norm(normal, axis=1)[:, np.newaxis])
    # The first generalized curvature X1 = \kappa (t)
    e1_d = derivative_calculation_per_feature(e1)
    kappa = divide(np.einsum("ij,ij->i", e1_d, e2), np.linalg.norm(gamma1, axis=1))
    # Should be equivalent to:
    # kappa = divide(np.linalg.norm(e1_d, axis=1), np.linalg.norm(gamma1, axis=1))
    # E3 is the binormal vector
    part1 = np.einsum("ij,ij->i", gamma3, e1)
    part2 = np.einsum("ij,ij->i", gamma3, e2)
    e3 = gamma3 - np.multiply(part1[:, np.newaxis], e1) - np.multiply(part2[:, np.newaxis], e2)
    E3 = divide(e3, np.linalg.norm(e3, axis=1)[:, np.newaxis])
    # Torsion is the second generalized curvature X2 = \tau(t)
    e2_d = derivative_calculation_per_feature(e2)
    tau = divide(np.einsum("ij,ij->i", e2_d, E3), np.linalg.norm(gamma1, axis=1))
    # Replace all -inf and inf values with the finite min and max
    tau = np.clip(tau, a_max=value_limit, a_min=-value_limit)
    kappa = np.clip(kappa, a_max=value_limit, a_min=-value_limit)
    tau = np.nan_to_num(tau, posinf=value_limit, neginf=-value_limit)
    kappa = np.nan_to_num(kappa, posinf=value_limit, neginf=-value_limit)
    return kappa, tau, speed, acceleration


def find_subsequence_groups_per_label(label_array: np.array, label: int):
    """This function creates a DataFrame with 'start' and 'end' index values for each occurrence of consecutive 'True' values provided by the Series created from the mask_labels. mask_labels is a True False Array which is True where the 'label' parameter is found in the 'label_array' parameter

    Args:
        label_array (np.array): array providing the class labels
        label (int): label for which the subsequences should be found in the label_array

    Returns:
        pd.DataFrame: _description_
    """
    # Create a mask for a single class
    mask_labels = np.array([True if (val == label) else False for val in label_array])
    ts = pd.Series(mask_labels)
    new_ts = pd.concat([pd.Series(np.array([False])), ts], ignore_index=True)
    df = pd.DataFrame({"times": new_ts.index - 1, "group": (new_ts.diff() == True).cumsum()})
    df = df.drop(0).reset_index(drop=True)
    fin_df = df.loc[df["group"] % 2 == 1].groupby("group")["times"].agg(["first", "last"]).rename(columns={"first": "start", "last": "end"})
    return fin_df


def compute_adapted_silhouette(df: pd.DataFrame) -> np.array:
    df_mean_cluster = df.groupby(["c_id"]).mean()
    # If only one cluster found, then set asc to 0:
    ascs = []
    if df.shape[0] == 1:
        ascs.append([df.index.get_level_values("c_id")[0], 0, 0])
    else:
        grouped = df.groupby(["c_id"])
        for name, group in grouped:
            group_A = group.groupby(["s_id"])
            for name_s, subsequence in group_A:
                # Find closest mean cluster center
                location_s = subsequence.values
                if group.shape[0] > 1:
                    mean_location_A_except_this = group.drop((name, name_s)).mean(axis=0).values
                    dist_As = np.linalg.norm(mean_location_A_except_this - location_s)
                else:
                    # mean_location_A_except_this = location_s
                    # dist_As = np.linalg.norm(location_s - np.absolute(location_s * df.index.get_level_values("std")[0]))
                    dist_As = 0
                # Calc distance to all other mean cluster centers
                # Drop current cluster and substract current subsequence location from all other cluster mean centers
                a_min_b = df_mean_cluster.drop(name).values - location_s
                # Calculate the euclidean distance using einsum here and take the minimum value
                dist_Bs = np.sqrt(np.einsum("ij,ij->i", a_min_b, a_min_b)).min()
                # pairwise_distances_chunked()
                # Now finally calculate the adapted silhouette coefficient
                # If dist_As is zero (due to the only subsequence found for this cluster) then asc would be 1 since it is dist_Bs / dist_Bs = 1
                # So we set asc in this case to zero. If dist_As and dist_Bs is zero the clusters are overlapping and so we set the asc to -1
                # What if the dist_As == 0 ? Can we take the std of all points in the one subsequence found?
                if dist_As == 0 and dist_Bs > 0:
                    asc = 0
                elif dist_As == 0 and dist_Bs == 0:
                    asc = -1
                else:
                    asc = (dist_Bs - dist_As) / np.array([dist_As, dist_Bs]).max()
                ascs.append([name, name_s, asc])
    return np.stack(ascs)


def mt3scm_score(X, labels, n_min_subs: int = 3, standardize_subs_curve: bool = True):
    """Compute the multivariate time series-subsequence clustering metric (mt3scm) score.
    #TODO: Explanation here!
    Procedure for finding nearest cluster:
    - Compute center position for each subsequence
    - Compute mean center position for each cluster
    - For each subsequence find the nearest cluster by finding minimal distance 'dist(B, s)' to other mean cluster centers
    - Analogue to silhouette computation: compute the adapted silhouette coefficient with (where A is the cluster the subsequence 's' belongs to )
        - asc = dist(B, s) - dist(A, s) / max{dist(A, s), dist(B, S)}
        - Where B is the closest mean cluster center

    Restrictions:
    -------------
        This metric should only be used on multivariate time series data, which also need to be a differentiable curve in n-dimensional space. Because the curvature for each point in time is being calculated. No 'movement' of the curve is being compensated in this calculation but the results may not be as expected or favor wrong or other parts of the data higher. If gradients become very steep (inf, -inf) they are being replaced with any existing finite min and max values found.


    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point. n_features must be 2 or more

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    n_min_subs : int, default=3
        Is the minimal number of consecutive points in a the same subsequence provided by the label array when computing the mt3scm score. This needs to be at least 3 for calculating a reasonable gradient when computing the curvature of the subsequence. Subsequence with consecutive points smaller than n_min_subs get a kappa and tau of 0 but curvature consistency coefficient of 1

    Returns
    -------
    score : float
        The resulting mt3scm score.

    References
    ----------
    #TODO: Insert publication
    """
    X, labels = check_X_y(X, labels, ensure_min_features=2)
    X = check_data_constant_values(X)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # label_freqs = np.bincount(labels)
    uniq_labels = np.unique(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    # n_min_subs is the minimal number of points in a subsequence
    # Calculate the curvature and the torsion for all points and find min and max for normalization later
    kappa_X, tau_X, _, _ = compute_curvature(X)
    # Set min max values for curvature and torsion
    kappa_max = kappa_X.max()
    kappa_min = kappa_X.min()
    kappa_mean = kappa_X.mean()
    kappa_std = np.nanstd(kappa_X)
    tau_max = tau_X.max()
    tau_min = tau_X.min()
    tau_mean = tau_X.mean()
    tau_std = np.nanstd(tau_X)

    cccs = []
    subs_curve_data = []
    subs_center_data = []
    # Iterate over unique labels or cluster ids
    for cluster_id in uniq_labels:
        # Find consecutive subsequences per cluster
        df_subs = find_subsequence_groups_per_label(labels, cluster_id)
        sccs = []
        # Iterate over each subsequence in this cluster
        for subsequence_id, row in enumerate(df_subs.itertuples(index=False)):
            idx_start = row[0]
            idx_end = row[1] + 1
            seq_len = idx_end - idx_start
            # Get the data of this subsequence
            subs_data = X[idx_start:idx_end]
            # Normalize subsequence data with min max of all data
            norm_subs_data = (subs_data - data_min) / (data_max - data_min)
            # Calculate standard deviation for the normalized subsequence data
            std_pos = norm_subs_data.std(axis=0).mean()
            # Get the center position as the middle of the subsequence
            center_pos = np.take(subs_data, subs_data.shape[0] // 2, axis=0)
            subs_center_data.append([int(cluster_id), int(subsequence_id), std_pos] + center_pos.tolist())
            if seq_len > n_min_subs:
                kappa_S, tau_S, _, _ = compute_curvature(subs_data)
                if standardize_subs_curve is True:
                    # Standardize curvature and torsion
                    kappa_norm = divide((kappa_S - kappa_mean), kappa_std)
                    tau_norm = divide((tau_S - tau_mean), tau_std)
                else:
                    # Normalize curvature and torsion
                    kappa_norm = (kappa_S - kappa_min) / (kappa_max - kappa_min)
                    tau_norm = (tau_S - tau_min) / (tau_max - tau_min)
                # Compute the subsequence curvature consistency (scc)
                scc = 1 - ((np.std(kappa_norm) + np.std(tau_norm)) / 2)
                # Compute the subsequence mean curvature and torsion
                mean_kappa_norm = np.mean(kappa_norm)
                mean_tau_norm = np.mean(tau_norm)
                # Create array of this subsequence id and mean kappa and tau
                subs_curve = np.array([int(cluster_id), int(subsequence_id), std_pos, mean_kappa_norm, mean_tau_norm])
                subs_curve_data.append(subs_curve)
                sccs.append(scc)
            else:
                subs_curve = np.array([int(cluster_id), int(subsequence_id), std_pos, 0, 0])
                subs_curve_data.append(subs_curve)
                sccs.append(1)
        # Compute the cluster curvature consistency (ccc) with the arithmetic mean of the sccs
        # cccs.append(stats.hmean(sccs))
        cccs.append(np.mean(sccs))
    # Mean normalized kappa and tau for each subsequence. Stack and create DataFrame
    cluster_curve_data = np.stack(subs_curve_data)
    df_curve = pd.DataFrame(
        cluster_curve_data[:, 3:],
        index=pd.MultiIndex.from_arrays(cluster_curve_data[:, 0:3].T.astype("int"), names=["c_id", "s_id", "std"]),
        columns=["mean_kappa_norm", "mean_tau_norm"],
    )
    # Mean center position for each subsequence. Stack and create DataFrame
    cluster_center_data = np.stack(subs_center_data)
    df_centers = pd.DataFrame(
        cluster_center_data[:, 3:],
        index=pd.MultiIndex.from_arrays(cluster_center_data[:, 0:3].T, names=["c_id", "s_id", "std"]),
        columns=[f"x{i}" for i in range(cluster_center_data[:, 3:].shape[1])],
    )
    # Compute adapted silhouette coefficient using cluster centers
    ascs_pos = compute_adapted_silhouette(df_centers)
    # Compute adapted silhouette coefficient using kappa and tau
    ascs_kt = compute_adapted_silhouette(df_curve)
    # Arithmetik mean cluster curvature consistency
    cc = np.mean(cccs)
    # Mean adapted silhouette scores
    masc_pos = np.mean(ascs_pos[:, 2])
    masc_kt = np.mean(ascs_kt[:, 2])
    masc = (masc_kt + masc_pos) / 2
    metric = (cc + masc) / 2
    return metric
