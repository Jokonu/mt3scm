"""Computation functions for 'Multivariate Time Series Sub-Sequence Clustering Metric'"""

# Author: Jonas Köhne <jokohonas@gmail.com>
# License: BSD 3 clause

# Third Party Libraries Import
import numpy as np
import pandas as pd
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import check_X_y


def mt3scm_score(X, labels, n_min_subs: int = 3, standardize_subs_curve: bool = True):
    cm = MT3SCM()
    return cm.mt3scm_score(X, labels, n_min_subs, standardize_subs_curve)


def check_data_constant_values(X: np.ndarray):
    """Checking for only constant values in the data and raising ValueError

    Args:
        X (np.array): multivariate time series data provided

    Raises:
        ValueError: "Some or all dimensions of X have only constant values over time! No clustering possible! Remove constant columns"

    Returns:
        _type_: np.ndarray
    """
    if ((X.min(axis=0) - X.max(axis=0)) == 0).any():
        raise ValueError(
            "Some or all dimensions of X have only constant values over time! No clustering possible! Remove constant columns"
        )
    return X


def divide(*args, **kwargs):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(*args, **kwargs)
    return np.nan_to_num(result)


def derivative_calculation_per_feature(X: np.ndarray, eps: float = 1e-5):
    dts = []
    for dim in range(X.shape[1]):
        dx_dt = np.gradient(X[:, dim])
        dts.append(dx_dt)
    deriv = np.stack(dts, axis=1)
    # Set absolute gradients which are lower than eps value to zero
    deriv[np.absolute(deriv) < eps] = 0
    return deriv


def compute_curvature(
    X: np.ndarray, value_limit: float = 1e6, eps: float = 1e-5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gamma1 = derivative_calculation_per_feature(X, eps)
    gamma2 = derivative_calculation_per_feature(gamma1, eps)
    gamma3 = derivative_calculation_per_feature(gamma2, eps)
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
    e3 = (
        gamma3
        - np.multiply(part1[:, np.newaxis], e1)
        - np.multiply(part2[:, np.newaxis], e2)
    )
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


def find_subsequence_groups_per_label(label_array: np.ndarray, label: int):
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
    df = pd.DataFrame(
        {"times": new_ts.index - 1, "group": (new_ts.diff() == 1).cumsum()}
    )
    df = df.drop(0).reset_index(drop=True)
    fin_df = (
        df.loc[df["group"] % 2 == 1]
        .groupby("group")["times"]
        .agg(["first", "last"])
        .rename(columns={"first": "start", "last": "end"})
    )
    return fin_df


def compute_adapted_silhouette(
    df: pd.DataFrame,
    min_distance: float = 1e-2,
    eps: float = 1e-5,
    distance_fn: str = "euclidean",
) -> np.ndarray:
    df_mean_cluster = df.groupby(["c_id"]).mean()
    # If only one cluster found, then set asc to 0:
    ascs = []
    dist_As: float = 0.0
    dist_Bs: float = 0.0
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
                    mean_location_A_except_this = (
                        group.drop((name, name_s)).mean(axis=0).values
                    )
                    if distance_fn == "euclidean":
                        dist_As = float(
                            np.linalg.norm(mean_location_A_except_this - location_s)
                        )
                    elif distance_fn == "manhatten":
                        dist_As = np.sum(
                            np.absolute(mean_location_A_except_this - location_s)
                        )
                    dist_As = 0 if (np.absolute(dist_As) < min_distance) else dist_As
                else:
                    # Set adapted silhouette coefficient to zero, since only one subsequence per cluster
                    dist_As = 0
                # Calc distance to all other mean cluster centers
                # Drop current cluster and substract current subsequence location from all other cluster mean centers
                a_min_b = df_mean_cluster.drop(name).values - location_s
                # Calculate the euclidean distance using einsum here and take the minimum value
                if distance_fn == "euclidean":
                    dist_Bs = np.sqrt(np.einsum("ij,ij->i", a_min_b, a_min_b)).min()
                elif distance_fn == "manhatten":
                    dist_Bs = np.sum(np.absolute(a_min_b), axis=1).min()
                # Set absolute distance which is lower than eps value to zero
                dist_Bs = 0 if np.absolute(dist_Bs) < min_distance else dist_Bs
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


class MT3SCM:
    def __init__(
        self,
        eps: float = 1e-8,
        include_speed_acceleration: bool = False,
        distance_fn: str = "manhatten",
    ) -> None:
        self.eps = eps
        self.cc: float = 0.0
        self.wcc: float = 0.0
        self.masc_pos: float = 0.0
        self.masc_kt: float = 0.0
        self.masc: float = 0.0
        self.metric: float = 0.0
        self.cccs: list = []
        self.np_cs: list = []
        self.kappa_X: np.ndarray = np.array([])
        self.tau_X: np.ndarray = np.array([])
        self.speed_X: np.ndarray = np.array([])
        self.acceleration_X: np.ndarray = np.array([])
        self.ascs_pos: np.ndarray = np.array([])
        self.ascs_kt: np.ndarray = np.array([])
        self.include_speed_acceleration: bool = include_speed_acceleration
        self.scale_input_data: bool = True
        self.distance_fn: str = (
            distance_fn if (distance_fn in ["manhatten", "euclidean"]) else "manhatten"
        )
        self.df_curve: pd.DataFrame = pd.DataFrame()
        self.df_centers: pd.DataFrame = pd.DataFrame()

    def mt3scm_score(
        self, X, labels, n_min_subs: int = 3, standardize_subs_curve: bool = True
    ):
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
        uniq_labels = np.unique(labels)
        n_samples, _ = X.shape
        n_labels = len(le.classes_)
        check_number_of_labels(n_labels, n_samples)
        if self.scale_input_data is True:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)
        # n_min_subs is the minimal number of points in a subsequence
        # Calculate the curvature and the torsion for all points and find min and max for normalization later
        # self.kappa_X, self.tau_X, self.speed_X, self.acceleration_X = compute_curvature(X, eps=self.eps)
        curve_data = compute_curvature(X, eps=self.eps)
        if standardize_subs_curve is True:
            scaler = StandardScaler()
            # Switch dimensions here since the scaler scales on feature axis shape (n_samples, n_features)
            curve_data_array = np.array(curve_data).T
            curve_norm_data = scaler.fit_transform(curve_data_array).T
            (
                self.kappa_X,
                self.tau_X,
                self.speed_X,
                self.acceleration_X,
            ) = curve_norm_data
        else:
            self.kappa_X, self.tau_X, self.speed_X, self.acceleration_X = curve_data
        # import pdb;pdb.set_trace()

        if self.include_speed_acceleration is True:
            features = {
                "kappa": self.kappa_X,
                "tau": self.tau_X,
                "speed": self.speed_X,
                "acceleration": self.acceleration_X,
            }
        else:
            features = {"kappa": self.kappa_X, "tau": self.tau_X}
        mins = {}
        maxs = {}
        means = {}
        stds = {}
        # Do calculations for all features based on all data:
        for name, feature in features.items():
            mins[name] = feature.min()
            maxs[name] = feature.max()
            means[name] = feature.mean()
            stds[name] = np.nanstd(feature)
        subs_curve_data = []
        subs_center_data = []
        # Iterate over unique labels or cluster ids
        for cluster_id in uniq_labels:
            features_data_C_list: list[np.ndarray] = []
            # Find consecutive subsequences per cluster
            df_subs = find_subsequence_groups_per_label(labels, cluster_id)
            sccs: np.ndarray = np.array([])
            np_c = 0
            # Iterate over each subsequence in this cluster
            for subsequence_id, row in enumerate(df_subs.itertuples(index=False)):
                idx_start = row[0]
                idx_end = row[1] + 1
                seq_len = idx_end - idx_start
                # Get the data of this subsequence
                subs_data = X[idx_start:idx_end]
                if self.scale_input_data is False:
                    # Normalize subsequence data with min max of all data
                    norm_subs_data = (subs_data - data_min) / (data_max - data_min)
                    std_pos = norm_subs_data.std(axis=0).mean()
                else:
                    std_pos = subs_data.std(axis=0).mean()
                # Calculate standard deviation for the normalized subsequence data
                std_pos = subs_data.std(axis=0).mean()
                # Get the center position as the middle of the subsequence
                # Should this be the normalized data position?? If full data is already normalized then no!
                center_pos = np.take(subs_data, subs_data.shape[0] // 2, axis=0)
                subs_center_data.append(
                    [int(cluster_id), int(subsequence_id), std_pos]
                    + center_pos.tolist()
                )
                features_S = {}
                for name, feature in features.items():
                    features_S[name] = feature[idx_start:idx_end]
                # concat the feature value arrays
                features_data = np.concatenate(
                    [
                        np.expand_dims(features_S[key], axis=1)
                        for key in sorted(features_S)
                    ],
                    1,
                )
                # collect feature data over this cluster
                # features_data_C = np.concatenate([features_data_C, features_data], axis=0)
                # features_data_C = np.vstack((features_data_C, features_data)) if features_data_C == 0 else features_data
                mean_normeds = features_data.mean(axis=0)
                features_data_C_list.append(features_data)
                subs_curve = np.array([int(cluster_id), int(subsequence_id), std_pos])
                subs_curve = np.concatenate([subs_curve, mean_normeds], axis=0)
                subs_curve_data.append(subs_curve)
                # Compute the subsequence curvature consistency (scc) with scc = 1 - s
                # where the empirical standard deviation (or unbiased sample standard deviation) for each feature vector {\overline {x}} is: s =\sqrt{{\frac {1}{n-1}}\sum \limits _{i=1}^{n}\left(x_{i}-{\overline {x}}\right)^{2}}
                # s = np.sqrt(np.power((features_data - features_data.mean(axis=0)), 2).sum(axis=0) / (features_data.shape[0] - 1))
                # This is equivalent to:
                # s = np.std(features_data, axis=0, ddof=1)

                np_c += seq_len  # sum the number of points per sequence in this cluster
            # Compute the cluster curvature consistency (ccc) with ...
            features_data_C: np.ndarray = np.vstack(features_data_C_list)
            if features_data_C.shape[0] == 1:
                # TODO This is subject for calibration! How to penalize clusters with only one subsequence?
                single_subsequence_in_cluster_value: float = 0.0
                sccs = np.full(
                    features_data_C.shape[1], single_subsequence_in_cluster_value
                )
            else:
                sccs = 1 - np.std(features_data_C, axis=0, ddof=1)
            # restrict it to 1 and -1
            sccs = np.clip(sccs, a_max=1, a_min=-1)
            # Collect the cluster curvature consistencies (ccc) with the arithmetic mean of the sccs
            self.cccs.append(np.mean(sccs))
            # TODO: should keep those split up? like: self.cccs.append(sccs)
            self.np_cs.append(np_c)  # collect the number of points per cluster
        # Mean normalized kappa and tau for each subsequence. Stack and create DataFrame
        cluster_curve_data = np.stack(subs_curve_data)
        column_names = [f"mean_{name}_norm" for name in features.keys()]
        self.df_curve = pd.DataFrame(
            cluster_curve_data[:, 3:],
            index=pd.MultiIndex.from_arrays(
                cluster_curve_data[:, 0:3].T.astype("int"),
                names=["c_id", "s_id", "std"],
            ),
            columns=column_names,
        )
        # Mean center position for each subsequence. Stack and create DataFrame
        cluster_center_data = np.stack(subs_center_data)
        self.df_centers = pd.DataFrame(
            cluster_center_data[:, 3:],
            index=pd.MultiIndex.from_arrays(
                cluster_center_data[:, 0:3].T, names=["c_id", "s_id", "std"]
            ),
            columns=[f"x{i}" for i in range(cluster_center_data[:, 3:].shape[1])],
        )
        # Compute adapted silhouette coefficient using cluster centers
        self.ascs_pos = compute_adapted_silhouette(
            self.df_centers, self.eps, distance_fn=self.distance_fn
        )
        # Compute adapted silhouette coefficient using kappa and tau
        self.ascs_kt = compute_adapted_silhouette(
            self.df_curve, self.eps, distance_fn=self.distance_fn
        )
        # Arithmetik mean cluster curvature consistency
        self.cc = np.mean(self.cccs)
        # Calculate the mean cluster curvature consistency by weighing with the number of datapoints per cluster:
        self.wcc = np.sum(np.array(self.cccs) * np.array(self.np_cs)) / np.sum(
            np.array(self.np_cs)
        )
        # Mean adapted silhouette scores
        self.masc_pos = np.mean(self.ascs_pos[:, 2])
        self.masc_kt = np.mean(self.ascs_kt[:, 2])
        self.masc = (self.masc_kt + self.masc_pos) / 2
        # self.metric = (self.wcc + self.masc) / 2
        self.metric = (self.cc + self.masc_pos + self.masc_kt) / 3
        return self.metric
