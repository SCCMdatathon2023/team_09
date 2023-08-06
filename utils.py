#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""utils.py: contains utility functions for the project.

Acknowledgment:
    Some methods are adapted from research made by Jack Gallifant and Adrien Carrel.
        Code available at: https://github.com/AdrienC21/tscls
    Some methods are adapted from the paper "Development and clinical utility
    of machine learning algorithms for dynamic longitudinal real-time
    estimation of progression risks in active surveillance of early
    prostate cancer" from the van der Schaar lab at the University
    of Cambridge.
        Code available at: https://github.com/vanderschaarlab/ml-as-prostate-cancer
"""

import os
import warnings
from abc import ABC
from typing import List, Optional, Union, Any, Callable, Tuple, Dict
from time import perf_counter

import kaleido  # import kaleido FIRST to avoid any conflicts
import plotly
import matplotlib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from nltk.cluster.kmeans import KMeansClusterer
from fastdtw import fastdtw
from hdbscan import HDBSCAN


warnings.filterwarnings(
    "ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning
)
array_like = Union[np.ndarray, pd.DataFrame, List[float], pd.Series, List[int]]


def fdtw(x: array_like, y: array_like) -> np.float64:
    """Calculate the Dynamic Time Warping distance between
    two timeseries using the FastDTW package

    Args:
        x (array_like): timeseries
        y (array_like): timeseries

    Returns:
        np.float64: Dynamic Time Warping Distance
    """
    # just extract the distance, not the matching
    return fastdtw(x, y)[0]


def euclidean2D(x: array_like, y: array_like) -> np.float64:
    """Calculate the sum of the euclidean distance between
    two sets of timeseries

    Args:
        x (array_like): timeseries
        y (array_like): timeseries

    Returns:
        np.float64: sum of the euclidean distances
    """
    res = 0
    for u, v in zip(x, y):
        if (np.isnan(u).sum() == 0) and (np.isnan(v).sum() == 0):
            res += euclidean(u, v)
    return res


class CustomHDBSCAN:
    """Custom HDBSCAN class that allows to use precomputed distance matrix"""

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the class

        Args:
            verbose (bool, optional): whether to print the progress or not. Defaults to True.
        """
        self.verbose = verbose
        self.clusterer = HDBSCAN(metric="precomputed")
        self.distance_matrix = None  # initial distance matrix on train dataset to None
        self.train_set = None  # initial train set to None
        self.train_labels = None  # initial train labels to None
        self.nb_clusters = None  # initial number of clusters to None
        self.calculated_means = None  # initial means to None
        self.shift = None  # initial shift to None

    def print(self, msg: str) -> None:
        """Print function for the clusterer.

        Args:
            msg (str): message to print
        """
        if self.verbose:
            print(msg)

    def is_trained(self) -> bool:
        """Check if the clusterer is trained

        Returns:
            bool: True if the clusterer is trained, False otherwise
        """
        return self.distance_matrix is not None

    def calculate_distance_matrix(self, X: array_like) -> np.ndarray:
        """Calculate the Dynamic Time Warping distance matrix between the timeseries

        Args:
            X (array_like): timeseries

        Returns:
            np.ndarray: distance matrix
        """
        self.print("Calculating distance matrix ...")
        res = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                res[i, j] = fdtw(X[i], X[j])
                res[j, i] = res[i, j]
        return res

    def fit_predict(self, X: array_like) -> np.ndarray:
        """Fit the clusterer

        Args:
            X (array_like): distance matrix

        Returns:
            np.ndarray: predicted clusters
        """
        self.print(f"Fitting the clusterer on {len(X)} samples ...")
        self.train_set = X  # store the train set
        distance_matrix = self.calculate_distance_matrix(X)
        self.distance_matrix = distance_matrix  # store the distance matrix
        self.clusterer.fit(distance_matrix)
        self.train_labels = self.clusterer.labels_  # store the train clusters
        self.nb_clusters = len(
            set(list(self.clusterer.labels_))
        )  # store the nb clusters
        self.shift = -min(self.train_labels)  # store the shift
        return self.train_labels

    def predict(self, X: array_like) -> np.ndarray:
        """Predict the clusters of the test set

        Args:
            X (array_like): test/valid set

        Returns:
            np.ndarray: predicted clusters
        """
        self.print(f"Predicting the clusters on {len(X)} samples ...")
        res = np.zeros(X.shape[0], dtype=np.int32)
        for i in range(X.shape[0]):
            min_dist = np.inf
            for j in range(self.train_set.shape[0]):
                dist = fdtw(X[i], self.train_set[j])
                if dist < min_dist:
                    min_dist = dist
                    res[i] = self.train_labels[j]
        return res

    def means(self) -> np.ndarray:
        """Calculate the cluster means

        Returns:
            np.ndarray: cluster means
        """
        if self.calculated_means is not None:
            return self.calculated_means
        self.print("Calculating the cluster means ...")
        res = np.zeros(
            (
                self.nb_clusters + 1,
                self.train_set.shape[1],
                self.train_set.shape[2],
            )
        )
        for i in range(self.train_set.shape[0]):
            res[self.train_labels[i] + self.shift] += self.train_set[i]
        for i in range(res.shape[0]):
            div = np.sum(self.train_labels == (i - self.shift))
            if div:
                res[i] /= div
        self.calculated_means = res
        return res


class PlotClass(ABC):
    """Contains the plot methods Need to be supercharged."""

    def __init__(self, results_folder: str = "results"):
        """Initialize the plot class

        Args:
            results_folder (str, optional): folder to save the plots. Defaults to "results".
        """
        # General parameters
        self.results_folder = results_folder

        # Create the results folder if it doesn't already exist
        self.create_results_folder()

    def create_results_folder(self) -> None:
        """Create a results folder if it doesn't already exist"""
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

    def save_image(
        self,
        fig: Any,
        title: str = "fig",
        dpi: int = 300,
    ) -> None:
        """Function to adjust the figure and save it.

        Args:
            fig (Any): plotly or matplotlib figure
            title (str, optional): name of the file. Defaults to "fig".
            dpi (int, optional): DPI (Dots per Inch) for matplotlib plots. Defaults to 300.
        """
        if isinstance(fig, plotly.graph_objs._figure.Figure):  # plotly
            fig.write_image(os.path.join(self.results_folder, title))
        else:  # matplotlib
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            fig.savefig(
                os.path.join(self.results_folder, title),
                bbox_inches="tight",
                dpi=dpi,
                transparent=False,
            )
            plt.close()
        return

    def get_color(self, i: int, dic_colors: Union[Dict, None] = None) -> str:
        """Return a color for a given cluster (integer i here).
        This is useful to maintain a coherence between the different
        plots.

        Args:
            i (int): integer = cluster id
            dic_colors (Union[Dict, None], optional): Dictionnary
                that map an integer to an hexadecimal color.
                Defaults to None which correspond to an already
                created dictionnary.

        Returns:
            str: Hexadecimal color
        """
        # Return a color for a cluster i
        if dic_colors is None:
            dic_colors = {
                0: "#636EFA",
                1: "#EF553B",
                2: "#00CC96",
                3: "#AB63FA",
                4: "#FFA15A",
                5: "#19D3F3",
                6: "#FF6692",
                7: "#B6E880",
                8: "#FF97FF",
                9: "#FECB52",
            }
        return dic_colors.get(i % len(dic_colors), "#FFFFFF")

    def analyze_clusters(self, rows: int = 3, threshold_histogram: int = 10) -> None:
        """Plot the result of the clustring analysis.

        Args:
            rows (int, optional): Number of rows for the analysis. Defaults to 3.
            threshold_histogram (int, optional): Beyond which number of different
                values for a feature we plot a histogram rather than a bar plot.
                Defaults to 10.

        Raises:
            RuntimeError: If no clustering algorithms have been fitted before
        """
        if not (self.clustering_trained()):
            raise RuntimeError("Please execute the method 'run_clustering' before.")

        # Time varying features
        cols = (
            len(self.feat_timevarying) // rows
            if (len(self.feat_timevarying) % rows) == 0
            else (len(self.feat_timevarying) // rows) + 1
        )
        if self.is_supervised:
            rng = self.K_time
        else:
            rng = self.km_time.nb_clusters
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=self.feat_timevarying)
        for feat_id, _ in enumerate(self.feat_timevarying):
            for cluster_id in range(rng):
                ts = np.array(self.km_time.means()[cluster_id])[:, feat_id]
                # TODO CHANGE X?
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(ts))),
                        y=ts,
                        marker_color=self.get_color(cluster_id),
                        name=f"Cluster {cluster_id}",
                        showlegend=(feat_id == 0),
                    ),
                    row=((feat_id // cols) + 1),
                    col=((feat_id % cols) + 1),
                )
        fig.update_layout(
            height=800, width=1000, title_text="Cluster mean time varying features"
        )
        self.save_image(fig, "cluster_mean_time_varying_features.png")

        # Static features + outcome
        feat_to_plot = self.feat_static
        if self.label is not None:
            feat_to_plot.append(self.label)
        subplot_titles = [
            f"{dataset} {x}"
            for dataset in ("Train", "Valid", "Test")
            for x in feat_to_plot
        ]
        fig = make_subplots(
            rows=3, cols=len(feat_to_plot), subplot_titles=subplot_titles
        )
        df_train = self.df_train.copy(deep=True).groupby(by=[self.id_name]).first()
        df_valid = self.df_valid.copy(deep=True).groupby(by=[self.id_name]).first()
        df_test = self.df_test.copy(deep=True).groupby(by=[self.id_name]).first()
        if self.cap_datasets is not None:
            df_train = df_train[: self.cap_datasets]
            df_valid = df_valid[: self.cap_datasets]
            df_test = df_test[: self.cap_datasets]
        shift = self.km_time.shift
        df_train["cluster_ts"] = self.tr_clusters
        df_valid["cluster_ts"] = self.va_clusters
        df_test["cluster_ts"] = self.te_clusters
        for feat_id, feat in enumerate(feat_to_plot):
            possible_values = list(
                set(df_train[feat].to_list())
                .union(df_valid[feat].to_list())
                .union(df_test[feat].to_list())
            )
            for cluster_id in range(rng):
                # train
                df_filtered = df_train[df_train["cluster_ts"] == (cluster_id - shift)]
                y = [len(df_filtered[df_filtered[feat] == x]) for x in possible_values]
                if len(possible_values) >= threshold_histogram:
                    fig.add_trace(
                        go.Histogram(
                            x=df_filtered[feat].to_numpy(),
                            marker_color=self.get_color(cluster_id),
                            name=f"Cluster {cluster_id - shift}",
                            showlegend=(feat_id == 0),
                        ),
                        row=1,
                        col=(feat_id + 1),
                    )
                else:
                    fig.add_trace(
                        go.Bar(
                            x=possible_values,
                            y=y,
                            marker_color=self.get_color(cluster_id),
                            name=f"Cluster {cluster_id - shift}",
                            showlegend=(feat_id == 0),
                        ),
                        row=1,
                        col=(feat_id + 1),
                    )
                # valid
                df_filtered = df_valid[df_valid["cluster_ts"] == (cluster_id - shift)]
                y = [len(df_filtered[df_filtered[feat] == x]) for x in possible_values]
                if len(possible_values) >= threshold_histogram:
                    fig.add_trace(
                        go.Histogram(
                            x=df_filtered[feat].to_numpy(),
                            marker_color=self.get_color(cluster_id),
                            name=f"Cluster {cluster_id - shift}",
                            showlegend=False,
                        ),
                        row=2,
                        col=(feat_id + 1),
                    )
                else:
                    fig.add_trace(
                        go.Bar(
                            x=possible_values,
                            y=y,
                            marker_color=self.get_color(cluster_id),
                            name=f"Cluster {cluster_id - shift}",
                            showlegend=False,
                        ),
                        row=2,
                        col=(feat_id + 1),
                    )
                # test
                df_filtered = df_test[df_test["cluster_ts"] == (cluster_id - shift)]
                y = [len(df_filtered[df_filtered[feat] == x]) for x in possible_values]
                if len(possible_values) >= threshold_histogram:
                    fig.add_trace(
                        go.Histogram(
                            x=df_filtered[feat].to_numpy(),
                            marker_color=self.get_color(cluster_id),
                            name=f"Cluster {cluster_id - shift}",
                            showlegend=False,
                        ),
                        row=3,
                        col=(feat_id + 1),
                    )
                else:
                    fig.add_trace(
                        go.Bar(
                            x=possible_values,
                            y=y,
                            marker_color=self.get_color(cluster_id),
                            name=f"Cluster {cluster_id - shift}",
                            showlegend=False,
                        ),
                        row=3,
                        col=(feat_id + 1),
                    )

        fig.update_layout(
            height=800, width=1000, title_text="Cluster bar plot static features"
        )
        self.save_image(fig, "cluster_bar_plot_static_features.png")


class Team9(PlotClass):
    """Team9 class to conduct our analysis.
    Inherit from a PlotClass that contains the analysis functions.
    """

    def __init__(
        self,
        id_name: str,
        feat_timevarying: List[str],
        feat_static: List[str],
        metric: Union[str, Callable[[array_like, array_like], float]] = "custom_dtw",
        K_time: int = 5,
        scaler: Optional[Union[Callable[[pd.DataFrame], pd.DataFrame], str]] = None,
        fillna_strategy: Optional[
            Union[Callable[[pd.DataFrame], pd.DataFrame], str, float]
        ] = "fill_forward",
        label: Optional[str] = None,
        tte_name: Optional[str] = None,
        time_name: Optional[str] = None,
        seed: int = 42,
        test_size: float = 0.2,
        is_bigquery: bool = False,
        query_or_path: str = "./",
        results_folder: str = "results",
        verbose: bool = True,
        cap_datasets: Optional[int] = None,
    ) -> None:
        """Initialize the Team9 class object with our general configuration

        Args:
            id_name (str): hospital admission id column name
            feat_timevarying (List[str]): List of time series features.
            feat_static (List[str]): List of static features.
            metric (Union[str, Callable[[array_like, array_like], float]], optional): metric to be used when computing the clusters. It's either the name of a distance to be used in a KMeans clustering or it is a function. Defaults to "custom_dtw".
            K_time (int, optional): number of clusters for supervised clustering methods. Defaults to 5.
            scaler (Optional[Union[Callable[[pd.DataFrame], pd.DataFrame], str]], optional): Scaler to use. Defaults to None.
            fillna_strategy (Optional[Union[Callable[[pd.DataFrame], pd.DataFrame], str, float]], optional): Value or method to fill NaN values with. Defaults to "fill_forward.
            label (Optional[str], optional): label column name. Defaults to None.
            tte_name (Optional[str], optional): time to event column name (date covid diagnosis). Defaults to None.
            time_name (Optional[str], optional): name of the time column. Defaults to None.
            seed (int, optional): Random seed to allow reproducibility. Defaults to 42.
            test_size (float, optional): Test size for train, valid, test split. Defaults to 0.2.
            is_bigquery (bool, optional): if True, the query_or_path is an SQL query, otherwise it is a path. Defaults to False.
            query_or_path (str, optional): bigquery SQL query or path. Defaults to "./".
            results_folder (str, optional): name of the folder to save the results. Defaults to "results".
            verbose (bool, optional): whether to print information or not. Defaults to True.
            cap_datasets (Optional[int], optional): cap the size of the datasets if not None. Defaults to None.
        """
        # Initialize the parent PlotClass class
        super().__init__(results_folder)

        # General parameters
        self.seed = seed
        self.test_size = test_size
        self.verbose = verbose

        # Initialize dataset specific parameters
        self.id_name = id_name
        self.label = label
        self.feat_timevarying = feat_timevarying
        self.feat_static = feat_static
        self.fillna_strategy = fillna_strategy
        self.tte_name = tte_name
        self.time_name = time_name
        self.scaler = scaler
        self.cap_datasets = cap_datasets

        columns_to_add = [self.id_name]
        if self.label is not None:
            columns_to_add.append(self.label)
        if self.tte_name is not None:
            columns_to_add.append(self.tte_name)
        if self.time_name is not None:
            columns_to_add.append(self.time_name)
        self.columns = (
            self.feat_timevarying + self.feat_static + columns_to_add
        )  # all columns to consider

        # Model parameters
        self.metric = metric
        self.K_time = K_time

        # Load the dataset
        self.is_bigquery = is_bigquery
        self.query_or_path = query_or_path
        self.load_dataset(self.is_bigquery, self.query_or_path)

    def __repr__(self) -> str:
        """Print the object

        Returns:
            str: object representation
        """
        return f"{self.__clas__.__name__}(id_name={self.id_name}, feat_timevarying={self.feat_timevarying}, feat_static={self.feat_static}, metric={self.metric}, K_time={self.K_time}, scaler={self.scaler}, fillna_strategy={self.fillna_strategy}, label={self.label}, tte_name={self.tte_name}, time_name={self.time_name}, seed={self.seed}, test_size={self.test_size}, is_bigquery={self.is_bigquery}, query_or_path={self.query_or_path}, results_folder={self.results_folder}, verbose={self.verbose})"

    def print(self, msg: str) -> None:
        """Print the object"""
        if self.verbose:
            print(msg)

    def general_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """General preprocessing of the dataset.

        Args:
            df (pd.DataFrame): dataset

        Returns:
            pd.DataFrame: preprocessed dataset
        """
        # Label encoder static features
        le_dic = {}
        for feature in self.feat_static:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            le_dic[feature] = le
        self.label_encoders = le_dic  # save the label encoder(s)
        return df

    def load_dataset(
        self, is_bigquery: bool = False, query_or_path: str = "./"
    ) -> None:
        """Load the dataset, either by running a bigquery query or by loading a csv file at a specific path.

        Args:
            is_bigquery (bool, optional): if True, the query_or_path is an SQL query, otherwise it is a path. Defaults to False.
            query_or_path (str): bigquery SQL query or path. Defaults to "./".
        """
        if is_bigquery:
            df = None
            raise NotImplementedError("Bigquery not implemented yet.")
        else:
            df = pd.read_csv(query_or_path)
        # Preprocess the general dataset
        df = self.general_preprocessing(df)
        self.df = df  # save the dataset

    def fillna(
        self,
        df: pd.DataFrame,
        fillna_strategy: Optional[
            Union[Callable[[pd.DataFrame], pd.DataFrame], str, float]
        ] = None,
    ) -> pd.DataFrame:
        """Method to fill the nan values of the timevarying features of a patient

        Args:
            df (pd.DataFrame): dataframe with timevarying features
            fillna_strategy (Optional[Union[Callable[[pd.DataFrame], pd.DataFrame], str, float]], optional): _description_. Defaults to None.

        Raises:
            ValueError: raise an error if the method is unknown

        Returns:
            pd.DataFrame: processed dataframe
        """
        if fillna_strategy is None:
            return df
        if isinstance(fillna_strategy, str):
            if fillna_strategy == "fill_forward":
                return df.fillna(method="ffill").fillna(0.0)
            else:
                raise ValueError(
                    f"Unknown fillna strategy for time varying features: {fillna_strategy}."
                )
        if isinstance(fillna_strategy, float):
            return df.fillna(0.0)
        else:  # callable
            return fillna_strategy(df)

    def process_dataset(
        self,
        dataset: pd.DataFrame,
        scaler: Optional[Union[str, Any]] = None,
    ) -> Tuple[
        Tuple[array_like, array_like, array_like, array_like, array_like],
        Tuple[List[str], List[str]],
        Tuple[List[int], List[int]],
        Optional[Any],
    ]:
        """Process a train, valid or test dataset by applying the preprocessing steps
        at the patient/global level and storing the values into arrays for later analysis.

        Args:
            dataset (pd.DataFrame): train, valid or test COVID dataset
            scaler (Optional[str], optional): Method to scale/normalize the data or scaler object. Could be None. Defaults to None.

        Returns:
            Tuple[
                Tuple[array_like, array_like, array_like, array_like, array_like],
                Tuple[List[str], List[str]],
                Tuple[List[int], List[int]],
                Optional[Any]
            ]:
            1) tuple with:
                - the static features
                - the timevarying features
                - the timestamps (if exists)
                - the outcome
                - the time to event (tte) (if exists)
            2) tuple with:
                - list of the names of the static features
                - list of the names of the timevarying features
            3) tuple with:
                - ids of timevarying features that are binary
                - ids of timevarying features that are continuous (>3 different values)
                "delta" not included
            4) the scaler, either the already fitted one provided or the
                one that has been fitted during the preprocessing, or None if
                no scaling applied
        """
        # Make a copy of the dataset
        df = dataset.copy(deep=True)
        # Normalize the data
        if scaler is not None:
            if isinstance(scaler, str):
                if scaler == "normal":
                    scaler = StandardScaler()
                elif scaler == "minmax":
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Invalid normalization type {scaler}.")
                df[self.feat_timevarying] = scaler.fit_transform(
                    df[self.feat_timevarying]
                )
            else:  # normalize is an already trained scaler object
                df[self.feat_timevarying] = scaler.transform(df[self.feat_timevarying])
        else:  # scaler is None, we don't preprocess the data
            scaler = None

        # Group by id
        grouped = df.groupby(by=[self.id_name])
        id_list = grouped.nunique().index

        tmp = grouped.count()
        num_samples = len(tmp)
        max_length = tmp.max().max()

        data_xs = np.zeros([num_samples, len(self.feat_static)])
        data_xs[:, :] = np.asarray(
            df.drop_duplicates(subset=[self.id_name])[self.feat_static]
        )

        data_y = np.zeros([num_samples, 1])
        if self.label is not None:
            data_y[:, 0] = np.asarray(
                df.drop_duplicates(subset=[self.id_name])[self.label]
            )

        data_tte = np.zeros([num_samples, 1])
        if self.tte_name is not None:
            data_tte[:, 0] = np.asarray(
                df.drop_duplicates(subset=[self.id_name])[self.tte_name]
            )

        data_xt = np.zeros(
            [num_samples, max_length, len(self.feat_timevarying) + 1]
        )  # including deltas of time
        data_time = np.zeros([num_samples, max_length, 1])

        for i, pid in enumerate(id_list):
            tmp = grouped.get_group(pid)
            # Add delta in time
            delta_time = (
                np.asarray(tmp[self.time_name].diff())[1:]
                if self.time_name is not None
                else np.zeros(len(tmp) - 1)
            )
            data_xt[i, 1 : len(tmp), 0] = delta_time
            # Add other timevarying features
            time_varying_df = self.fillna(
                tmp[self.feat_timevarying], self.fillna_strategy
            )
            data_xt[i, : len(time_varying_df), 1:] = np.asarray(time_varying_df)
            # Time
            data_time[i, : len(tmp), 0] = (
                np.asarray(tmp[self.time_name])
                if self.time_name is not None
                else np.zeros(len(tmp))
            )

        if self.time_name is not None:
            data_xt[:, :, 0] = (
                data_xt[:, :, 0] / data_xt[:, :, 0].max()
            )  # min-max on delta's

        # Split the timevarying features into the binary and the "continuous" ones
        xt_bin_list, xt_con_list = [], []  # contains the ids

        for f_idx, feat in enumerate(self.feat_timevarying):
            if feat == "delta":
                continue
            if len(df[feat].unique()) == 2:
                xt_bin_list += [f_idx + 1]
            else:
                xt_con_list += [f_idx + 1]

        return (
            (data_xs, data_xt, data_time, data_y, data_tte),
            (self.feat_static, self.feat_timevarying),
            (xt_bin_list, xt_con_list),
            scaler,
        )

    def train_cluster(self, model: Any, data: array_like) -> array_like:
        """Train a clustering model on some data.

        Args:
            model (Any): Clustering model
            data (array_like): data

        Returns:
            array_like: cluster predictions on the data
        """
        if hasattr(model, "fit_predict"):
            return model.fit_predict(data)
        if hasattr(model, "cluster"):
            return model.cluster(data, assign_clusters=True)
        raise ValueError(
            f"The model {model.__class__.__name__} must have a 'fit_predict' or 'cluster' method to be trained on the dataset."
        )

    def classify(self, model: Any, data: array_like) -> array_like:
        """Make clustering predictions based on a pre-trained clustering model.

        Args:
            model (Any): trained clustering model
            data (array_like): data

        Returns:
            array_like: predicted clusters
        """
        if hasattr(model, "predict"):
            return model.predict(data)
        if hasattr(model, "classify"):
            return [model.classify(x) for x in data]
        raise ValueError(
            f"The model {model.__class__.__name__} must have a 'predict' or 'classify' method that allows cluster predictions on unseen data."
        )

    def run_clustering(self) -> None:
        """Process the dataset and split it into a train, valid and test set
        without data leakage.
        Then, fit a clustering algorithm on the timevarying features.

        Raises:
            ValueError: return an error if a parameter contains an
                unknown value. E.g. an unknown metric for the clustering
                algorithm or an unknown fillna strategy.
        """
        ###########
        # Preprocessing
        ###########

        self.print("Loading the file, group shuffle split, etc")
        df = self.df

        # Process time to get date and convert to timestamp to store into arrays
        if self.time_name is not None:
            df[self.time_name] = df[self.time_name].apply(
                lambda x: pd.to_datetime(x)  # format=...
            )
            df[self.time_name] = df[self.time_name].apply(
                lambda x: datetime.datetime.timestamp(x)
            )

        # Split and keep similar patient ids in the same folds
        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.seed
        )
        for train_idx, test_idx in gss.split(
            df, y=None, groups=df[self.id_name].to_numpy()
        ):
            df_train, df_test = df.loc[train_idx, :].reset_index(drop=True), df.loc[
                test_idx, :
            ].reset_index(drop=True)
        for train_idx, valid_idx in gss.split(
            df_train, y=None, groups=df_train[self.id_name].to_numpy()
        ):
            df_train, df_valid = df_train.loc[train_idx, :].reset_index(
                drop=True
            ), df_train.loc[valid_idx, :].reset_index(drop=True)

        # Feed the preprocessing and the models only with the columns of interest
        df_train_c = df_train[self.columns]
        df_valid_c = df_valid[self.columns]
        df_test_c = df_test[self.columns]

        # Preprocess train
        self.print("Preprocess train")
        (
            (tr_data_s, tr_data_t, tr_time, tr_label, tr_tte),
            (_, _),
            (xt_bin_list, xt_con_list),
            scaler,
        ) = self.process_dataset(df_train_c, scaler=self.scaler)

        # Preprocess valid
        self.print("Preprocess valid")
        (
            (va_data_s, va_data_t, va_time, va_label, va_tte),
            _,
            (_, _),
            _,
        ) = self.process_dataset(df_valid_c, scaler=scaler)

        # Preprocess test
        self.print("Preprocess test")
        (
            (te_data_s, te_data_t, te_time, te_label, te_tte),
            _,
            (_, _),
            _,
        ) = self.process_dataset(df_test_c, scaler=scaler)

        if self.cap_datasets is not None:
            if not (isinstance(self.cap_datasets, int) and self.cap_datasets > 0):
                raise ValueError(
                    f"The parameter 'cap_datasets' must be a positive integer, not {self.cap_datasets}."
                )
            tr_data_s = tr_data_s[: self.cap_datasets]
            va_data_s = va_data_s[: self.cap_datasets]
            te_data_s = te_data_s[: self.cap_datasets]
            tr_data_t = tr_data_t[: self.cap_datasets]
            va_data_t = va_data_t[: self.cap_datasets]
            te_data_t = te_data_t[: self.cap_datasets]
            tr_time = tr_time[: self.cap_datasets]
            va_time = va_time[: self.cap_datasets]
            te_time = te_time[: self.cap_datasets]
            tr_label = tr_label[: self.cap_datasets]
            va_label = va_label[: self.cap_datasets]
            te_label = te_label[: self.cap_datasets]
            tr_tte = tr_tte[: self.cap_datasets]
            va_tte = va_tte[: self.cap_datasets]
            te_tte = te_tte[: self.cap_datasets]

        # Add delta time to the list of time varying features
        self.feat_timevarying = ["delta"] + self.feat_timevarying

        ###########
        # Clustering
        ###########

        if isinstance(self.metric, str):
            if self.metric == "euclidean":
                km_time = KMeansClusterer(
                    self.K_time, distance=euclidean2D, avoid_empty_clusters=True
                )
                self.is_supervised = True
            elif self.metric == "dtw":
                km_time = KMeansClusterer(
                    self.K_time, distance=fdtw, avoid_empty_clusters=True
                )
                self.is_supervised = True
            elif self.metric == "custom_dtw":
                km_time = CustomHDBSCAN(self.verbose)
                self.is_supervised = False
            else:
                raise ValueError(f"Unknown clustering metric: {self.metric}")
        else:  # metric is a callable
            km_time = KMeansClusterer(
                self.K_time, distance=self.metric, avoid_empty_clusters=True
            )
            self.is_supervised = True

        # Train the clustering algorithm
        self.print("Training dynamic clustering ...")
        begin_train = perf_counter()
        tr_clusters = self.train_cluster(km_time, tr_data_t)
        self.print(f"Training took: {round(perf_counter() - begin_train, 3)} s")

        # Predictions on validation set
        self.print("Predicting validation clusters ...")
        top = perf_counter()
        va_clusters = self.classify(km_time, va_data_t)
        self.print(f"Predictions took: {round(perf_counter() - top, 3)} s")

        # Predictions on test set
        self.print("Predicting test clusters ...")
        top = perf_counter()
        te_clusters = self.classify(km_time, te_data_t)
        self.print(f"Predictions took: {round(perf_counter() - top, 3)} s")

        # Add the cluster to the timeseries
        self.feat_static = self.feat_static + ["cluster_ts"]

        # Save the objects
        self.print("Saving the objects")
        # Datasets
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        # Fitted scaler on train
        self.scaler = scaler
        # Clusters and fitted clustering model
        self.tr_clusters = tr_clusters
        self.va_clusters = va_clusters
        self.te_clusters = te_clusters
        self.km_time = km_time
        # Save the timeseries
        self.tr_data_t = tr_data_t
        self.va_data_t = va_data_t
        self.te_data_t = te_data_t
        # Save static features
        self.tr_data_s = tr_data_s
        self.va_data_s = va_data_s
        self.te_data_s = te_data_s

    def clustering_trained(self) -> bool:
        """Return whether or not a clustering algorithm has been
        fitted or not.

        Returns:
            bool: clustering fitted or not
        """
        return hasattr(self, "tr_clusters")
