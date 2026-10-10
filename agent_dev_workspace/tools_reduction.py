import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional
from langchain_core.tools import tool
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: int = 42
    ) -> dict:
        """Reduces the dimensionality of the document-term matrix using PCA, Truncated SVD, or t-SNE,
        stores the reduced coordinates in session.artifacts, and generates a scatter plot colored by category
        set on session.pending_figure.

        Args:
            method (str): Dimensionality reduction technique to use ("pca", "svd", or "tsne"). Default is "pca".
            n_components (int): Number of dimensions to reduce to. Default is 2.
            random_state (int): Random seed for reproducibility. Default is 42.

        Returns:
            dict: Summary of the reduction containing result_id, method, n_components, and shape of reduced matrix.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix found. Build DTM first."}
        
        # Convert sparse feature matrix to dense if necessary for certain methods
        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)

        # Apply the chosen dimensionality reduction technique
        method_lower = method.lower()
        if method_lower == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced_coords = reducer.fit_transform(X_dense)
            explained_variance = getattr(reducer, "explained_variance_ratio_", None)
            if explained_variance is not None:
                explained_variance = explained_variance.tolist()
        elif method_lower in ["svd", "trunc_svd", "truncated_svd"]:
            reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
            reduced_coords = reducer.fit_transform(X_dense)
            explained_variance = getattr(reducer, "explained_variance_ratio_", None)
            if explained_variance is not None:
                explained_variance = explained_variance.tolist()
        elif method_lower == "tsne":
            # t-SNE doesn't have an explained_variance_ratio_
            reducer = TSNE(n_components=n_components, random_state=random_state, init="pca", learning_rate="auto")
            reduced_coords = reducer.fit_transform(X_dense)
            explained_variance = None
        else:
            return {"error": f"Unknown reduction method: {method}. Choose from 'pca', 'svd', or 'tsne'."}

        # Store reduced coordinates in artifacts
        artifact_key = f"reduced_{method_lower}"
        session.artifacts[artifact_key] = reduced_coords

        # Plotting requirement: scatter plot of the reduced coordinates colored by category
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Determine labels/categories for coloring
        if session.labels is not None and len(session.labels) == reduced_coords.shape[0]:
            labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        else:
            labels = np.zeros(reduced_coords.shape[0])

        sns.scatterplot(
            x=reduced_coords[:, 0],
            y=reduced_coords[:, 1],
            hue=labels,
            palette="Set1",
            ax=ax,
            legend="full"
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method_lower,
            "n_components": n_components,
            "shape": list(reduced_coords.shape),
            "explained_variance_ratio": explained_variance
        }
        
        params_dict = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state
        }
        full_report = pd.DataFrame(reduced_coords)
        session.store_result("reduce_dimensions_tool", params_dict, summary, full_report=full_report)
        return summary

    @tool
    def binarize_labels_tool(
        target_column: str = "category_name"
    ) -> dict:
        """One-hot encodes the categorical labels from the working dataframe,
        storing the resulting binary label matrix in session.artifacts["binary_labels"].

        Args:
            target_column (str): Name of the column in session.dataframe to binarize. Default is "category_name".

        Returns:
            dict: Summary containing result_id, target_column, unique_classes, and shape of the binary label matrix.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found. Load dataset first."}
        
        if target_column not in session.dataframe.columns:
            return {"error": f"Column '{target_column}' not found in dataframe."}

        series = session.dataframe[target_column]
        dummies = pd.get_dummies(series, prefix=target_column)
        binary_matrix = dummies.astype(int).values

        session.artifacts["binary_labels"] = binary_matrix
        session.artifacts["binary_labels_columns"] = list(dummies.columns)

        unique_classes = list(series.unique())

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "target_column": target_column,
            "unique_classes": unique_classes,
            "shape": list(binary_matrix.shape)
        }

        params_dict = {
            "target_column": target_column
        }
        full_report = dummies
        session.store_result("binarize_labels_tool", params_dict, summary, full_report=full_report)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
