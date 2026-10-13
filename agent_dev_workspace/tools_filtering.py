from agent_pipeline.session_state import SessionState
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool

def make_tools(session: SessionState):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filter out features whose variance is below the given threshold.

        Args:
            threshold: Minimum variance threshold to keep a feature.
        """
        X = session.feature_matrix
        feature_names = session.feature_names

        if X is None or feature_names is None:
            raise ValueError(
                "Feature matrix or feature names not found in session. "
                "Load the dataset, then run build_dtm_tool before variance_filter_tool."
            )

        if hasattr(X, "toarray"):
            X_dense = X.toarray()
            variances = np.var(X_dense, axis=0, ddof=0)
        else:
            variances = np.var(np.asarray(X), axis=0, ddof=0)
        variances = np.asarray(variances).ravel()

        variances_dict = {feat: float(var) for feat, var in zip(feature_names, variances)}
        kept_features = [feat for feat, var in variances_dict.items() if var >= threshold]
        kept_indices = [feature_names.index(feat) for feat in kept_features]
        session.feature_matrix = X[:, kept_indices]
        session.feature_names = kept_features

        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances,
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "kept_count": len(kept_features),
            "kept_features": kept_features,
        }
        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=df_report)
        return summary

    @tool
    def pearson_filter_tool(target_class: str = "", threshold: float = 0.0) -> dict:
        """Compute Pearson correlation between term counts and a target class.

        Args:
            target_class: Class label to correlate against (defaults to first category).
            threshold: Minimum absolute correlation required to keep a feature.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("Feature matrix not found in session. Build DTM first.")
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError("Session dataframe or category_name not found.")

        X = session.feature_matrix
        feature_names = session.feature_names
        labels = session.dataframe["category_name"].values
        if not target_class:
            target_class = session.categories[0] if session.categories else labels[0]

        y = (labels == target_class).astype(int)
        corrs = []
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            if hasattr(col_data, "toarray"):
                col_data = col_data.toarray().ravel()
            else:
                col_data = np.asarray(col_data).ravel()

            if np.std(col_data) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r, _ = pearsonr(col_data, y)
                if np.isnan(r):
                    r = 0.0
            corrs.append(float(r))

        df_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": corrs,
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)
        kept_features = [
            row["term"] for _, row in df_report.iterrows()
            if abs(row["pearson_r"]) >= threshold
        ]

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "kept_features": kept_features,
        }
        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    @tool
    def spearman_filter_tool(target_class: str = "", threshold: float = 0.0) -> dict:
        """Compute Spearman rank correlation between term counts and a target class.

        Args:
            target_class: Class label to correlate against (defaults to first category).
            threshold: Minimum absolute correlation required to keep a feature.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("Feature matrix not found in session. Build DTM first.")
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError("Session dataframe or category_name not found.")

        X = session.feature_matrix
        feature_names = session.feature_names
        labels = session.dataframe["category_name"].values
        if not target_class:
            target_class = session.categories[0] if session.categories else labels[0]

        y = (labels == target_class).astype(int)
        corrs = []
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            if hasattr(col_data, "toarray"):
                col_data = col_data.toarray().ravel()
            else:
                col_data = np.asarray(col_data).ravel()

            if np.std(col_data) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r, _ = spearmanr(col_data, y)
                if np.isnan(r):
                    r = 0.0
            corrs.append(float(r))

        df_report = pd.DataFrame({
            "term": feature_names,
            "spearman_r": corrs,
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)
        kept_features = [
            row["term"] for _, row in df_report.iterrows()
            if abs(row["spearman_r"]) >= threshold
        ]

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "kept_features": kept_features,
        }
        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
