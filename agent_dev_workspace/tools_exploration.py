from langchain_core.tools import tool
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_idx_1: int, doc_idx_2: int) -> dict:
        """Computes the cosine similarity between two documents in the global document-term matrix.

        Args:
            doc_idx_1: Index of the first document row in session.feature_matrix.
            doc_idx_2: Index of the second document row in session.feature_matrix.
        """
        if session.feature_matrix is None:
            return {"error": "Feature matrix (DTM) has not been built yet. Run build_dtm_tool first."}

        mat = session.feature_matrix
        if doc_idx_1 < 0 or doc_idx_1 >= mat.shape[0] or doc_idx_2 < 0 or doc_idx_2 >= mat.shape[0]:
            return {"error": f"Document index out of bounds for matrix of shape {mat.shape}"}

        row1 = mat[doc_idx_1:doc_idx_1+1]
        row2 = mat[doc_idx_2:doc_idx_2+1]

        # Handle both sparse matrices and dense numpy arrays safely
        arr1 = row1.toarray() if hasattr(row1, "toarray") else np.asarray(row1)
        arr2 = row2.toarray() if hasattr(row2, "toarray") else np.asarray(row2)

        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0.0 or norm2 == 0.0:
            sim = 0.0
        else:
            sim = float(cosine_similarity(arr1, arr2)[0, 0])

        summary = {
            "result_id": session.next_result_id("cosine_similarity"),
            "doc_idx_1": doc_idx_1,
            "doc_idx_2": doc_idx_2,
            "similarity": round(sim, 4)
        }
        session.store_result("cosine_similarity_tool", {"doc_idx_1": doc_idx_1, "doc_idx_2": doc_idx_2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Computes a feature-vs-feature Pearson correlation heatmap over the top-20 terms by variance.

        Operates on the global DTM (session.feature_matrix / session.feature_names).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Feature matrix and names must be built before computing correlations."}

        mat = session.feature_matrix
        dense_mat = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
        feature_names = session.feature_names

        df = pd.DataFrame(dense_mat, columns=feature_names)
        
        # Compute variance for each term
        variances = df.var(axis=0)
        
        # Select top 20 terms by variance (or all if fewer than 20)
        n_top = min(20, len(feature_names))
        top_terms = variances.nlargest(n_top).index.tolist()

        # Compute correlation matrix across top variance terms
        corr_df = df[top_terms].corr()

        # Build heatmap and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(14, 14))
        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            square=True,
            linewidths=0.5,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=90)
        ax.set_title("Feature Correlation Matrix (Top Terms by Variance)")
        plt.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_df.round(4).to_dict()
        }
        session.store_result("feature_correlation_matrix_tool", {}, summary, full_report=corr_df)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
