from agent_pipeline.session_state import SessionState
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

def build_dtm_tool(session: SessionState):
    @tool
    def build_dtm_tool(text_column: str = "text") -> dict:
        """Build a Document-Term Matrix using CountVectorizer, keeping feature_matrix sparse.

        Args:
            text_column: Name of the column in session.dataframe containing the text to vectorize.
        """
        if session.dataframe is None:
            raise ValueError("No dataframe found in session. Load a dataset first.")
        
        if text_column not in session.dataframe.columns:
            text_column = session.dataframe.columns[0]

        if "category_name" not in session.dataframe.columns:
            raise ValueError("Dataset missing 'category_name' column.")

        corpus = session.dataframe[text_column].fillna("").astype(str).tolist()
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(corpus)

        session.feature_matrix = matrix
        session.feature_names = vectorizer.get_feature_names_out().tolist()
        session.artifacts["count_vectorizer"] = vectorizer

        labels = session.dataframe["category_name"].values
        session.set_labels(labels)

        n_docs, n_terms = session.feature_matrix.shape
        non_zero = int(matrix.nnz)
        total_elements = n_docs * n_terms
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "vocabulary_size": n_terms,
            "n_documents": n_docs,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
        }

        session.store_result("build_dtm_tool", {"text_column": text_column}, summary)
        return summary

    return build_dtm_tool

def term_frequency_tool(session: SessionState):
    @tool
    def term_frequency_tool() -> dict:
        """Aggregate term frequencies across all documents.

        Args:
            None.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("Feature matrix not found. Run build_dtm_tool first.")

        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)

        term_sums = np.sum(X_dense, axis=0)
        frequencies = {
            term: int(freq) for term, freq in zip(session.feature_names, term_sums)
        }

        result_id = session.next_result_id("term_frequency")
        summary = {"result_id": result_id, "term_frequencies": frequencies}

        session.store_result("term_frequency_tool", {}, summary)
        return summary

    return term_frequency_tool

def dtm_heatmap_tool(session: SessionState):
    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Render a heatmap of a slice of the raw document-term matrix.

        Args:
            n_terms: Number of terms (columns) to include in the slice.
            n_documents: Number of documents (rows) to include in the slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("Feature matrix not found. Run build_dtm_tool first.")

        total_docs, total_terms = session.feature_matrix.shape

        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms)

        matrix_slice = session.feature_matrix[:actual_docs, :actual_terms]
        if hasattr(matrix_slice, "toarray"):
            slice_mat = matrix_slice.toarray()
        else:
            slice_mat = np.asarray(matrix_slice)

        subset_terms = session.feature_names[:actual_terms]
        document_indices = list(range(actual_docs))
        if session.labels is not None:
            document_labels = (
                session.labels[:actual_docs].tolist()
                if hasattr(session.labels, "tolist")
                else list(session.labels[:actual_docs])
            )
        else:
            document_labels = []

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            slice_mat,
            annot=True,
            fmt="d",
            xticklabels=subset_terms,
            yticklabels=document_labels if len(document_labels) == actual_docs else True,
            cmap="Blues",
        )
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        ax.set_title("Document-Term Matrix Heatmap")
        plt.tight_layout()

        session.pending_figure = plt.gcf()
        plt.close()

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "matrix_slice": slice_mat.tolist(),
            "terms": subset_terms,
            "document_indices": document_indices,
            "document_labels": document_labels,
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return dtm_heatmap_tool

def make_tools(session: SessionState):
    return [build_dtm_tool(session), term_frequency_tool(session), dtm_heatmap_tool(session)]
