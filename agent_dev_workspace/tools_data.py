import os
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_core.tools import tool

def make_tools(session):
    def _invalidate_derived_state():
        session.feature_matrix = None
        session.feature_names = None
        session.labels = None
        session.categories = []
        session.pending_figure = None
        session.artifacts.pop("count_vectorizer", None)
        for key in list(session.artifacts.keys()):
            if (
                key == "tokens"
                or key.startswith("binary_labels")
                or key.startswith("reduced_")
                or key.startswith("patterns_")
            ):
                session.artifacts.pop(key, None)

    @tool
    def tokenize_tool() -> dict:
        """Tokenizes each document's text into unigrams using nltk.word_tokenize.

        Populates session.dataframe['unigrams'] and session.artifacts['tokens'].
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "Dataset not loaded or 'text' column missing."}

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        tokens_list = []
        for text in session.dataframe["text"]:
            if pd.isna(text) or not str(text).strip():
                tokens_list.append([])
            else:
                tokens_list.append(nltk.word_tokenize(str(text).lower()))

        session.dataframe["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list

        summary = {
            "result_id": session.next_result_id("tokenize"),
            "n_documents": len(tokens_list),
            "status": "success"
        }
        session.store_result("tokenize_tool", {}, summary, full_report=pd.DataFrame({"tokens": tokens_list}))
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """Lists files and folders in a specified repository subdirectory.

        Args:
            subdirectory: Subdirectory path relative to repo root (e.g., 'newdataset' or '').
        """
        target_path = subdirectory if subdirectory else "."
        if not os.path.exists(target_path):
            return {"directories": [], "files": [], "error": f"Path '{target_path}' does not exist."}

        try:
            entries = os.listdir(target_path)
            directories = [e for e in entries if os.path.isdir(os.path.join(target_path, e))]
            files = [e for e in entries if os.path.isfile(os.path.join(target_path, e))]
            
            summary = {
                "result_id": session.next_result_id("list_files"),
                "subdirectory": subdirectory,
                "directories": sorted(directories),
                "files": sorted(files)
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary
        except Exception as e:
            return {"error": str(e)}

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """Peeks at the first n rows of the working DataFrame.

        Args:
            n: Number of rows to inspect (default 5).
        """
        if session.dataframe is None:
            return {"error": "Dataset not loaded."}

        head_df = session.dataframe.head(n)
        summary = {
            "result_id": session.next_result_id("inspect_data"),
            "n_rows_shown": len(head_df),
            "columns": list(session.dataframe.columns.astype(str)),
            "records": head_df.to_dict(orient="records")
        }
        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Checks for missing or empty text values in the working DataFrame."""
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "Dataset not loaded or 'text' column missing."}

        df = session.dataframe
        missing_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_indices = df[missing_mask].index.tolist()

        summary = {
            "result_id": session.next_result_id("check_missing"),
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices
        }
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for and optionally drops duplicate documents in the working DataFrame.

        duplicate_count reflects the total number of rows involved in duplicate
        groups (identified via keep=False), matching the drop behavior.

        Args:
            drop: If True, drops all rows involved in duplicate groups using keep=False.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "Dataset not loaded or 'text' column missing."}

        df = session.dataframe
        dups = df[df.duplicated(subset=["text"], keep=False)]
        dup_count = len(dups)

        dropped_count = 0
        if drop:
            initial_len = len(df)
            df.drop_duplicates(subset=["text"], keep=False, inplace=True)
            dropped_count = initial_len - len(df)
            _invalidate_derived_state()

        summary = {
            "result_id": session.next_result_id("check_duplicates"),
            "duplicate_count": dup_count,
            "dropped": drop,
            "dropped_count": dropped_count,
            "current_n_documents": len(df)
        }
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 4, random_state: int = 42) -> dict:
        """Subsamples the working DataFrame.

        Args:
            n: Number of samples to draw.
            random_state: Seed for reproducibility.
        """
        if session.dataframe is None:
            return {"error": "Dataset not loaded."}

        df = session.dataframe
        sample_n = min(n, len(df))
        sampled_df = df.sample(n=sample_n, random_state=random_state)
        session.dataframe = sampled_df
        _invalidate_derived_state()
        sampled_indices = sampled_df.index.tolist()

        summary = {
            "result_id": session.next_result_id("sample_data"),
            "n_sampled": sample_n,
            "random_state": random_state,
            "sampled_indices": sampled_indices,
            "records": sampled_df.to_dict(orient="records")
        }
        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Computes statistical summary of document text lengths and produces a category box plot.

        Matches Master's data-quality and text length analysis.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "Dataset not loaded or 'text' column missing."}

        df = session.dataframe.copy()
        df["text_length"] = df["text"].apply(lambda x: len(str(x)) if not pd.isna(x) else 0)

        overall_desc = df["text_length"].describe().to_dict()

        cat_col = "category_name" if "category_name" in df.columns else "category"
        per_category = {}
        for cat, group in df.groupby(cat_col):
            per_category[str(cat)] = group["text_length"].describe().to_dict()

        # Build box plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, x=cat_col, y="text_length", ax=ax)
        ax.set_title("Document Text Length by Category")
        plt.tight_layout()
        session.pending_figure = fig

        summary = {
            "result_id": session.next_result_id("describe_data"),
            "overall": {k: round(v, 4) for k, v in overall_desc.items() if isinstance(v, (int, float))},
            "per_category": {cat: {k: round(v, 4) for k, v in stats.items() if isinstance(v, (int, float))} for cat, stats in per_category.items()}
        }
        session.store_result("describe_data_tool", {}, summary, full_report=df)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool
    ]
