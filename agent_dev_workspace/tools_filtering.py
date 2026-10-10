from agent_pipeline.session_state import SessionState
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def make_tools(session: SessionState):
    class VarianceFilterTool:
        name = "variance_filter_tool"
        description = "Filters out features with variance below a given threshold."
        
        def invoke(self, params: dict):
            threshold = params.get("threshold", 0.0)
            X = session.feature_matrix
            feature_names = session.feature_names
            
            if X is None or feature_names is None:
                raise ValueError("Feature matrix or feature names not found in session.")
            
            variances = np.var(X, axis=0, ddof=0)
            variances_dict = {feat: float(var) for feat, var in zip(feature_names, variances)}
            
            kept_features = [feat for feat, var in variances_dict.items() if var >= threshold]
            
            kept_indices = [feature_names.index(feat) for feat in kept_features]
            if X.ndim == 2:
                session.feature_matrix = X[:, kept_indices]
            session.feature_names = kept_features
            
            df_report = pd.DataFrame({
                "term": feature_names,
                "variance": variances
            }).sort_values(by="variance", ascending=False).reset_index(drop=True)
            
            result_id = session.next_result_id("variance_filter")
            summary = {
                "result_id": result_id,
                "kept_count": len(kept_features),
                "kept_features": kept_features
            }
            session.store_result("variance_filter_tool", params, summary, full_report=df_report)
            return summary

    class PearsonFilterTool:
        name = "pearson_filter_tool"
        description = "Computes Pearson correlation between term counts and a target class."
        
        def invoke(self, params: dict):
            target_class = params.get("target_class")
            threshold = params.get("threshold", 0.0)
            
            if session.feature_matrix is None or session.feature_names is None:
                raise ValueError("Feature matrix not found in session. Build DTM first.")
            if session.dataframe is None or "category_name" not in session.dataframe.columns:
                raise ValueError("Session dataframe or category_name not found.")
                
            X = session.feature_matrix
            feature_names = session.feature_names
            labels = session.dataframe["category_name"].values
            
            if target_class is None:
                target_class = session.categories[0] if session.categories else labels[0]
                
            y = (labels == target_class).astype(int)
            
            corrs = []
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                if np.std(col_data) == 0 or np.std(y) == 0:
                    r = 0.0
                else:
                    r, _ = pearsonr(col_data, y)
                    if np.isnan(r):
                        r = 0.0
                corrs.append(float(r))
                
            df_report = pd.DataFrame({
                "term": feature_names,
                "pearson_r": corrs
            }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)
            
            kept_features = [row["term"] for _, row in df_report.iterrows() if abs(row["pearson_r"]) >= threshold]
            
            result_id = session.next_result_id("pearson_filter")
            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "kept_features": kept_features
            }
            session.store_result("pearson_filter_tool", params, summary, full_report=df_report)
            return summary

    class SpearmanFilterTool:
        name = "spearman_filter_tool"
        description = "Computes Spearman rank correlation between term counts and a target class."
        
        def invoke(self, params: dict):
            target_class = params.get("target_class")
            threshold = params.get("threshold", 0.0)
            
            if session.feature_matrix is None or session.feature_names is None:
                raise ValueError("Feature matrix not found in session. Build DTM first.")
            if session.dataframe is None or "category_name" not in session.dataframe.columns:
                raise ValueError("Session dataframe or category_name not found.")
                
            X = session.feature_matrix
            feature_names = session.feature_names
            labels = session.dataframe["category_name"].values
            
            if target_class is None:
                target_class = session.categories[0] if session.categories else labels[0]
                
            y = (labels == target_class).astype(int)
            
            corrs = []
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                if np.std(col_data) == 0 or np.std(y) == 0:
                    r = 0.0
                else:
                    r, _ = spearmanr(col_data, y)
                    if np.isnan(r):
                        r = 0.0
                corrs.append(float(r))
                
            df_report = pd.DataFrame({
                "term": feature_names,
                "spearman_r": corrs
            }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)
            
            kept_features = [row["term"] for _, row in df_report.iterrows() if abs(row["spearman_r"]) >= threshold]
            
            result_id = session.next_result_id("spearman_filter")
            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "kept_features": kept_features
            }
            session.store_result("spearman_filter_tool", params, summary, full_report=df_report)
            return summary

    return [VarianceFilterTool(), PearsonFilterTool(), SpearmanFilterTool()]
