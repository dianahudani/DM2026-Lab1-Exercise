from agent_pipeline.session_state import SessionState
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import tempfile
import os
from langchain_core.tools import tool

def make_tools(session: SessionState):
    class MinePatternsTool:
        name = "mine_patterns_tool"
        description = "Mines frequent patterns for a specific category using PAMI algorithms (FPGrowth, TopK, or MaxFPGrowth) after vocabulary filtering."
        
        def invoke(self, params: dict):
            category_name = params.get("category_name")
            filtering_method = params.get("filtering_method", "tfidf")
            algorithm = params.get("algorithm", "fpgrowth")
            min_sup = params.get("min_sup", 2)
            k = params.get("k", 1)
            
            if session.dataframe is None:
                raise ValueError("Dataset not loaded in session.")
            if "category_name" not in session.dataframe.columns:
                raise ValueError("Category name column not found in session dataframe.")
            if not category_name:
                raise ValueError("category_name must be specified.")
                
            # 1. Filter dataframe and build fresh category-local DTM
            df_cat = session.dataframe[session.dataframe["category_name"] == category_name].copy()
            if df_cat.empty:
                return {
                    "patterns": [],
                    "note": f"No documents found for category {category_name}"
                }
                
            text_column = "text" if "text" in df_cat.columns else df_cat.columns[0]
            corpus = df_cat[text_column].fillna("").astype(str).tolist()
            
            vectorizer = CountVectorizer(stop_words="english")
            try:
                X_counts = vectorizer.fit_transform(corpus)
            except ValueError:
                # Vocabulary is empty or all stop words
                return {
                    "patterns": [],
                    "note": "no terms survived filtering"
                }
                
            feature_names = vectorizer.get_feature_names_out()
            if len(feature_names) == 0:
                return {
                    "patterns": [],
                    "note": "no terms survived filtering"
                }
                
            X_dense = X_counts.toarray()
            
            # 2. Apply chosen filtering method (mirroring helpers/text_filtering_helpers.py percentiles)
            if filtering_method == "variance":
                variances = np.var(X_dense, axis=0, ddof=0)
                if len(variances) == 0:
                    surviving_terms = []
                else:
                    p5 = np.percentile(variances, 5)
                    p95 = np.percentile(variances, 95)
                    surviving_terms = [feat for feat, var in zip(feature_names, variances) if p5 <= var <= p95]
            elif filtering_method == "term_frequency":
                term_sums = np.sum(X_dense, axis=0)
                if len(term_sums) == 0:
                    surviving_terms = []
                else:
                    p5 = np.percentile(term_sums, 5)
                    p95 = np.percentile(term_sums, 95)
                    surviving_terms = [feat for feat, s in zip(feature_names, term_sums) if p5 <= s <= p95]
            elif filtering_method == "tfidf":
                transformer = TfidfTransformer()
                X_tfidf = transformer.fit_transform(X_counts).toarray()
                mean_tfidf = np.mean(X_tfidf, axis=0)
                if len(mean_tfidf) == 0:
                    surviving_terms = []
                else:
                    p20 = np.percentile(mean_tfidf, 20)
                    surviving_terms = [feat for feat, score in zip(feature_names, mean_tfidf) if score >= p20]
            else:
                raise ValueError(f"Unknown filtering_method: {filtering_method}")
                
            # 3. Handle zero terms survived
            if not surviving_terms:
                result_id = session.next_result_id("mine_patterns")
                summary = {
                    "result_id": result_id,
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "patterns": [],
                    "note": "no terms survived filtering"
                }
                session.store_result("mine_patterns_tool", params, summary)
                artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
                session.artifacts[artifact_key] = []
                return summary
                
            # Filter X_dense and feature_names to surviving terms
            surviving_indices = [list(feature_names).index(t) for t in surviving_terms]
            X_filtered = X_dense[:, surviving_indices]
            
            # Create a pandas DataFrame for the filtered terms where columns are the surviving terms
            df_filtered = pd.DataFrame(X_filtered, columns=list(surviving_terms))
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_out:
                tmp_out_path = tmp_out.name
                
            try:
                from PAMI.extras.convert import DF2DB
                obj = DF2DB.DF2DB(df_filtered)
                obj.convert2TransactionalDatabase(tmp_out_path, ">=", 1)
                
                patterns_list = []
                if algorithm == "fpgrowth":
                    from PAMI.frequentPattern.basic import FPGrowth
                    model = FPGrowth.FPGrowth(tmp_out_path, minSup=min_sup, sep="\t")
                    model.startMine()
                    patterns = model.getPatterns()
                    for pat, sup in patterns.items():
                        pat_str = list(pat) if isinstance(pat, tuple) else [pat]
                        patterns_list.append({"pattern": pat_str, "support": int(sup)})
                elif algorithm == "topk":
                    from PAMI.frequentPattern.topk import FAE
                    model = FAE.FAE(tmp_out_path, k=k, sep="\t")
                    model.startMine()
                    patterns = model.getPatterns()
                    for pat, sup in patterns.items():
                        pat_str = list(pat) if isinstance(pat, tuple) else [pat]
                        patterns_list.append({"pattern": pat_str, "support": int(sup)})
                elif algorithm == "maxfpgrowth":
                    from PAMI.frequentPattern.maximal import MaxFPGrowth
                    model = MaxFPGrowth.MaxFPGrowth(tmp_out_path, minSup=min_sup, sep="\t")
                    model.startMine()
                    patterns = model.getPatterns()
                    for pat, sup in patterns.items():
                        pat_str = list(pat) if isinstance(pat, tuple) else [pat]
                        patterns_list.append({"pattern": pat_str, "support": int(sup)})
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
            finally:
                if os.path.exists(tmp_out_path):
                    os.unlink(tmp_out_path)
                    
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": patterns_list
            }
            session.store_result("mine_patterns_tool", params, summary)
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = patterns_list
            return summary

    implementation = MinePatternsTool()

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: int = 2,
        k: int = 1,
    ) -> dict:
        """Mine frequent patterns for a category using a PAMI algorithm.

        Args:
            category_name: Category whose documents should be mined.
            filtering_method: Vocabulary filter: "variance", "term_frequency", or "tfidf".
            algorithm: Mining algorithm: "fpgrowth", "topk", or "maxfpgrowth".
            min_sup: Minimum support for FPGrowth or MaxFPGrowth.
            k: Number of patterns for TopK mining.
        """
        return implementation.invoke({
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        })

    return [mine_patterns_tool]
