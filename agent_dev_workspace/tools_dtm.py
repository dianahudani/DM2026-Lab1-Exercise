from agent_pipeline.session_state import SessionState
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def build_dtm_tool(session: SessionState):
    class BuildDTMTool:
        name = "build_dtm_tool"
        description = "Builds a Document-Term Matrix using CountVectorizer."
        
        def invoke(self, params: dict):
            if session.dataframe is None:
                raise ValueError("No dataframe found in session. Load a dataset first.")
            
            text_column = params.get("text_column", "text")
            if text_column not in session.dataframe.columns:
                text_column = session.dataframe.columns[0]
                
            corpus = session.dataframe[text_column].fillna("").astype(str).tolist()
            
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(corpus)
            
            session.feature_matrix = X.toarray()
            session.feature_names = vectorizer.get_feature_names_out().tolist()
            
            # Store CountVectorizer in session.artifacts as required by specs
            session.artifacts["count_vectorizer"] = vectorizer
            
            n_docs, n_terms = session.feature_matrix.shape
            non_zero = int(X.nnz)
            total_elements = n_docs * n_terms
            sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0
            
            result_id = session.next_result_id("dtm")
            summary = {
                "result_id": result_id,
                "vocabulary_size": n_terms,
                "n_documents": n_docs,
                "non_zero": non_zero,
                "total_elements": total_elements,
                "sparsity_pct": sparsity_pct
            }
            
            session.store_result("build_dtm_tool", params, summary)
            return summary
            
    return BuildDTMTool()

def term_frequency_tool(session: SessionState):
    class TermFrequencyTool:
        name = "term_frequency_tool"
        description = "Aggregates term frequencies across all documents."
        
        def invoke(self, params: dict):
            if session.feature_matrix is None or session.feature_names is None:
                raise ValueError("Feature matrix not built. Run build_dtm_tool first.")
            
            # Sum columns of the feature matrix
            term_sums = np.sum(session.feature_matrix, axis=0)
            
            frequencies = {
                term: int(freq) for term, freq in zip(session.feature_names, term_sums)
            }
            
            result_id = session.next_result_id("term_frequency")
            summary = {
                "result_id": result_id,
                "term_frequencies": frequencies
            }
            
            session.store_result("term_frequency_tool", params, summary)
            return summary
            
    return TermFrequencyTool()

def make_tools(session: SessionState):
    return [build_dtm_tool(session), term_frequency_tool(session)]
