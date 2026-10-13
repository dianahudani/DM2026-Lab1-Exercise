import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools
from tools_dtm import make_tools as make_dtm_tools

def test_cosine_similarity_and_correlation():
    session = SessionState()
    
    # 1. Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_ds_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_ds_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 2. Build DTM using student's build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    # 3. Test cosine_similarity_tool (Known Answer #16)
    exploration_tools = make_tools(session)
    cos_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    # Doc 0 vs Doc 1 (both catA, near-identical text): expected ~0.9847
    res_0_1 = cos_sim_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Doc 0 vs Doc 3 (catA vs catB, share only "always"): expected ~0.0909
    res_0_3 = cos_sim_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)
    
    # Doc 3 vs Doc 4 (both catB, near-identical text): expected ~0.9847
    res_3_4 = cos_sim_tool.invoke({"doc_idx_1": 3, "doc_idx_2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Doc 0 vs Doc 7 (catA vs empty text row): expected 0.0
    res_0_7 = cos_sim_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 7})
    assert res_0_7["similarity"] == 0.0

    # 4. Test feature_correlation_matrix_tool (Known Answer #19)
    corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    res_corr = corr_tool.invoke({})
    
    # Check that correlation matrix contains expected terms and values
    matrix = res_corr["correlation_matrix"]
    terms = res_corr["terms"]
    
    # Terms in descending variance order per Known Answer #19
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert terms == expected_terms
    
    # Verify all entries in 5x5 correlation matrix against Known Answer #19
    expected_corr = {
        "alpha": {"alpha": 1.0000, "gamma": -0.5718, "delta": -0.6882, "beta": 0.8885, "always": 0.2601},
        "gamma": {"alpha": -0.5718, "gamma": 1.0000, "delta": 0.8307, "beta": -0.6435, "always": 0.3140},
        "delta": {"alpha": -0.6882, "gamma": 0.8307, "delta": 1.0000, "beta": -0.7746, "always": 0.3780},
        "beta": {"alpha": 0.8885, "gamma": -0.6435, "delta": -0.7746, "beta": 1.0000, "always": 0.2928},
        "always": {"alpha": 0.2601, "gamma": 0.3140, "delta": 0.3780, "beta": 0.2928, "always": 1.0000},
    }
    for t1 in expected_terms:
        for t2 in expected_terms:
            assert matrix[t1][t2] == pytest.approx(expected_corr[t1][t2], abs=1e-3)

    # Check that pending_figure is set to the correlation heatmap
    assert session.pending_figure is not None
