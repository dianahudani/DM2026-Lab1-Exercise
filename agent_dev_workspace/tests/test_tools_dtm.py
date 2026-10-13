import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from scipy.sparse import issparse
from tools_dtm import make_tools

def test_build_dtm_and_term_frequency():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get student tools from tools_dtm
    tools = make_tools(session)
    build_dtm = next(t for t in tools if t.name == "build_dtm_tool")
    term_freq = next(t for t in tools if t.name == "term_frequency_tool")

    # 1. Test build_dtm_tool against known answer #4
    dtm_res = build_dtm.invoke({})
    assert dtm_res["n_documents"] == 8
    assert dtm_res["vocabulary_size"] == 5
    assert pytest.approx(dtm_res["sparsity_pct"], rel=1e-2) == 47.5
    assert "result_id" in dtm_res

    # Check that feature_matrix is sparse (issparse) and labels are set
    assert issparse(session.feature_matrix)
    assert session.labels is not None
    assert len(session.labels) == 8
    assert list(session.categories) == ["catA", "catB"]

    # 2. Test term_frequency_tool against known answer #14
    tf_res = term_freq.invoke({})
    term_freqs = tf_res["term_frequencies"]

    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    for term, expected_freq in expected_frequencies.items():
        assert term in term_freqs
        assert term_freqs[term] == expected_freq
    assert "result_id" in tf_res

def test_dtm_heatmap_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset & build DTM
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap = next(t for t in tools if t.name == "dtm_heatmap_tool")

    build_dtm.invoke({})

    # Test dtm_heatmap_tool with default parameters (n_terms=20, n_documents=20) against known answer #21
    default_res = dtm_heatmap.invoke({})
    assert default_res["n_documents"] == 8
    assert default_res["n_terms"] == 5
    assert len(default_res["matrix_slice"]) == 8
    assert len(default_res["matrix_slice"][0]) == 5
    assert default_res["document_indices"] == list(range(8))
    assert default_res["document_labels"] == ["catA", "catA", "catA", "catB", "catB", "catB", "catB", "catA"]
    assert default_res["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert default_res["matrix_slice"] == expected_full_matrix
    assert "result_id" in default_res
    assert session.pending_figure is not None

    # Test dtm_heatmap_tool with custom slice (n_terms=2, n_documents=3) against known answer #21
    heatmap_res = dtm_heatmap.invoke({"n_terms": 2, "n_documents": 3})
    assert heatmap_res["n_documents"] == 3
    assert heatmap_res["n_terms"] == 2
    assert heatmap_res["terms"] == ["alpha", "always"]
    assert heatmap_res["matrix_slice"] == [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
    assert "result_id" in heatmap_res
    assert session.pending_figure is not None
