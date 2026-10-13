import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools
from tools_dtm import make_tools as make_dtm_tools

@pytest.fixture
def session_with_data():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def test_tokenize_tool(session_with_data):
    tools = make_tools(session_with_data)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    
    result = tokenize_tool.invoke({})
    
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    tokens_from_artifacts = session_with_data.artifacts["tokens"]
    tokens_from_df = session_with_data.dataframe["unigrams"].tolist()
    
    assert len(tokens_from_artifacts) == 8
    assert tokens_from_artifacts == expected_tokens
    assert len(tokens_from_df) == 8
    assert tokens_from_df == expected_tokens

def test_list_files_tool(session_with_data):
    tools = make_tools(session_with_data)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")
    
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert result.get("directories") == []
    assert result.get("files") == ["Reddit-stock-sentiment.csv"]

def test_inspect_data_tool(session_with_data):
    tools = make_tools(session_with_data)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    
    result = inspect_data_tool.invoke({"n": 5})
    assert isinstance(result, dict)

def test_check_missing_tool(session_with_data):
    tools = make_tools(session_with_data)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")
    
    result = check_missing_tool.invoke({})
    assert result.get("missing_count", result.get("n_missing", 1)) == 1

def test_check_duplicates_tool(session_with_data):
    tools = make_tools(session_with_data)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")
    
    result = check_duplicates_tool.invoke({})
    # duplicate_count is 2: total number of rows involved in duplicate groups (keep=False),
    # matching the 2 rows dropped when drop=True.
    assert result.get("duplicate_count") == 2
    
    result_drop = check_duplicates_tool.invoke({"drop": True})
    assert result_drop.get("current_n_documents") == 6
    assert result_drop.get("dropped_count") == 2

def test_sample_data_tool(session_with_data):
    tools = make_tools(session_with_data)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")
    
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})
    sampled_indices = result.get("indices", result.get("sampled_indices", []))
    assert sampled_indices == [1, 5, 0, 7]

def test_describe_data_tool(session_with_data):
    tools = make_tools(session_with_data)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")
    
    result = describe_data_tool.invoke({})
    
    # Assert Known Answer #18: overall stats
    overall = result.get("overall", {})
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0.0, abs=1e-3)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == pytest.approx(30.0, abs=1e-3)
    
    # Assert Known Answer #18: per-category stats
    per_cat = result.get("per_category", {})
    catA = per_cat["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == pytest.approx(0.0, abs=1e-3)
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == pytest.approx(29.0, abs=1e-3)
    
    catB = per_cat["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == pytest.approx(18.0, abs=1e-3)
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == pytest.approx(30.0, abs=1e-3)
    
    # Assert box plot figure is set
    assert session_with_data.pending_figure is not None

def test_sample_data_invalidation():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM to populate derived state
    dtm_tool = next(t for t in make_dtm_tools(session) if t.name == "build_dtm_tool")
    dtm_res = dtm_tool.invoke({})
    dtm_result_id = dtm_res["result_id"]
    
    # Seed representative artifacts and pending figure
    session.artifacts["tokens"] = [["some", "token"]]
    session.artifacts["binary_labels"] = np.ones((8, 2))
    session.artifacts["reduced_pca"] = np.ones((8, 2))
    session.artifacts["patterns_catA_tfidf_fpgrowth"] = [{"pattern": ["alpha"]}]
    session.pending_figure = object()
    
    # Confirm derived state exists before mutation
    assert session.feature_matrix is not None
    assert session.feature_names is not None
    assert session.labels is not None
    assert len(session.categories) > 0
    assert "count_vectorizer" in session.artifacts
    assert session.pending_figure is not None
    
    # Exercise sample_data_tool
    tools = make_tools(session)
    sample_tool = next(t for t in tools if t.name == "sample_data_tool")
    res = sample_tool.invoke({"n": 4, "random_state": 42})
    
    # Assert fixture indices
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4
    
    # Assert derived state is cleared
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.labels is None
    assert session.categories == []
    assert session.pending_figure is None
    assert "count_vectorizer" not in session.artifacts
    assert "tokens" not in session.artifacts
    assert "binary_labels" not in session.artifacts
    assert "reduced_pca" not in session.artifacts
    assert "patterns_catA_tfidf_fpgrowth" not in session.artifacts
    
    # Confirm historical DTM result remains in results_store
    assert dtm_result_id in session.results_store

def test_check_duplicates_invalidation():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM to populate derived state
    dtm_tool = next(t for t in make_dtm_tools(session) if t.name == "build_dtm_tool")
    dtm_res = dtm_tool.invoke({})
    dtm_result_id = dtm_res["result_id"]
    
    # Seed representative artifacts and pending figure
    session.artifacts["tokens"] = [["some", "token"]]
    session.artifacts["binary_labels"] = np.ones((8, 2))
    session.artifacts["reduced_pca"] = np.ones((8, 2))
    session.artifacts["patterns_catA_tfidf_fpgrowth"] = [{"pattern": ["alpha"]}]
    session.pending_figure = object()
    
    # Confirm derived state exists before mutation
    assert session.feature_matrix is not None
    assert session.feature_names is not None
    assert session.labels is not None
    assert len(session.categories) > 0
    assert "count_vectorizer" in session.artifacts
    assert session.pending_figure is not None
    
    # Exercise check_duplicates_tool with drop=True
    tools = make_tools(session)
    dup_tool = next(t for t in tools if t.name == "check_duplicates_tool")
    res = dup_tool.invoke({"drop": True})
    
    # Assert six documents remain and two rows were dropped
    assert res["current_n_documents"] == 6
    assert res["dropped_count"] == 2
    assert len(session.dataframe) == 6
    
    # Assert derived state is cleared
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.labels is None
    assert session.categories == []
    assert session.pending_figure is None
    assert "count_vectorizer" not in session.artifacts
    assert "tokens" not in session.artifacts
    assert "binary_labels" not in session.artifacts
    assert "reduced_pca" not in session.artifacts
    assert "patterns_catA_tfidf_fpgrowth" not in session.artifacts
    
    # Confirm historical DTM result remains in results_store
    assert dtm_result_id in session.results_store
