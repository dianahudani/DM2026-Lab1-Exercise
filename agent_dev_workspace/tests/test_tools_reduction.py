import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

@pytest.fixture(scope="function")
def session_with_dtm():
    session = SessionState()
    # Load dataset using workspace loader
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM so reduce_dimensions_tool has a feature matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    return session

def test_binarize_labels_tool(session_with_dtm):
    reduction_tools = make_reduction_tools(session_with_dtm)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    res = binarize_tool.invoke({"target_column": "category_name"})
    assert res["shape"] == [8, 2]
    assert "catA" in res["unique_classes"]
    assert "catB" in res["unique_classes"]
    assert "binary_labels" in session_with_dtm.artifacts
    
    binary_matrix = session_with_dtm.artifacts["binary_labels"]
    assert binary_matrix.shape == (8, 2)

def test_reduce_dimensions_tool_pca(session_with_dtm):
    reduction_tools = make_reduction_tools(session_with_dtm)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({
        "method": "pca",
        "n_components": 2,
        "random_state": 42
    })
    
    assert res["method"] == "pca"
    assert res["n_components"] == 2
    assert res["shape"] == [8, 2]
    
    # Check against known answer #7 PC1 and PC2 values with tolerance
    expected_pca = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    
    actual_coords = session_with_dtm.artifacts["reduced_pca"]
    assert np.allclose(actual_coords, expected_pca, atol=1e-3)
    
    # Check explained variance ratio known answer [0.7532, 0.1965]
    expected_var = [0.7532, 0.1965]
    assert np.allclose(res["explained_variance_ratio"], expected_var, atol=1e-3)
    
    # Confirm plotting requirement: pending_figure is set
    assert session_with_dtm.pending_figure is not None
