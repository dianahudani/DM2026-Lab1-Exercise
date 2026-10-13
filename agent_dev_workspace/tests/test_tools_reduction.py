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

def test_reduce_dimensions_tool_n_components_1(session_with_dtm):
    reduction_tools = make_reduction_tools(session_with_dtm)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "pca", "n_components": 1, "random_state": 42})
    assert res["n_components"] == 1
    assert res["shape"] == [8, 1]
    
    fig = session_with_dtm.pending_figure
    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Component 1"
    assert ax.get_ylabel() == ""
    assert "1D" in ax.get_title()
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "catA" in legend_texts
    assert "catB" in legend_texts

def test_reduce_dimensions_tool_n_components_3(session_with_dtm):
    reduction_tools = make_reduction_tools(session_with_dtm)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "pca", "n_components": 3, "random_state": 42})
    assert res["n_components"] == 3
    assert res["shape"] == [8, 3]
    
    fig = session_with_dtm.pending_figure
    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Component 1"
    assert ax.get_ylabel() == "Component 2"
    assert "Components 1 & 2" in ax.get_title()
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "catA" in legend_texts
    assert "catB" in legend_texts

def test_reduce_dimensions_plot_category_colors(session_with_dtm):
    reduction_tools = make_reduction_tools(session_with_dtm)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    reduce_tool.invoke({"method": "pca", "n_components": 2, "random_state": 42})
    fig = session_with_dtm.pending_figure
    assert fig is not None
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "catA" in legend_texts
    assert "catB" in legend_texts

def test_reduce_dimensions_tool_tsne(session_with_dtm):
    reduction_tools = make_reduction_tools(session_with_dtm)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })
    
    assert res["method"] == "tsne"
    assert res["n_components"] == 2
    assert res["shape"] == [8, 2]
    
    actual_coords = session_with_dtm.artifacts["reduced_tsne"]
    assert actual_coords.shape == (8, 2)
    
    # Check against known answer #8 t-SNE Dim 1 and Dim 2 values with tolerance
    expected_tsne = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ])
    assert np.allclose(actual_coords, expected_tsne, atol=1e-3)
    assert session_with_dtm.pending_figure is not None

def test_reduce_dimensions_tool_umap(session_with_dtm):
    reduction_tools = make_reduction_tools(session_with_dtm)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })
    
    assert res["method"] == "umap"
    assert res["n_components"] == 2
    assert res["shape"] == [8, 2]
    
    # Check against known answer #9 UMAP Dim 1 and Dim 2 values with tolerance
    expected_umap = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])
    
    actual_coords = session_with_dtm.artifacts["reduced_umap"]
    assert np.allclose(actual_coords, expected_umap, atol=1e-3)
    assert session_with_dtm.pending_figure is not None
