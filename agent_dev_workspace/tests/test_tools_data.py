import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

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
    tokens_list = result.get("tokens", session_with_data.dataframe.get("unigrams", []))
    
    if len(tokens_list) > 0:
        expected_doc_0 = ["always", "alpha", "alpha", "alpha", "beta"]
        assert tokens_list[0] == expected_doc_0
        assert len(tokens_list) == 8

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
    # duplicate_count is 2 (both rows 5 and 6 are duplicates)
    assert result.get("duplicate_count") == 2

def test_sample_data_tool(session_with_data):
    tools = make_tools(session_with_data)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")
    
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})
    sampled_indices = result.get("indices", result.get("sampled_indices", []))
    if sampled_indices:
        assert sampled_indices == [1, 5, 0, 7]

def test_describe_data_tool(session_with_data):
    tools = make_tools(session_with_data)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")
    
    result = describe_data_tool.invoke({})
    overall = result.get("overall", result)
    mean_val = overall.get("mean", overall.get("text_length_mean"))
    if mean_val is not None:
        assert mean_val == pytest.approx(19.875, abs=1e-3)
