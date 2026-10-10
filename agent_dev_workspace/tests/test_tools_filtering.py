import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools
from tools_dtm import build_dtm_tool

def test_variance_filter_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tool = build_dtm_tool(session)
    dtm_tool.invoke({})
    
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    filter_result = variance_filter_tool.invoke({"threshold": 0.15})
    
    assert "always" not in filter_result["kept_features"]
    for term in ["alpha", "beta", "delta", "gamma"] :
        assert term in filter_result["kept_features"]

def test_pearson_and_spearman_filter_tools():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tool = build_dtm_tool(session)
    dtm_tool.invoke({})
    
    filtering_tools = make_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    spearman_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")
    
    pearson_res = pearson_tool.invoke({"target_class": "catB"})
    spearman_res = spearman_tool.invoke({"target_class": "catB"})
    
    # Retrieve full reports from session results store using result_id
    pearson_record = session.results_store.get(pearson_res["result_id"])
    spearman_record = session.results_store.get(spearman_res["result_id"])
    
    pearson_report = pearson_record.get("full_report") if pearson_record else None
    spearman_report = spearman_record.get("full_report") if spearman_record else None
    
    assert pearson_report is not None
    assert spearman_report is not None
    
    # Check known answer 6 values from TEST_FIXTURE.md
    # Known answers:
    # alpha: Pearson -0.6882, Spearman -0.7500
    # always: Pearson 0.3780, Spearman 0.3780
    # beta: Pearson -0.7746, Spearman -0.7746
    # delta: Pearson 1.0000, Spearman 1.0000
    # gamma: Pearson 0.8307, Spearman 0.9363
    
    pearson_dict = dict(zip(pearson_report["term"], pearson_report["pearson_r"]))
    spearman_dict = dict(zip(spearman_report["term"], spearman_report["spearman_r"]))
    
    assert pearson_dict["delta"] == pytest.approx(1.0000, rel=1e-3, abs=1e-3)
    assert spearman_dict["delta"] == pytest.approx(1.0000, rel=1e-3, abs=1e-3)
    
    assert pearson_dict["alpha"] == pytest.approx(-0.6882, rel=1e-3, abs=1e-3)
    assert spearman_dict["alpha"] == pytest.approx(-0.7500, rel=1e-3, abs=1e-3)
    
    assert pearson_dict["beta"] == pytest.approx(-0.7746, rel=1e-3, abs=1e-3)
    assert spearman_dict["beta"] == pytest.approx(-0.7746, rel=1e-3, abs=1e-3)
    
    assert pearson_dict["gamma"] == pytest.approx(0.8307, rel=1e-3, abs=1e-3)
    assert spearman_dict["gamma"] == pytest.approx(0.9363, rel=1e-3, abs=1e-3)
