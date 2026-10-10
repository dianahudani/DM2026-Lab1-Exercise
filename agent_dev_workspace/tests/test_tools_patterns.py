import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def test_mine_patterns_tool_fixture():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    pattern_tools = make_tools(session)
    mine_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")
    
    # Test 1: variance and term_frequency filtering keep zero terms
    res_var_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var_catA["patterns"] == []
    assert res_var_catA.get("note") == "no terms survived filtering"
    
    res_tf_catB = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf_catB["patterns"] == []
    assert res_tf_catB.get("note") == "no terms survived filtering"
    
    # Test 2: tfidf filtering + fpgrowth (Known answer 17)
    res_catA_fp = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    # Should find {alpha} with support 3
    patterns_catA = res_catA_fp["patterns"]
    assert len(patterns_catA) > 0
    alpha_found = any("alpha" in p["pattern"] and p["support"] == 3 for p in patterns_catA)
    assert alpha_found
    
    res_catB_fp = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    patterns_catB = res_catB_fp["patterns"]
    assert len(patterns_catB) > 0
    gamma_found = any("gamma" in p["pattern"] and p["support"] == 4 for p in patterns_catB)
    assert gamma_found
    
    # Test 3: topk algorithm (Known answer 20)
    res_catA_topk = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_catA_topk["patterns"]) >= 1
    assert any("alpha" in p["pattern"] and p["support"] == 3 for p in res_catA_topk["patterns"])
    
    # Test 4: maxfpgrowth algorithm (Known answer 20)
    # MaxFPGrowth on single surviving term returns zero patterns
    res_catA_max = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_catA_max["patterns"] == []
