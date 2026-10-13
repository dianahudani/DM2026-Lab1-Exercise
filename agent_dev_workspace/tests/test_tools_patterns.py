import pytest
import pandas as pd
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
    for k_val in [1, 2, 3]:
        res_catA_topk = mine_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_catA_topk["patterns"] == [{"pattern": ["alpha"], "support": 3}]
        
        res_catB_topk = mine_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_catB_topk["patterns"] == [{"pattern": ["gamma"], "support": 4}]
    
    # Test 4: maxfpgrowth algorithm (Known answer 20)
    # MaxFPGrowth on single surviving term returns zero patterns for both categories across min_sup=1 and 4
    for cat in ["catA", "catB"]:
        for min_sup in [1, 4]:
            res_max = mine_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_sup
            })
            assert res_max["patterns"] == []

def test_mine_patterns_unknown_category():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    pattern_tools = make_tools(session)
    mine_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")
    
    res = mine_tool.invoke({
        "category_name": "unknown_cat",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    
    result_id = res["result_id"]
    assert result_id.startswith("mine_patterns_")
    assert res["patterns"] == []
    assert res["note"] == "No documents found for category unknown_cat"
    assert result_id in session.results_store
    assert session.results_store[result_id]["record"].summary == res
    
    expected_artifact_key = "patterns_unknown_cat_tfidf_fpgrowth"
    assert expected_artifact_key in session.artifacts
    assert session.artifacts[expected_artifact_key] == []

def test_mine_patterns_empty_vocabulary_category():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Append rows with all-stopword / empty text using fixture columns
    extra_rows = pd.DataFrame({
        "text": ["the and a in on", "", "  "],
        "category_name": ["cat_stopwords", "cat_stopwords", "cat_stopwords"],
        "category": [9, 9, 9]
    })
    session.dataframe = pd.concat([session.dataframe, extra_rows], ignore_index=True)
    
    pattern_tools = make_tools(session)
    mine_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")
    
    res = mine_tool.invoke({
        "category_name": "cat_stopwords",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    
    result_id = res["result_id"]
    assert result_id.startswith("mine_patterns_")
    assert res["patterns"] == []
    assert res["note"] == "no terms survived filtering"
    assert result_id in session.results_store
    assert session.results_store[result_id]["record"].summary == res
    
    expected_artifact_key = "patterns_cat_stopwords_tfidf_fpgrowth"
    assert expected_artifact_key in session.artifacts
    assert session.artifacts[expected_artifact_key] == []
