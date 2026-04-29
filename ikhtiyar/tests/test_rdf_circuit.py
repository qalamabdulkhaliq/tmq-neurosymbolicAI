import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.epistemic_clusters import get_cluster, get_cluster_uri

def test_certainty_cluster():
    assert get_cluster("علم") == "certainty"
    assert get_cluster("يقن") == "certainty"

def test_conjecture_cluster():
    assert get_cluster("ظن") == "conjecture"

def test_unknown_root_returns_none():
    assert get_cluster("xyz") is None

def test_cluster_uri_format():
    uri = get_cluster_uri("علم")
    assert uri == "http://quran.data/epistemic/certainty"

def test_cluster_uri_none_for_unknown():
    assert get_cluster_uri("xyz") is None
