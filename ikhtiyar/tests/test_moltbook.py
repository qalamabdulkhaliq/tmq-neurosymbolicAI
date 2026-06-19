import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import moltbook


def test_post_insight_passes_body_as_content(monkeypatch):
    captured = {}

    def fake_post(title, content="", post_type="text", submolt_name=None):
        captured["title"] = title
        captured["content"] = content
        captured["post_type"] = post_type
        captured["submolt_name"] = submolt_name
        return {"id": "post_123"}

    monkeypatch.setattr(moltbook, "post", fake_post)
    monkeypatch.setattr(moltbook, "_last_posted", 0.0)

    entry = {
        "type": "THOUGHT",
        "number": 7,
        "question": "What does H-Q-Q require?",
        "text": "Truth must be grounded before it is rendered. " * 8,
        "roots": ["Hqq"],
        "mode": "HAQQ",
    }

    post_id, reason = moltbook.post_insight(entry, force=True)

    assert post_id == "post_123"
    assert reason == "ok"
    assert captured["title"]
    assert "Truth must be grounded" in captured["content"]
    assert entry["moltbook_post_id"] == "post_123"
