from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_main_workflow_has_history_writeback_guardrails():
    workflow = (ROOT / ".github/workflows/main.yml").read_text(encoding="utf-8")
    assert "contents: write" in workflow
    assert "group: send-emails-daily" in workflow
    assert "CLASSICS_ENABLED: true" in workflow
    assert "CLASSIC_HISTORY_WRITEBACK: true" in workflow
    assert "git add data/classic_recommendation_history.json" in workflow
    assert 'git checkout -B classic-history-writeback "origin/${TARGET_REF}"' in workflow


def test_test_workflow_keeps_classic_history_read_only():
    workflow = (ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
    assert "CLASSICS_ENABLED: false" in workflow
    assert "CLASSIC_HISTORY_WRITEBACK: false" in workflow
    assert "git add data/classic_recommendation_history.json" not in workflow
