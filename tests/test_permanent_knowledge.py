from pathlib import Path

from dopa_coder_n1.model.permanent_knowledge import (
    FailureDrivenKnowledgeLearner,
    PermanentKnowledgeBase,
)


def test_pkb_create_query_update_merge_delete_cycle(tmp_path):
    pkb = PermanentKnowledgeBase(tmp_path, max_files=3, max_file_bytes=20_000)
    first = pkb.create(
        rule="Use erase-remove idiom when deleting values from C++ vector.",
        domain="C++ containers",
        importance=0.9,
        created_from="user_feedback",
    )
    second = pkb.create(
        rule="When removing vector elements, avoid invalidated iterators.",
        domain="C++ containers",
        importance=0.7,
        created_from="self_induction",
    )

    results = pkb.query("C++ vector erase remove", top_k=2)
    assert results[0].id == first.id
    assert results[0].access_count == 1

    updated = pkb.update(first.id, add_positive="std::remove followed by erase")
    assert "std::remove" in updated.positive_examples[-1]

    merged = pkb.merge(first.id, second.id)
    assert second.id in merged.dependencies
    assert not (tmp_path / f"{second.id}.json").exists()

    assert pkb.delete(first.id)
    assert not (tmp_path / f"{first.id}.json").exists()


def test_pkb_prunes_to_file_limit_by_importance(tmp_path):
    pkb = PermanentKnowledgeBase(tmp_path, max_files=2, max_file_bytes=20_000)
    low = pkb.create(rule="low importance note", domain="misc", importance=0.1, created_from="self_induction")
    high = pkb.create(rule="important boundary condition", domain="testing", importance=0.95, created_from="user_feedback")
    mid = pkb.create(rule="medium note", domain="misc", importance=0.5, created_from="self_induction")

    report = pkb.prune_to_limit()
    ids = {point.id for point in pkb.list_points()}

    assert report["deleted_count"] == 1
    assert low.id not in ids
    assert high.id in ids
    assert mid.id in ids


def test_failure_driven_learner_creates_knowledge_point(tmp_path):
    pkb = PermanentKnowledgeBase(tmp_path, max_files=10)
    learner = FailureDrivenKnowledgeLearner(pkb)

    point = learner.learn_from_failure(
        user_requirement="Write C++ that deletes negative numbers from vector.",
        failed_output="for loop erases while incrementing iterator",
        corrected_output="use erase(remove_if(v.begin(), v.end(), pred), v.end())",
    )

    assert point.created_from == "user_feedback"
    assert "erase" in point.rule.lower()
    assert len(pkb.query("delete negative numbers vector", top_k=1)) == 1
