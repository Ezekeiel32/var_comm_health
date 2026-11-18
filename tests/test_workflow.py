"""Tests for the deterministic parts of the workflow. No LLM and no API key are needed."""
import pytest

from var_comm_health.workflow import (
    CommunicationHealthState,
    _parse_analysis_result,
    aggregate_health_node,
    preprocess_node,
)

DIMENSIONS = ["clarity", "completeness", "correctness", "courtesy", "audience", "timeliness"]


def test_parse_plain_json():
    result = _parse_analysis_result('{"score": 0.8, "reasoning": "clear"}', "clarity")
    assert result == {"score": 0.8, "reasoning": "clear"}


def test_parse_fenced_json_with_a_dimension_key():
    result = _parse_analysis_result('```json\n{"clarity_score": 0.9, "reasoning": "ok"}\n```', "clarity")
    assert result["score"] == 0.9
    assert result["reasoning"] == "ok"


def test_parse_invalid_json_falls_back_to_a_neutral_score():
    result = _parse_analysis_result("this is not json", "clarity")
    assert result["score"] == 0.5
    assert "failed" in result["reasoning"].lower()


def test_aggregate_averages_the_six_dimensions():
    scores = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    state = CommunicationHealthState(**{f"{d}_score": {"score": s, "reasoning": ""} for d, s in zip(DIMENSIONS, scores)})
    out = aggregate_health_node(state)
    assert out["status"] == "aggregated"
    assert out["aggregated_health"]["overall"] == pytest.approx(0.5)
    assert out["aggregated_health"]["dimensions"] == dict(zip(DIMENSIONS, scores))
    assert out["aggregated_health"]["total_dimensions"] == 6


def test_aggregate_waits_until_every_dimension_is_scored():
    state = CommunicationHealthState(**{f"{d}_score": {"score": 0.7, "reasoning": ""} for d in DIMENSIONS[:-1]})
    assert aggregate_health_node(state) == {}


def test_preprocess_detects_an_email_thread():
    emails = [
        {"from": "dana@example.com", "subject": "Launch date", "body": "Can we ship on Monday?", "date": "2025-01-06T09:00:00"},
        {"from": "omer@example.com", "subject": "Re: Launch date", "body": "Yes, Monday works.", "date": "2025-01-06T09:30:00"},
    ]
    out = preprocess_node(CommunicationHealthState(emails=emails))
    assert out["status"] == "preprocessed"
    assert out["input_type"] == "emails"
    assert "Can we ship on Monday?" in out["preprocessed_content"]
    assert "Yes, Monday works." in out["preprocessed_content"]
    assert set(out["context"]["participants"]) == {"dana@example.com", "omer@example.com"}


def test_preprocess_detects_a_transcript_and_its_speakers():
    transcript = "Dana: Can we ship on Monday?\nOmer: Yes, Monday works."
    out = preprocess_node(CommunicationHealthState(transcript=transcript))
    assert out["status"] == "preprocessed"
    assert out["input_type"] == "transcript"
    assert out["preprocessed_content"] == transcript
    assert set(out["context"]["speakers"]) == {"Dana", "Omer"}


def test_preprocess_without_input_reports_an_error():
    out = preprocess_node(CommunicationHealthState())
    assert out["status"] == "error"
    assert "No input provided" in out["error"]
