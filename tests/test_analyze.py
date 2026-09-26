"""Analysis: schema validity, deterministic caps, and consistency guarantees."""
from __future__ import annotations

import pytest

from app.analyze import NO_ASK_CAP, NO_BUDGET_CAP, SCHEMA, TIMELINE_CAP, analyze_lead
from app.llm import gemini_schema
from app.models import Analysis, LeadIntake
from tests.conftest import ANALYSIS_PAYLOAD

INTAKE = LeadIntake(name="Rahul Mehta", location="Whitefield",
                    property_requirement="3BHK", budget="1.4 crore",
                    timeline="2 months", message="Pre-approved, visiting Saturday.")


# ---- schema portability ----------------------------------------------------- #
def test_schema_is_groq_strict_compatible():
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == set(SCHEMA["properties"])


def test_gemini_schema_strips_unsupported_keys():
    """Regression: Gemini 400s on additionalProperties, which Groq requires."""
    import json
    g = gemini_schema(SCHEMA)
    assert "additionalProperties" not in json.dumps(g)
    assert g["properties"]["intent"]["enum"]          # enums survive
    assert g["required"]                               # required survives
    assert g["properties"]["key_requirements"]["items"] == {"type": "string"}


def test_every_schema_field_maps_to_the_model_or_a_cap_input():
    cap_inputs = {"timeline_bucket", "has_budget", "explicit_ask"}
    assert set(SCHEMA["properties"]) - cap_inputs <= set(Analysis.model_fields)


# ---- happy path ------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_analysis_parses_and_keeps_all_six_required_outputs(settings_obj, fake_llm):
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.summary and a.intent == "ready_to_buy"
    assert a.key_requirements == ["3BHK", "east-facing"]
    assert a.objections == []
    assert a.next_action and a.suggested_response
    assert a.engine.startswith("groq:")


@pytest.mark.asyncio
async def test_prompt_carries_the_lead_context(settings_obj, fake_llm):
    await analyze_lead(INTAKE, settings_obj)
    sent = fake_llm["calls"][0]["messages"][0]["content"]
    for token in ("Rahul Mehta", "Whitefield", "1.4 crore", "Pre-approved"):
        assert token in sent


# ---- deterministic caps ----------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("bucket,cap", sorted(TIMELINE_CAP.items()))
async def test_timeline_bucket_caps_the_score(settings_obj, fake_llm, bucket, cap):
    """A far-off lead can never outrank a near one, whatever the model says."""
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": 99,
                           "timeline_bucket": bucket}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.priority_score == min(99, cap)


@pytest.mark.asyncio
async def test_missing_budget_caps_the_score(settings_obj, fake_llm):
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": 95, "has_budget": False}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.priority_score == NO_BUDGET_CAP


@pytest.mark.asyncio
async def test_no_explicit_ask_caps_the_score(settings_obj, fake_llm):
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": 95, "explicit_ask": False}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.priority_score == NO_ASK_CAP


@pytest.mark.asyncio
async def test_strictest_cap_wins(settings_obj, fake_llm):
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": 99,
                           "timeline_bucket": "over_12_months", "has_budget": False}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.priority_score == TIMELINE_CAP["over_12_months"]  # 40 < 65


@pytest.mark.asyncio
async def test_cap_is_explained_in_the_reasoning(settings_obj, fake_llm):
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": 99,
                           "timeline_bucket": "6_12_months"}
    a = await analyze_lead(INTAKE, settings_obj)
    assert "Capped at 55" in a.score_reasoning


@pytest.mark.asyncio
async def test_a_score_below_the_cap_is_untouched(settings_obj, fake_llm):
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": 30,
                           "timeline_bucket": "immediate"}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.priority_score == 30 and "Capped" not in a.score_reasoning


# ---- consistency ------------------------------------------------------------ #
@pytest.mark.asyncio
@pytest.mark.parametrize("score,expect", [(95, "hot"), (75, "hot"), (74, "warm"),
                                          (45, "warm"), (44, "cold"), (0, "cold")])
async def test_temperature_is_derived_from_the_score(settings_obj, fake_llm, score, expect):
    """A band that disagrees with the score makes the ranked list look broken."""
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": score,
                           "temperature": "cold", "timeline_bucket": "immediate"}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.temperature == expect


@pytest.mark.asyncio
async def test_cold_lead_never_lands_in_act_today(settings_obj, fake_llm):
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": 20,
                           "urgency": "high", "timeline_bucket": "over_12_months"}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.temperature == "cold" and a.urgency != "high"


@pytest.mark.asyncio
async def test_price_shopper_cannot_be_top_priority(settings_obj, fake_llm):
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "intent": "price_shopping",
                           "priority_score": 95, "timeline_bucket": "immediate"}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.priority_score <= 60


@pytest.mark.asyncio
async def test_score_is_clamped_to_range(settings_obj, fake_llm):
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": 5000,
                           "timeline_bucket": "immediate"}
    a = await analyze_lead(INTAKE, settings_obj)
    assert 0 <= a.priority_score <= 100


@pytest.mark.asyncio
async def test_list_fields_are_cleaned(settings_obj, fake_llm):
    fake_llm["payload"] = {**ANALYSIS_PAYLOAD,
                           "key_requirements": ["  3BHK  ", "", "   ", "east"]}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.key_requirements == ["3BHK", "east"]


# ---- provider fallback ------------------------------------------------------ #
@pytest.mark.asyncio
async def test_falls_back_to_gemini_when_groq_fails(settings_obj, fake_llm):
    fake_llm["fail"] = {"groq"}
    a = await analyze_lead(INTAKE, settings_obj)
    assert a.engine.startswith("gemini:")


@pytest.mark.asyncio
async def test_error_when_every_provider_fails(settings_obj, fake_llm):
    from app.llm import LLMError
    fake_llm["fail"] = {"groq", "gemini"}
    with pytest.raises(LLMError) as ei:
        await analyze_lead(INTAKE, settings_obj)
    assert len(ei.value.attempts) == 2
