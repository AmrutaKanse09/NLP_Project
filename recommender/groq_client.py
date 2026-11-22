"""OpenAI LLM client for real-time AI-powered recommendations and chat assistant.

Three public functions:
  - llm_recommend(profile, resources_df) -> list[dict]
      Uses the LLM to rank and pick the best resources for a learner in real time.
  - explain_recommendations(profile, recs) -> str
      Returns a conversational explanation of why the top recs suit the learner.
  - chat(messages, profile, resources_df) -> str
      Multi-turn chat assistant that knows the learner's profile and catalogue.
"""

import json
import os
from typing import Optional

import pandas as pd
from openai import OpenAI

_client: Optional[OpenAI] = None

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found. Make sure your .env file is loaded."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# --------------------------------------------------------------------------- helpers

def _profile_summary(profile) -> str:
    interests = ", ".join(profile.interests) if profile.interests else "not specified"
    formats = ", ".join(profile.preferred_formats) if profile.preferred_formats else "any"
    rated_items = len(profile.ratings)
    liked = [rid for rid, r in profile.ratings.items() if r >= 4]
    disliked = [rid for rid, r in profile.ratings.items() if r <= 2]
    return (
        f"Name: {profile.name or profile.username}\n"
        f"Skill level: {profile.level}\n"
        f"Interests: {interests}\n"
        f"Learning goal: {profile.goals or 'not specified'}\n"
        f"Preferred formats: {formats}\n"
        f"Max hours per resource: {profile.max_duration or 'no limit'}\n"
        f"Free only: {profile.free_only}\n"
        f"Resources rated: {rated_items} (liked IDs: {liked[:10]}, disliked IDs: {disliked[:5]})\n"
        f"Completed IDs: {profile.completed[:10]}"
    )


def _catalogue_text(resources: pd.DataFrame) -> str:
    """Format full catalogue as a compact numbered list for the LLM."""
    lines = []
    for _, row in resources.iterrows():
        cost = "Free" if str(row["cost"]).lower() == "free" else "Paid"
        lines.append(
            f"ID:{int(row['id'])} | {row['title']} | {row['topic']} | {row['level']} | "
            f"{row['format']} | {row['duration_hours']}h | {cost} | ★{row['rating']} | "
            f"{row['provider']} | {row['description'][:80]}"
        )
    return "\n".join(lines)


def _recs_summary(recs: list) -> str:
    lines = []
    for i, rec in enumerate(recs[:5], 1):
        r = rec.resource
        reasons = "; ".join(rec.reasons[:3]) if rec.reasons else "good fit"
        lines.append(
            f"{i}. {r['title']} ({r['level']} {r['format']}, {r['duration_hours']}h, "
            f"{r['cost']}, ★{r['rating']}) — {r['description'][:80]}... "
            f"[Signals: {reasons}]"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- public API

def llm_recommend(profile, resources: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Ask the LLM to select the best resources for this learner from the catalogue.

    Returns a list of dicts, each with keys: id, title, reason.
    The caller should cross-reference IDs back to the full resource records.
    """
    import re

    # Exclude already seen/completed
    seen = set(profile.ratings.keys()) | set(profile.completed)
    available = resources[~resources["id"].isin(seen)]

    # Apply hard constraints
    if profile.free_only:
        available = available[available["cost"].str.lower() == "free"]
    if profile.max_duration:
        available = available[available["duration_hours"] <= profile.max_duration]

    if available.empty:
        return []

    catalogue = _catalogue_text(available)
    profile_text = _profile_summary(profile)

    system = (
        "You are an expert learning advisor AI. "
        "Given a learner profile and a catalogue, select the best resources for that learner. "
        f"Pick exactly {top_n} resources. "
        'Return ONLY a JSON object with one key "picks" whose value is an array of objects. '
        'Each object must have: "id" (integer), "title" (string), "reason" (one sentence, specific to this learner). '
        "Order from best to least fit. Only use IDs present in the catalogue."
    )

    user = (
        f"LEARNER PROFILE:\n{profile_text}\n\n"
        f"CATALOGUE (ID | Title | Topic | Level | Format | Hours | Cost | Rating | Provider | Description):\n"
        f"{catalogue}\n\n"
        f'Return a JSON object: {{"picks": [{{"id": ..., "title": ..., "reason": ...}}, ...]}}'
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_tokens=1200,
    )

    raw = response.choices[0].message.content.strip()

    # Primary parse: expect {"picks": [...]}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for key in ("picks", "recommendations", "results", "resources"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            vals = list(parsed.values())
            if vals and isinstance(vals[0], list):
                return vals[0]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON array from raw text
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"LLM returned unparseable response. Raw output:\n{raw[:500]}")


def web_search_recommend(profile, top_n: int = 5) -> str:
    """Use the LLM to suggest real-world learning resources based on the learner's profile."""
    interests = ", ".join(profile.interests) if profile.interests else "programming"
    level = profile.level
    goal = profile.goals or f"learn {interests}"
    formats = ", ".join(profile.preferred_formats) if profile.preferred_formats else "any format"
    free_pref = "preferably free" if profile.free_only else "free or paid"

    prompt = (
        f"Recommend {top_n} real online learning resources for this learner:\n\n"
        f"- Skill level: {level}\n"
        f"- Interests: {interests}\n"
        f"- Goal: {goal}\n"
        f"- Preferred formats: {formats}\n"
        f"- Budget: {free_pref}\n\n"
        f"For each resource include: title, provider/platform, URL, cost (free/paid), "
        f"and why it fits this learner. Format as a numbered markdown list. "
        f"Only include real, currently available resources."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def explain_recommendations(profile, recs: list) -> str:
    """Call LLM to produce a short, friendly explanation of the top recommendations."""
    if not recs:
        return "No recommendations to explain."

    system = (
        "You are a helpful, concise learning advisor. "
        "Explain in 2-3 sentences why the given resources were recommended for this specific learner. "
        "Be personal, encouraging, and specific. Give an overall narrative — don't just list each resource."
    )

    user = (
        f"Learner profile:\n{_profile_summary(profile)}\n\n"
        f"Top recommended resources:\n{_recs_summary(recs)}\n\n"
        "In 2-3 sentences, explain why these recommendations are a great fit."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def chat(messages: list, profile=None, resources: Optional[pd.DataFrame] = None) -> str:
    """Multi-turn chat with the AI learning assistant."""
    context_parts = [
        "You are an expert AI learning assistant in a personalized learning recommender app.",
        "Help learners find resources, plan study paths, and answer learning questions.",
        "Be concise, friendly and specific. Only suggest resources that exist in the catalogue below.",
    ]

    if profile is not None:
        context_parts.append(f"\nCurrent learner:\n{_profile_summary(profile)}")

    if resources is not None:
        context_parts.append(f"\nCatalogue:\n{_catalogue_text(resources)}")

    full_messages = [{"role": "system", "content": "\n".join(context_parts)}] + messages

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=full_messages,
        temperature=0.7,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()
