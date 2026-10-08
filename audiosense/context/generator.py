"""
Context Statement and Database Summary Generators for AudioSense.
"""

import json
from typing import Dict, Any, List, Optional
from .llm import LLMClient
from .translator import translate_text

def generate_context_statement(
    detected_labels: List[Dict[str, Any]],
    llm_client: LLMClient,
    language: str = "en"
) -> str:
    """
    Generate a concise natural language statement of the surrounding environment
    based on detected labels, confidence scores, and context.
    """
    if not detected_labels:
        statement = "No significant sound events detected in the immediate environment."
        return translate_text(statement, target_lang=language, llm_client=llm_client)

    # Format the prompt
    prompt = f"""
You are the sensory interpretation module for an autonomous robot or intelligent audio system.
Analyze the following detected sound labels and confidence scores in the current environment:

Detected Sounds:
{json.dumps(detected_labels, indent=2)}

Based on these sounds, generate a single, clear, natural language statement summarizing what is currently happening in the immediate surroundings.
Context Statement:
"""
    raw_statement = llm_client.generate(prompt).strip()

    # If offline fallback returned generic message, synthesize a rich specific one:
    if "Environment analysis indicates active sound events" in raw_statement:
        top_labels = [item.get("label", "") for item in detected_labels[:3]]
        raw_statement = f"Surroundings currently exhibit sounds of {', '.join(top_labels)}."

    return translate_text(raw_statement, target_lang=language, llm_client=llm_client)


def generate_database_summary(
    db_data: Dict[str, Any],
    llm_client: LLMClient,
    language: str = "en"
) -> str:
    """
    Generate a well-formatted summary of the entire audio database using the LLM.
    Requirement 7: summary_db()
    """
    current = db_data.get("current_labels", {})
    history = db_data.get("old_labels", {})

    if not current and not history:
        msg = "The audio database is currently empty. No sound events have been recorded."
        return translate_text(msg, target_lang=language, llm_client=llm_client)

    prompt = f"""
You are the intelligence log summarizer for AudioSense.
Analyze the following acoustic database which tracks active sounds and historical sound events:

Database Content:
{json.dumps(db_data, indent=2)}

Provide a well-formatted, executive summary highlighting:
1. Current active sounds and their confidence.
2. Most frequently occurring sounds over time.
3. Overall environmental situation assessment.
Summary:
"""
    raw_summary = llm_client.generate(prompt).strip()

    if "Database audit summary" in raw_summary:
        curr_str = ", ".join(current.keys()) if current else "None"
        hist_counts = [f"{lbl} ({info.get('count', 0)} times)" for lbl, info in history.items()]
        hist_str = ", ".join(hist_counts) if hist_counts else "None"
        raw_summary = (
            f"=== AudioSense Database Summary ===\n"
            f"• Current Active Sound: {curr_str}\n"
            f"• Historical Sounds Logged: {hist_str}\n"
            f"• Assessment: Continuous monitoring active with recorded auditory events."
        )

    return translate_text(raw_summary, target_lang=language, llm_client=llm_client)
