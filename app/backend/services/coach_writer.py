"""Turns one scored attempt into a single, prioritized coaching message.

Two layers, so the app never depends on an LLM to be correct:
  1. compose_template() -- deterministic, built only from the validated CP
     results (plain sentences from mouna.coach.parameters). Always computed.
  2. Claude (only when ANTHROPIC_API_KEY is configured) rewrites those SAME
     facts into a warmer, shorter message. It is instructed to use only the
     facts given -- never to describe how a sign is made from its own
     knowledge -- and any failure falls back to the template.
The per-parameter cards in the UI stay the source of truth either way.
"""

import json
import logging

import anthropic

from config import settings

logger = logging.getLogger(__name__)

# Most important first: a wrong number of hands or wrong place changes the
# sign more than a slightly-off handshape or palm angle.
PRIORITY = ["hand_count", "location", "handshape", "orientation"]
NAMES = {"hand_count": "number of hands", "location": "location", "handshape": "handshape", "orientation": "palm direction"}

SYSTEM_PROMPT = """You are a friendly ASL practice coach inside a learning app. After each practice attempt you get structured results from the app's checker and write the feedback the learner reads.

Rules:
- Use ONLY the facts provided. Do not describe how the sign is made beyond the given sign description and the given correction sentences; never add handshapes, movements or meanings from your own knowledge.
- Lead with what matters most: if something needs fixing, name the single most important fix (the first item in "fixes", which is already ordered by priority) and say concretely what to do. Mention at most one more fix.
- Briefly acknowledge what was right when the facts list something as correct.
- Plain, everyday words. Never mention axes, codes, buckets, scores, percentages, models or "the checker".
- 2 to 4 short sentences, under 80 words. No lists, no headings, no emoji. (A per-part checklist is shown under your text automatically, so do not repeat every part; focus on what to do on the next try.)
- If there are no fixes and the verdict is good, give short encouragement."""


def _facts(target: str, description: str, score: dict) -> dict:
    params = score.get("parameters") or {}
    fixes, correct = [], []
    variant_note = next((n for n in score.get("notes") or [] if "-handed version of this sign" in n), None)
    if score.get("hand_count_match") is False:
        fix = "Use the same number of hands as the target sign."
        if variant_note:  # e.g. one-handed SAME: correct ASL, just not the form practised here
            fix = "You " + variant_note.replace(" -- ", " - ").rstrip(".") + "."
        fixes.append({"part": "hand_count", "fix": fix})
    elif score.get("hand_count_match") is True:
        correct.append(NAMES["hand_count"])
    for key in PRIORITY[1:]:
        p = params.get(key)
        if not p or p.get("uncertain"):
            continue  # signers vary on this part: neither right nor wrong
        if p.get("match"):
            correct.append(NAMES[key])
        elif p.get("why"):
            fixes.append({"part": key, "fix": p["why"]})
    fixes.sort(key=lambda f: PRIORITY.index(f["part"]))
    return {
        "sign": target,
        "sign_description": description or "",
        "verdict": score.get("overall_verdict"),
        "correct": correct,
        "fixes": fixes,
        # Not part of the facts sent to Claude; used for the checklist.
        "_params": params,
        "_hand_count": score.get("hand_count_match"),
        "_variant_note": variant_note,
    }


LABELS = {"handshape": "Handshape", "location": "Location", "orientation": "Palm direction"}


def checklist(facts: dict) -> str:
    """One line per checked part, straight from the checks (never from the
    LLM), plus movement shown as ungraded."""
    lines = []
    hc = facts["_hand_count"]
    if hc is not None:
        if hc:
            lines.append("✓ Number of hands: right.")
        elif facts.get("_variant_note"):
            lines.append("! Number of hands: you " + facts["_variant_note"].replace(" -- ", " - "))
        else:
            lines.append("! Number of hands: use the same number of hands as the target sign.")
    for key, label in LABELS.items():
        p = facts["_params"].get(key)
        if not p:
            continue
        if p.get("uncertain"):
            lines.append(f"· {label}: signers vary on this part, so it isn't graded for this sign.")
            continue
        if p.get("match"):
            detail = f" ({p['yours_plain']})" if p.get("yours_plain") else ""
            lines.append(f"✓ {label}: right{detail}.")
        else:
            lines.append(f"! {label}: {p.get('why') or 'different from the target.'}")
    if facts["sign_description"]:
        lines.append(f"· Movement: counted in your overall result, but not checked on its own yet ({facts['sign_description'].lower()}). Compare with the example clip.")
    return "\n".join(lines)


def compose_template(facts: dict) -> str:
    target = facts["sign"]
    fixes = facts["fixes"]
    correct = facts["correct"]
    if not fixes:
        if facts["verdict"] == "good":
            return f"Nice, that looks like '{target}'. Keep it up, and try it once more to make it stick."
        return f"Close. Your hand matched '{target}' on every part we check, so the difference is probably in the movement. Watch the example clip and try the motion again."
    parts = [f"Not quite '{target}' yet." if facts["verdict"] != "needs_work" else f"Close to '{target}'."]
    if correct:
        parts.append(f"Your {_join(correct)} {'was' if len(correct) == 1 else 'were'} right.")
    parts.append("Most important: " + fixes[0]["fix"])
    if len(fixes) > 1:
        parts.append(f"After that, check your {_join([NAMES[f['part']] for f in fixes[1:]])}.")
    return " ".join(parts)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic | None:
    global _client
    if not settings.anthropic_api_key:
        return None
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=20.0, max_retries=1)
    return _client


_gemini_client = None


async def _gemini(system: str, turns: list[dict]) -> str | None:
    """Gemini fallback when only GEMINI_API_KEY is set. `turns` use the
    Claude roles (user/assistant). Returns None on any failure, so callers
    fall back to the template text."""
    global _gemini_client
    if not settings.gemini_api_key:
        return None
    try:
        from google import genai
        from google.genai import types
        if _gemini_client is None:
            _gemini_client = genai.Client(api_key=settings.gemini_api_key)
        contents = [{"role": "model" if t["role"] == "assistant" else "user", "parts": [{"text": t["content"]}]}
                    for t in turns]
        resp = await _gemini_client.aio.models.generate_content(
            model=settings.gemini_model, contents=contents,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=1024),
        )
        return (resp.text or "").strip() or None
    except Exception as e:  # noqa: BLE001 -- any SDK/network/key error means "use the template"
        logger.warning("Gemini unavailable (%s: %s)", type(e).__name__, str(e)[:120])
        return None


async def write_feedback(target: str, description: str, score: dict) -> tuple[str, str]:
    """Returns (message, source) where source is "claude" or "template".
    The message is a short summary followed by the per-part checklist."""
    if score.get("embedding_similarity") is None and not score.get("parameters"):
        # No reference recording for this sign: nothing was compared, so say
        # exactly that instead of implying the parts matched.
        how = f" The example shows: {description}." if description else ""
        return (
            f"I don't have a reference recording of '{target}' yet, so I can't grade this attempt or draw the target "
            f"over your video.{how} Watch your recording next to the example clip to compare them yourself.",
            "template",
        )
    facts = _facts(target, description, score)
    summary, source = await _summary(facts)
    lines = checklist(facts)
    return (f"{summary}\n\n{lines}" if lines else summary), source


async def _summary(facts: dict) -> tuple[str, str]:
    template = compose_template(facts)
    client = _get_client()
    facts = {k: v for k, v in facts.items() if not k.startswith("_")}
    if client is None:
        text = await _gemini(SYSTEM_PROMPT, [{"role": "user", "content": "Results for this attempt:\n"
                                             + json.dumps(facts, indent=2) + "\n\nWrite the feedback the learner reads."}])
        return (text, "gemini") if text else (template, "template")
    try:
        response = await client.beta.messages.create(
            model=settings.coach_llm_model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{
                "role": "user",
                "content": "Results for this attempt:\n" + json.dumps(facts, indent=2)
                + "\n\nWrite the feedback the learner reads.",
            }],
        )
    except anthropic.AuthenticationError:
        logger.warning("ANTHROPIC_API_KEY rejected; using template coaching message")
        return template, "template"
    except anthropic.RateLimitError:
        logger.warning("Claude rate-limited; using template coaching message")
        return template, "template"
    except anthropic.APIStatusError as e:
        logger.warning("Claude API error %s; using template coaching message", e.status_code)
        return template, "template"
    except anthropic.APIConnectionError:
        logger.warning("Claude unreachable; using template coaching message")
        return template, "template"

    if response.stop_reason == "refusal":
        return template, "template"
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return (text, "claude") if text else (template, "template")


# ---- chat ----

MOVEMENT_EXPLAINER = (
    "Movement is part of your overall result: the verdict compares your whole motion against the reference "
    "recording. What I don't have yet is a separate movement check with its own right/wrong line, because the "
    "two versions tested (hand speed, then direction of travel) couldn't reliably tell signs apart, and a check "
    "that's often wrong would mislead you. For now, compare your motion with the example clip."
)

CHAT_SYSTEM = """You are the coach chat inside an ASL practice app. The learner may ask about their latest attempt, the sign, or how the app's feedback works.

What you know (use only this; never describe how a sign is made beyond the given sign description and results):
- The app grades each attempt by comparing the learner's whole motion against a reference recording (that is the overall verdict), plus per-part checks for handshape, location, palm direction and number of hands.
- Movement has no separate per-part check yet: """ + MOVEMENT_EXPLAINER + """
- Facial-expression checks are an early, unvalidated hint.

Answer in plain, friendly words, 1 to 4 short sentences. If you don't know from the facts given, say so and suggest watching the example clip. Never mention models, scores, axes or codes."""


async def answer_chat(question: str, target: str, description: str, last_feedback: str | None,
                      history: list[dict]) -> tuple[str, str]:
    """Returns (reply, source). Claude when ANTHROPIC_API_KEY is set; otherwise
    a truthful fixed answer for the common question, and a plain note that
    free-form chat needs a key -- never a fake 'connection' error."""
    client = _get_client()
    context = f"Sign being practiced: '{target}'. Sign description: {description or 'n/a'}.\n"
    context += f"Latest attempt feedback shown to the learner:\n{last_feedback}" if last_feedback else "No attempt recorded yet."
    messages = [*history[-6:], {"role": "user", "content": f"{context}\n\nLearner's question: {question}"}]
    if client is None and settings.gemini_api_key:
        text = await _gemini(CHAT_SYSTEM, messages)
        return (text, "gemini") if text else ("I couldn't reach the AI coach just now. Please try again in a moment.", "template")
    if client is None:
        q = question.lower()
        if any(w in q for w in ("grade", "graded", "movement", "motion", "move")):
            return MOVEMENT_EXPLAINER, "template"
        return (
            "I can only answer free-form questions once an AI key is set up for the app (ANTHROPIC_API_KEY or GEMINI_API_KEY "
            "in the backend .env). Your results above don't need it: they come from the app's own checks.",
            "template",
        )
    try:
        response = await client.beta.messages.create(
            model=settings.coach_llm_model,
            max_tokens=2000,
            system=CHAT_SYSTEM,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
        )
    except (anthropic.AuthenticationError, anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        logger.warning("Claude chat unavailable (%s)", type(e).__name__)
        return "I couldn't reach the AI coach just now. Please try again in a moment.", "template"
    if response.stop_reason == "refusal":
        return "I can't help with that one, but I'm happy to answer questions about your signing.", "template"
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return (text or "Sorry, I didn't catch that. Could you ask again?"), "claude"
