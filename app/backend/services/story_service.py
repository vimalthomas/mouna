"""v5 Practice: short stories written from the signs a learner has learned.

The LLM (Gemini) writes 1-2 sentences in English plus ASL gloss; code then
checks every sign is in the learner's pool, so a story can never ask for a
sign they haven't learned. Prototyped in the lab (lab/stories/server.py).
Each Gemini call runs in a worker thread under a hard time limit: in the lab,
the async client could block the event loop so its own limit never fired.
"""
import asyncio
import json
import logging
import random
import time

from config import settings

logger = logging.getLogger(__name__)

MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
CALL_TIMEOUT_S = 30
BUDGET_S = 100
OFFER = 14          # signs offered per story (a random handful of the pool, for variety)
THEMES = ["home", "food", "school", "family", "weather and time", "work", "a small surprise",
          "friends", "a question", "a busy day"]

PROMPT = """You write very short practice stories for someone learning American Sign Language (ASL).
Use ONLY these signs (each gloss is one ASL sign): {vocab}

Write 1 or 2 short sentences that make a tiny, natural, everyday story (2 to 5 signs per sentence,
at most 8 signs in total). The English must say ONLY what the signs say: no extra ideas, adjectives
or verbs that have no sign in the gloss (small words like "the", "is", "a", "to" are fine).
Give each sentence in natural English and as ASL gloss: the signs in ASL order, in CAPITALS,
chosen ONLY from the list above. Follow basic ASL structure:
- time signs (NOW, LATER, YEAR, TIME) come first;
- topic first, then the comment about it (e.g. BIRD BLACK = "the bird is black");
- no words for "the", "a", "is", "are", "to" (ASL has no signs for them here);
- question signs (WHO) go at the end of the sentence;
- FINISH after a verb marks that it already happened; NO can go at the end to negate.
Do not invent signs, fingerspell, or use pronouns that are not in the list.
{must}{avoid}
Reply with JSON only:
{{"title": "<3-5 word title>",
  "sentences": [{{"english": "...", "gloss": ["SIGN", "SIGN", ...]}}],
  "structure": ["<one short note per sentence on how the ASL order differs from English>"]}}"""

_client = None


def enabled() -> bool:
    return bool(settings.gemini_api_key)


def _gemini():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types
        _client = genai.Client(api_key=settings.gemini_api_key,
                               http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
    return _client


def check(story: dict, vocab: set[str]) -> str | None:
    """None if the story only uses signs in `vocab` and has the right shape, else what's wrong."""
    sents = story.get("sentences")
    if not isinstance(sents, list) or not 1 <= len(sents) <= 2:
        return "give 1 or 2 sentences"
    total = 0
    for s in sents:
        g = s.get("gloss")
        if not isinstance(g, list) or not 2 <= len(g) <= 5 or not s.get("english"):
            return "each sentence needs English and 2-5 gloss signs"
        bad = [x for x in g if str(x).strip().lower() not in vocab]
        if bad:
            return f"these are not in the sign list: {', '.join(map(str, bad))}"
        total += len(g)
    return None if total <= 8 else "at most 8 signs in total"


async def write_story(pool: list[str], stretch: str | None = None, avoid: list[str] | None = None) -> dict:
    """A checked story from `pool` (lowercase glosses). `stretch`: one not-yet-learned sign to include.
    Raises RuntimeError if no valid story comes back within the time budget."""
    from google.genai import types
    offer = random.sample(pool, min(len(pool), OFFER))
    if stretch and stretch not in offer:
        offer.append(stretch)
    vocab = set(offer)
    must = (f"Use the sign {stretch.upper()} once (the learner is still practising it).\n" if stretch else "")
    av = ("Make it different from these recent stories: " + " | ".join((avoid or [])[-6:]) + "\n") if avoid else ""
    prompt = PROMPT.format(vocab=", ".join(w.upper() for w in offer), must=must, avoid=av) + \
        f"\nLoose theme, if the signs allow: {random.choice(THEMES)}."
    problem, t0 = None, time.time()
    for attempt in range(4):
        text = prompt + (f"\nYour previous answer was rejected: {problem}. Fix that." if problem else "")
        for model in MODELS:
            left = BUDGET_S - (time.time() - t0)
            if left < 3:
                raise RuntimeError("the story model is slow right now")
            try:
                r = await asyncio.wait_for(asyncio.to_thread(
                    _gemini().models.generate_content, model=model, contents=text,
                    config=types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=4096,
                                                       temperature=1.0)),
                    timeout=min(CALL_TIMEOUT_S, left))
                story = json.loads(r.text)
                if isinstance(story, list):
                    story = next((x for x in story if isinstance(x, dict)), {})
                problem = check(story, vocab)
                if problem is None:
                    for s in story["sentences"]:
                        s["gloss"] = [str(x).strip().upper() for x in s["gloss"]]
                    story["structure"] = [str(x) for x in (story.get("structure") or [])][:2]
                    story["title"] = str(story.get("title") or "A short story")[:200]
                    story.update({"model": model, "seconds": round(time.time() - t0, 1)})
                    return story
                logger.info("story rejected (%s)", problem)
                break    # answered, but wrongly: ask again with the problem stated
            except Exception as e:  # noqa: BLE001 -- busy / timeout / bad JSON: next model
                logger.warning("story: %s failed after %.0fs (%s)", model, time.time() - t0, type(e).__name__)
    raise RuntimeError(f"no valid story ({problem or 'no answer'})")
