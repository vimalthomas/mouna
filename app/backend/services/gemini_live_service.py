import asyncio
import logging

from config import settings

logger = logging.getLogger(__name__)

ASL_COACHING_SYSTEM_PROMPT = """You are an expert ASL (American Sign Language) coach in a webcam practice app.

When providing coaching feedback on a sign attempt, respond ONLY with this JSON:
{"feedback": "your coaching feedback here", "tips": ["tip 1", "tip 2"]}

Rules:
- Correct + high confidence: praise, suggest speed or related signs
- Correct + low confidence: acknowledge but suggest technique improvements
- Incorrect: explain the difference between predicted and target sign
- Tips must be specific to the ASL sign (hand shape, position, movement)
- Feedback: max 2 sentences. Each tip: max 1 sentence.

When answering user questions, respond in plain text (NOT JSON). Be specific to ASL."""

LIVE_MODEL = "gemini-2.5-flash-native-audio-latest"
CHAT_MODEL = "gemini-flash-latest"

LIVE_TIMEOUT = 8.0   # seconds to wait for Live API before falling back
CHAT_TIMEOUT = 15.0  # seconds to wait for Chat API before giving up


class GeminiLiveSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._connected = False
        self._history: list[dict] = []
        self._live_available: bool | None = None  # None = not yet probed

    async def connect(self) -> None:
        if not settings.gemini_api_key or settings.gemini_api_key == "your-gemini-api-key":
            raise RuntimeError("GEMINI_API_KEY not configured")
        # Don't probe the Live API here — it can take seconds and blocks session_start.
        # Live availability is detected lazily on the first _send() call.
        self._connected = True
        logger.info("Gemini session %s connected", self.session_id)

    async def send_context(
        self,
        target_sign: str,
        predicted_sign: str,
        confidence: float,
        is_correct: bool,
        attempt_number: int,
    ) -> str:
        prompt = (
            f"Attempt #{attempt_number}. "
            f"Target sign: '{target_sign}'. "
            f"Model prediction: '{predicted_sign}' (confidence: {confidence:.2f}). "
            f"Correct: {'Yes' if is_correct else 'No'}. "
            f"Respond with coaching feedback in JSON: "
            f'{{"feedback": "...", "tips": ["...", "..."]}}'
        )
        return await self._send(prompt)

    async def send_chat(self, user_text: str, target_sign: str, attempts: int, accuracy: float) -> str:
        prompt = (
            f"User is practicing ASL sign '{target_sign}' "
            f"({attempts} attempts, {accuracy:.0%} accuracy). "
            f"They ask: \"{user_text}\"\n"
            f"Answer as their ASL coach. Max 3 sentences. Plain text, NOT JSON."
        )
        return await self._send(prompt)

    async def _send(self, prompt: str) -> str:
        if not self._connected:
            raise RuntimeError("Gemini not connected")

        if self._live_available is None:
            # Lazy probe: try Live API once with a short timeout so we don't stall the user.
            try:
                text = await asyncio.wait_for(self._send_live(prompt), timeout=LIVE_TIMEOUT)
                self._live_available = bool(text)
            except asyncio.TimeoutError:
                logger.warning("Live API timed out for session %s", self.session_id)
                self._live_available = False
                text = ""

            if not self._live_available:
                logger.warning("Live API not available for session %s, using chat API", self.session_id)

            if text:
                return text
            return await self._send_chat(prompt)

        if self._live_available:
            try:
                text = await asyncio.wait_for(self._send_live(prompt), timeout=LIVE_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("Live API timed out, falling back to chat API")
                text = ""
            if text:
                return text
            logger.warning("Live API failed, falling back to chat API")

        return await self._send_chat(prompt)

    async def _send_live(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=settings.gemini_api_key)
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
                    )
                ),
                system_instruction=types.Content(
                    parts=[types.Part(text=ASL_COACHING_SYSTEM_PROMPT)]
                ),
                output_audio_transcription=types.AudioTranscriptionConfig(),
            )

            async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                await session.send_client_content(
                    turns=types.Content(parts=[types.Part(text=prompt)]),
                    turn_complete=True,
                )
                text = ""
                async for resp in session.receive():
                    if resp.server_content:
                        if (
                            resp.server_content.output_transcription
                            and resp.server_content.output_transcription.text
                        ):
                            text += resp.server_content.output_transcription.text
                        if resp.server_content.turn_complete:
                            break

            logger.info("Gemini Live [%s]: %s", self.session_id, text[:150])
            return text.strip()
        except Exception:
            logger.exception("Gemini Live API error for session %s", self.session_id)
            return ""

    async def _send_chat(self, prompt: str) -> str:
        try:
            from google import genai

            self._history.append({"role": "user", "parts": [{"text": prompt}]})

            def call_api():
                client = genai.Client(api_key=settings.gemini_api_key)
                response = client.models.generate_content(
                    model=CHAT_MODEL,
                    contents=self._history,
                    config={
                        "system_instruction": ASL_COACHING_SYSTEM_PROMPT,
                        "temperature": 0.7,
                    },
                )
                return response.text.strip() if response.text else ""

            loop = asyncio.get_event_loop()
            text = await asyncio.wait_for(
                loop.run_in_executor(None, call_api),
                timeout=CHAT_TIMEOUT,
            )

            self._history.append({"role": "model", "parts": [{"text": text}]})
            if len(self._history) > 20:
                self._history = self._history[-20:]

            logger.info("Gemini Chat [%s]: %s", self.session_id, text[:150])
            return text
        except asyncio.TimeoutError:
            logger.error("Gemini Chat API timed out after %.0fs for session %s", CHAT_TIMEOUT, self.session_id)
            raise
        except Exception:
            logger.exception("Gemini Chat API error for session %s", self.session_id)
            raise

    async def close(self) -> None:
        self._connected = False
        self._history = []
        logger.info("Gemini session %s closed", self.session_id)


class GeminiLiveManager:
    def __init__(self) -> None:
        self._sessions: dict[str, GeminiLiveSession] = {}

    async def create_session(self, session_id: str) -> GeminiLiveSession:
        session = GeminiLiveSession(session_id)
        await session.connect()
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> GeminiLiveSession | None:
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        for sid in list(self._sessions):
            await self.close_session(sid)


gemini_manager = GeminiLiveManager()
