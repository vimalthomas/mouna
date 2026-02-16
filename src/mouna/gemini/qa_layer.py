"""Gemini QA layer for label validation and failure mode explanation."""

import os
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
from loguru import logger


class GeminiQA:
    """Gemini-based QA layer for sign language recognition validation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-pro",
        confidence_threshold: float = 0.7,
        max_retries: int = 3,
    ):
        """
        Initialize Gemini QA layer.

        Args:
            api_key: Gemini API key. If None, reads from environment.
            model_name: Gemini model to use.
            confidence_threshold: Only validate predictions below this confidence.
            max_retries: Maximum number of API retries.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or arguments")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

        logger.info(f"Initialized Gemini QA with model: {model_name}")

    def validate_prediction(
        self,
        video_path: str,
        predicted_label: str,
        confidence: float,
        top_k_predictions: Optional[List[Tuple[str, float]]] = None,
    ) -> Dict[str, any]:
        """
        Validate a prediction using Gemini.

        Args:
            video_path: Path to video file.
            predicted_label: Model's predicted sign label.
            confidence: Confidence score of prediction.
            top_k_predictions: List of (label, confidence) tuples for top-k predictions.

        Returns:
            Dictionary with:
                - should_validate: Whether validation was performed
                - gemini_response: Raw Gemini response
                - validation_result: Parsed validation result
                - failure_explanation: Explanation if prediction seems wrong
        """
        result = {
            "should_validate": False,
            "gemini_response": None,
            "validation_result": None,
            "failure_explanation": None,
        }

        # Only validate low-confidence predictions
        if confidence >= self.confidence_threshold:
            logger.debug(
                f"Skipping validation for high-confidence prediction: {confidence:.3f}"
            )
            return result

        result["should_validate"] = True

        # Prepare prompt
        prompt = self._create_validation_prompt(
            predicted_label, confidence, top_k_predictions
        )

        # Upload video and get response
        try:
            response = self._query_gemini_with_video(video_path, prompt)
            result["gemini_response"] = response
            result["validation_result"] = self._parse_validation_response(response)
            result["failure_explanation"] = self._extract_explanation(response)

            logger.info(
                f"Gemini validation for '{predicted_label}': {result['validation_result']}"
            )

        except Exception as e:
            logger.error(f"Gemini validation failed: {e}")
            result["error"] = str(e)

        return result

    def explain_failure(
        self,
        video_path: str,
        true_label: str,
        predicted_label: str,
        top_k_predictions: Optional[List[Tuple[str, float]]] = None,
    ) -> str:
        """
        Get explanation for why model made incorrect prediction.

        Args:
            video_path: Path to video file.
            true_label: Ground truth label.
            predicted_label: Model's incorrect prediction.
            top_k_predictions: Top-k predictions from model.

        Returns:
            Explanation text from Gemini.
        """
        prompt = f"""
        This is a sign language video. The correct sign is "{true_label}",
        but the model predicted "{predicted_label}".

        Please analyze why the model might have made this mistake:
        1. Are there visual similarities between these two signs?
        2. What distinguishing features should the model focus on?
        3. What could improve the recognition of "{true_label}"?

        Top predictions from the model:
        {self._format_top_k(top_k_predictions)}

        Provide a concise explanation (2-3 sentences).
        """

        try:
            response = self._query_gemini_with_video(video_path, prompt)
            return response
        except Exception as e:
            logger.error(f"Failure explanation failed: {e}")
            return f"Error: {str(e)}"

    def batch_validate(
        self,
        videos_and_predictions: List[Tuple[str, str, float]],
    ) -> List[Dict]:
        """
        Validate multiple predictions in batch.

        Args:
            videos_and_predictions: List of (video_path, predicted_label, confidence).

        Returns:
            List of validation results.
        """
        results = []
        for video_path, predicted_label, confidence in videos_and_predictions:
            result = self.validate_prediction(video_path, predicted_label, confidence)
            results.append(result)
            time.sleep(1)  # Rate limiting

        return results

    def _create_validation_prompt(
        self,
        predicted_label: str,
        confidence: float,
        top_k_predictions: Optional[List[Tuple[str, float]]],
    ) -> str:
        """Create prompt for Gemini validation."""
        prompt = f"""
        This is a sign language video. A machine learning model predicted this sign as "{predicted_label}"
        with confidence {confidence:.2%}.

        Please analyze the video and answer:
        1. Does this appear to be the sign for "{predicted_label}"? (Yes/No/Uncertain)
        2. If not, what sign does it look like?
        3. Brief explanation (1-2 sentences).

        Top predictions from the model:
        {self._format_top_k(top_k_predictions)}

        Respond in this format:
        Validation: [Yes/No/Uncertain]
        Alternative: [alternative sign if not predicted_label]
        Explanation: [brief explanation]
        """
        return prompt

    def _format_top_k(
        self, top_k_predictions: Optional[List[Tuple[str, float]]]
    ) -> str:
        """Format top-k predictions for prompt."""
        if not top_k_predictions:
            return "Not provided"

        lines = []
        for i, (label, conf) in enumerate(top_k_predictions, 1):
            lines.append(f"{i}. {label}: {conf:.2%}")
        return "\n".join(lines)

    def _query_gemini_with_video(self, video_path: str, prompt: str) -> str:
        """
        Query Gemini with video and prompt.

        Args:
            video_path: Path to video file.
            prompt: Text prompt.

        Returns:
            Gemini response text.
        """
        # Upload video file
        video_file = genai.upload_file(video_path)

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed for {video_path}")

        # Generate response
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content([video_file, prompt])
                return response.text
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def _parse_validation_response(self, response: str) -> Optional[str]:
        """Parse validation result from Gemini response."""
        if "Validation:" in response:
            validation_line = [
                line for line in response.split("\n") if "Validation:" in line
            ]
            if validation_line:
                return validation_line[0].split("Validation:")[1].strip()
        return None

    def _extract_explanation(self, response: str) -> Optional[str]:
        """Extract explanation from Gemini response."""
        if "Explanation:" in response:
            explanation_line = [
                line for line in response.split("\n") if "Explanation:" in line
            ]
            if explanation_line:
                return explanation_line[0].split("Explanation:")[1].strip()
        return None
