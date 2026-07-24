import base64
import json
import os
import time

import cv2
import numpy as np
import google.generativeai as genai
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

class ScoreOutput(BaseModel):
    score: float

class VLMInterface:
    # pick whichever model tier your quota allows
    #_MODEL = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    _MODEL = genai.GenerativeModel("gemini-2.0-flash-lite")

    def __init__(self, vlm_type: str = "gemini"):
        """Create a VLM client.

        ``vlm_type`` accepts ``gemini``/``vlm_gemini``, ``openai``/``vlm_openai``,
        or ``anthropic``/``vlm_anthropic``.  OpenAI and Anthropic clients are
        imported only when selected, so Gemini remains the only required SDK.
        """
        provider = vlm_type.lower().removeprefix("vlm_")
        self.vlm_type = provider
        self._client = None

        if provider == "gemini":
            self._model = self._MODEL
        elif provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("OpenAI VLM support requires `pip install openai`.") from exc
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self._model = os.environ.get("OPENAI_VLM_MODEL", "gpt-4o-mini")
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ImportError("Anthropic VLM support requires `pip install anthropic`.") from exc
            self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            self._model = os.environ.get("ANTHROPIC_VLM_MODEL", "claude-3-5-haiku-latest")
        else:
            raise ValueError(
                f"Unsupported VLM type {vlm_type!r}. Choose gemini, openai, or anthropic."
            )

    @staticmethod
    def _load_mp4_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def _sample_video_frames(path: str, count: int = 8) -> list[bytes]:
        """Return evenly spaced JPEG frames for providers without video input."""
        capture = cv2.VideoCapture(path)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_numbers = np.linspace(0, max(total - 1, 0), num=count, dtype=int)
        frames = []
        for frame_number in np.unique(frame_numbers):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, frame = capture.read()
            if ok:
                encoded, jpg = cv2.imencode(".jpg", frame)
                if encoded:
                    frames.append(jpg.tobytes())
        capture.release()
        if not frames:
            raise ValueError(f"Could not read frames from {path}")
        return frames

    def task_success(self, video_path: str, prompt: str) -> float:
        """
        Ask Gemini to grade whether the *desired* behaviour occurred.
        The model responds with a float 0-1 in JSON.
        """
        system_instruction = (
            "You are an automated evaluator. "
            "Return ONLY valid JSON: {\"score\": <0-1 float>} where 1.0 = full success."
        )
        user_instruction = (
            f"Here is the task description: {prompt}\n"
            "Watch the video and judge whether the task was accomplished."
        )

        if self.vlm_type == "gemini":
            video_bytes = self._load_mp4_bytes(video_path)
            response = self._model.generate_content(
            [
                {"mime_type": "video/mp4", "data": video_bytes},
                system_instruction,
                user_instruction,
            ],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ScoreOutput,
                "temperature": 0.0,
            },
            )
            j = json.loads(response.text)
        else:
            # OpenAI and Anthropic accept images, not MP4 data URLs.  A small,
            # evenly distributed sample preserves the temporal context cheaply.
            j = self._score_with_messages(
                system_instruction,
                f"{user_instruction}\nThe attached images are ordered video frames.",
                [("image/jpeg", frame) for frame in self._sample_video_frames(video_path)],
            )

        # Sleep to prevent rate limits.
        # Max 30 RPM.
        time.sleep(0.5)
        return j["score"]
    
    def negative_visual_quality(self, frame: np.ndarray) -> float:
        """
        Returns a penalty (0-1) where 0 = pristine and 1 = unusable.
        """
        # encode OpenCV BGR frame → JPEG bytes
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Could not encode frame")

        prompt = (
            "Rate the VISUAL QUALITY of this frame on a continuous scale "
            "where 0 = excellent, 1 = terrible. "
            "Only respond with JSON: {\"score\": <float>}."
        )

        if self.vlm_type == "gemini":
            response = self._model.generate_content(
                [
                    {"mime_type": "image/jpeg", "data": jpg.tobytes()},
                    prompt,
                ],
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": ScoreOutput,
                    "temperature": 0.0,
                },
            )
            j = json.loads(response.text)
        else:
            j = self._score_with_messages(
                "You are an automated evaluator. Return only valid JSON.",
                prompt,
                [("image/jpeg", jpg.tobytes())],
            )
        # Sleep to prevent rate limits.
        # Max 30 RPM.
        time.sleep(0.5)
        return float(j["score"])

    def _score_with_messages(
        self, system: str, prompt: str, attachments: list[tuple[str, bytes]]
    ) -> dict:
        """Request a JSON score from an OpenAI- or Anthropic-compatible VLM."""
        if self.vlm_type == "openai":
            content = [{"type": "text", "text": prompt}]
            content.extend({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
                },
            } for mime_type, data in attachments)
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            )
            return json.loads(response.choices[0].message.content)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=64,
            temperature=0,
            system=system,
            messages=[{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                } for mime_type, data in attachments] + [{"type": "text", "text": prompt}],
            }],
        )
        return json.loads(response.content[0].text)


if __name__ == "__main__":
    """
    Quick check that Gemini can be called end-to-end.

    Usage:
        python vlm_interface.py               # → uses auto-generated black video
        python vlm_interface.py path/to.mp4   # → uses your video file

    Requires `GOOGLE_API_KEY` (or equivalent) to be set in the environment.
    """

    import sys, tempfile, cv2, numpy as np, pathlib

    vlm = VLMInterface()
    video_path = 'input_video.mp4'

    prompt = (
        "Open the book"
    )
    ts_score = vlm.task_success(str(video_path), prompt)
    assert 0.0 <= ts_score <= 1.0, f"task_success score out of range: {ts_score}"
    print(f"task_success → {ts_score:.3f}")

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame for quality check")

    nq_score = vlm.negative_visual_quality(frame)
    assert 0.0 <= nq_score <= 1.0, f"negative_visual_quality out of range: {nq_score}"
    print(f"negative_visual_quality → {nq_score:.3f}")
