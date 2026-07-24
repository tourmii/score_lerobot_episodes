import base64
import json
import os

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()


class ScoreOutput(BaseModel):
    score: float


class VLLMInterface:

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 3600,
    ):
        self._base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self._api_key = api_key or os.environ.get("VLLM_API_KEY") or os.environ.get(
            "OPENAI_API_KEY", "EMPTY"
        )
        self._model = model or os.environ.get("OPENAI_VLM_MODEL", "Qwen/Qwen3.5-4B")
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=timeout)

    @staticmethod
    def _load_mp4_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def _sample_video_frames(path: str, count: int = 8) -> list[bytes]:
        """Return evenly spaced JPEG frames; the API takes images, not MP4."""
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
        """Grade whether the described task was accomplished, as a 0-1 float."""
        system_instruction = (
            "You are an automated evaluator. "
            "Return ONLY valid JSON: {\"score\": <0-1 float>} where 1.0 = full success."
        )
        user_instruction = (
            f"Here is the task description: {prompt}\n"
            "Watch the video and judge whether the task was accomplished."
        )

        j = self._score_with_messages(
            system_instruction,
            f"{user_instruction}\nThe attached images are ordered video frames.",
            [("image/jpeg", frame) for frame in self._sample_video_frames(video_path)],
        )
        return float(j["score"])

    def negative_visual_quality(self, frame: np.ndarray) -> float:
        """Returns a penalty (0-1) where 0 = pristine and 1 = unusable."""
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Could not encode frame")

        prompt = (
            "Rate the VISUAL QUALITY of this frame on a continuous scale "
            "where 0 = excellent, 1 = terrible. "
            "Only respond with JSON: {\"score\": <float>}."
        )

        j = self._score_with_messages(
            "You are an automated evaluator. Return only valid JSON.",
            prompt,
            [("image/jpeg", jpg.tobytes())],
        )
        return float(j["score"])

    def _score_with_messages(
        self, system: str, prompt: str, attachments: list[tuple[str, bytes]]
    ) -> dict:
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
            },
        } for mime_type, data in attachments)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return self._parse_score(response.choices[0].message.content)

    @staticmethod
    def _parse_score(text: str) -> dict:
        """Parse the model's JSON, tolerating fenced or prose-wrapped output."""
        if text is None:
            raise ValueError("Model returned no content")
        try:
            return ScoreOutput.model_validate_json(text).model_dump()
        except Exception:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"Could not parse score from model output: {text!r}")
        return ScoreOutput(**json.loads(text[start:end + 1])).model_dump()


if __name__ == "__main__":
    vlm = VLLMInterface()

    video_path = "input_video.mp4"
    prompt = "Pick up the pink cup"
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
