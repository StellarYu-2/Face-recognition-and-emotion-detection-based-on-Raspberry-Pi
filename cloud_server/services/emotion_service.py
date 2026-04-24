from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import EmotionConfig
from .ort_providers import build_provider_request, provider_names


EMOTION_LABELS = [
    "Calm",      # Neutral
    "Happy",
    "Happy",    # Surprise is grouped as Happy for the UI.
    "Sad",
    "Angry",
    "Angry",    # Disgust is grouped as Angry.
    "Sad",      # Fear is grouped as Sad.
    "Angry",    # Contempt is grouped as Angry.
]


@dataclass(frozen=True)
class EmotionStatus:
    enabled: bool
    ready: bool
    model_path: str
    provider: str
    available_providers: List[str]
    requested_providers: List[str]
    session_providers: List[str]
    input_name: str
    output_name: str
    error: Optional[str]
    warning: Optional[str]


class EmotionService:
    def __init__(self, config: EmotionConfig):
        self._config = config
        self._session = None
        self._input_name = ""
        self._output_name = ""
        self._provider = "none"
        self._available_providers: List[str] = []
        self._requested_providers: List[str] = []
        self._session_providers: List[str] = []
        self._error: Optional[str] = None
        self._warning: Optional[str] = None

        if not config.enabled:
            self._error = "disabled_by_config"
            return
        if not config.model_path.exists():
            self._error = f"model_not_found: {config.model_path}"
            return

        try:
            import onnxruntime as ort

            self._available_providers = list(ort.get_available_providers())
            providers = build_provider_request(
                config.preferred_provider,
                config.fallback_provider,
                self._available_providers,
            )
            self._requested_providers = provider_names(providers)
            if not providers:
                raise RuntimeError(f"no usable onnxruntime providers: {self._available_providers}")

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(config.model_path),
                sess_options=session_options,
                providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._session_providers = list(self._session.get_providers())
            self._provider = self._session_providers[0] if self._session_providers else "none"
            if config.preferred_provider in self._available_providers and config.preferred_provider not in self._session_providers:
                self._warning = (
                    f"preferred provider {config.preferred_provider} was available but session used "
                    f"{self._session_providers}"
                )
        except Exception as exc:
            self._session = None
            self._error = str(exc)

    @property
    def ready(self) -> bool:
        return self._session is not None

    def status(self) -> EmotionStatus:
        return EmotionStatus(
            enabled=self._config.enabled,
            ready=self.ready,
            model_path=str(self._config.model_path),
            provider=self._provider,
            available_providers=self._available_providers,
            requested_providers=self._requested_providers,
            session_providers=self._session_providers,
            input_name=self._input_name,
            output_name=self._output_name,
            error=self._error,
            warning=self._warning,
        )

    def infer(self, image_bytes: bytes) -> Dict:
        if self._session is None:
            raise RuntimeError(self._error or "emotion_service_not_ready")

        tensor = self._preprocess(image_bytes)
        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if logits.size == 0:
            raise RuntimeError("empty_emotion_output")

        probs = self._softmax(logits)
        grouped = self._group_probs(probs)
        label, confidence = self._decide(grouped)
        return {
            "label": label,
            "confidence": round(confidence * 100.0, 3),
            "probs": {k: round(v, 6) for k, v in grouped.items()},
            "debug": {
                "provider": self._provider,
                "session_providers": self._session_providers,
                "model": self._config.model_path.name,
                "input": self._input_name,
                "output": self._output_name,
                "floor": self._config.non_calm_floor,
                "handoff": self._config.handoff_margin,
                "sad_floor": self._config.sad_floor,
                "sad_handoff": self._config.sad_handoff_margin,
                "sad_vs_other_margin": self._config.sad_vs_other_margin,
            },
        }

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            raise RuntimeError("decode_failed")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.resize(gray, (self._config.input_size, self._config.input_size), interpolation=cv2.INTER_LINEAR)

        tensor = gray.astype(np.float32)
        if self._config.input_layout == "nhwc_gray":
            return tensor.reshape(1, self._config.input_size, self._config.input_size, 1)
        return tensor.reshape(1, 1, self._config.input_size, self._config.input_size)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32)
        logits = logits - float(np.max(logits))
        exp = np.exp(logits)
        total = float(np.sum(exp))
        if total <= 1e-9:
            return np.zeros_like(exp)
        return exp / total

    @staticmethod
    def _group_probs(probs: np.ndarray) -> Dict[str, float]:
        def get(idx: int) -> float:
            return float(probs[idx]) if idx < probs.size else 0.0

        return {
            "Calm": get(0),
            "Happy": get(1) + get(2),
            "Sad": get(3) + get(6),
            "Angry": get(4) + get(5) + get(7),
        }

    def _decide(self, grouped: Dict[str, float]) -> tuple[str, float]:
        calm = grouped["Calm"]
        sad = grouped["Sad"]
        happy = grouped["Happy"]
        angry = grouped["Angry"]
        if (
            sad >= self._config.sad_floor
            and sad + self._config.sad_handoff_margin >= calm
            and sad >= max(happy, angry) + self._config.sad_vs_other_margin
        ):
            return "Sad", sad

        non_calm = {k: v for k, v in grouped.items() if k != "Calm"}
        best_non_calm_label, best_non_calm_prob = max(non_calm.items(), key=lambda item: item[1])
        if (
            best_non_calm_prob >= self._config.non_calm_floor
            and best_non_calm_prob + self._config.handoff_margin >= calm
        ):
            return best_non_calm_label, best_non_calm_prob
        return "Calm", calm
