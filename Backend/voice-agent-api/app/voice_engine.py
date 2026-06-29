import asyncio
import logging
from pathlib import Path

import soundfile as sf
from voxcpm import VoxCPM

from app.config import Settings
from app.exceptions import ModelInferenceError

logger = logging.getLogger(__name__)


class VoiceEngine:
    """Thread-safe VoxCPM wrapper with per-voice LoRA hot-swap.

    All inference is serialized behind a single asyncio.Lock so GPU state
    is never mutated concurrently.  The hot-swap is handled inside
    _generate_sync (already under the lock) — no extra locking needed.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._model: VoxCPM | None = None
        self._current_lora_path: str | None = None  # tracks which LoRA is currently loaded

    def load(self) -> None:
        source = self._settings.model_local_path or self._settings.model_id
        logger.info("Loading VoxCPM from %s", source)
        self._model = VoxCPM.from_pretrained(source)
        logger.info("VoxCPM loaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def synthesize(
        self,
        *,
        text: str,
        output_path: Path,
        lora_path: str | None,
        prompt_wav_path: Path | None,
        prompt_text: str | None,
        cfg_value: float,
        inference_timesteps: int,
        normalize: bool,
        denoise: bool,
    ) -> float:
        if self._model is None:
            raise ModelInferenceError("Model not loaded")

        async with self._lock:
            loop = asyncio.get_running_loop()
            try:
                duration = await loop.run_in_executor(
                    None,
                    self._generate_sync,
                    text,
                    output_path,
                    lora_path,
                    prompt_wav_path,
                    prompt_text,
                    cfg_value,
                    inference_timesteps,
                    normalize,
                    denoise,
                )
            except ModelInferenceError:
                raise
            except Exception as exc:
                logger.exception("VoxCPM generation failed")
                raise ModelInferenceError(f"Generation failed: {exc}") from exc

        return duration

    def _ensure_lora(self, lora_path: str | None) -> None:
        """Hot-swap LoRA adapter if needed. Must be called inside _lock."""
        if lora_path == self._current_lora_path:
            return  # nothing to do

        if self._current_lora_path is not None:
            # unload the currently active adapter before loading another
            try:
                self._model.unload_lora()  # type: ignore[union-attr]
            except Exception:
                # if unload_lora not supported on this model version, fall through
                pass
            self._current_lora_path = None

        if lora_path is not None:
            self._model.load_lora(lora_path)  # type: ignore[union-attr]
            self._current_lora_path = lora_path
            logger.info("LoRA loaded: %s", lora_path)
        else:
            # switching back to base model (no LoRA)
            try:
                self._model.set_lora_enabled(False)  # type: ignore[union-attr]
            except Exception:
                pass
            logger.info("LoRA unloaded, using base model")

    def _generate_sync(
        self,
        text: str,
        output_path: Path,
        lora_path: str | None,
        prompt_wav_path: Path | None,
        prompt_text: str | None,
        cfg_value: float,
        inference_timesteps: int,
        normalize: bool,
        denoise: bool,
    ) -> float:
        assert self._model is not None

        self._ensure_lora(lora_path)

        # When a LoRA is active the voice is baked into weights — no reference audio needed.
        # When no LoRA, use prompt_wav_path for zero-shot cloning.
        effective_prompt_wav = None if lora_path else (str(prompt_wav_path) if prompt_wav_path else None)
        effective_prompt_text = None if lora_path else prompt_text

        wav = self._model.generate(
            text=text,
            prompt_wav_path=effective_prompt_wav,
            prompt_text=effective_prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=True,
            retry_badcase_max_times=3,
            retry_badcase_ratio_threshold=6.0,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wav, self._settings.sample_rate)
        return len(wav) / self._settings.sample_rate
