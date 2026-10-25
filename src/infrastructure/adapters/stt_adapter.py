# src/infrastructure/adapters/stt_adapter.py
import logging

from faster_whisper import WhisperModel

from src.domain.exceptions.core_errors import HardwareDegradationError
from src.domain.ports.audio_service import IAudioService

logger = logging.getLogger(__name__)


class STTAdapter(IAudioService):
    """Adapter for speech-to-text using Faster-Whisper."""

    def __init__(self, model_size: str = "base", device: str = "cpu") -> None:
        try:
            logger.info(f"Cargando modelo Whisper '{model_size}' en {device}...")
            self._model = WhisperModel(model_size, device=device, compute_type="int8")
            self._available = True
            logger.info("Modelo de transcripcion listo.")
        except Exception as e:
            logger.error(f"Error critico al cargar el modelo de voz: {e}")
            self._model = None
            self._available = False

    # --- IAudioService ---
    def is_stt_available(self) -> bool:
        return self._available and self._model is not None

    def is_tts_available(self) -> bool:
        return False

    def listen(self) -> str:
        raise HardwareDegradationError(
            "STTAdapter.listen requiere un archivo de audio; use transcribe(path)."
        )

    def speak(self, text: str) -> None:
        raise HardwareDegradationError("TTS no esta implementado en STTAdapter.")

    # --- Concrete transcription ---
    def transcribe(self, audio_path: str | None) -> str:
        if not self.is_stt_available():
            raise HardwareDegradationError("El motor de transcripcion no esta disponible.")
        if not audio_path:
            return ""

        try:
            segments, info = self._model.transcribe(
                audio_path,
                beam_size=5,
                language="es",
                task="transcribe",
                condition_on_previous_text=False,
            )

            transcription = " ".join([segment.text for segment in segments]).strip()

            logger.info("--- AUDITORIA DE VOZ ---")
            logger.info(
                f"Idioma detectado: {info.language} (Probabilidad: {info.language_probability:.2f})"
            )
            logger.info(f"Texto detectado: '{transcription}'")
            logger.info("------------------------")

            if len(transcription) < 3:
                logger.warning("La transcripcion fue ignorada por ser demasiado corta o ruido.")
                return ""

            return transcription
        except Exception as e:
            logger.error(f"Error tecnico durante la transcripcion: {e}")
            raise HardwareDegradationError("Fallo en la transcripcion de audio.", e) from e
