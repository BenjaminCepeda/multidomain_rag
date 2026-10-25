# src/infrastructure/adapters/stt_adapter.py
import logging
from typing import Optional
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class STTAdapter:
    """Adaptador para transcripción de audio a texto utilizando Faster-Whisper con optimización técnica."""

    def __init__(self, model_size: str = "base", device: str = "cpu") -> None:
        """
        Inicializa el motor de transcripción local con configuraciones de alta precisión.
        """
        try:
            logger.info(f"Cargando modelo Whisper '{model_size}' en {device}...")
            # compute_type="int8" es el estándar para inferencia eficiente en CPU
            self._model = WhisperModel(model_size, device=device, compute_type="int8")
            logger.info("Modelo de transcripción listo.")
        except Exception as e:
            logger.error(f"Error crítico al cargar el modelo de voz: {e}")
            raise RuntimeError("No se pudo inicializar el motor de transcripción.")

    def transcribe(self, audio_path: Optional[str]) -> str:
        """
        Transcribe un archivo de audio a texto aplicando restricciones de lenguaje estrictas.
        """
        if not audio_path:
            return ""
        
        try:
            # 1. Fuerza el lenguaje a español.
            # 2. condition_on_previous_text=False es crucial para evitar que el silencio 
            #    o ruido de Gradio sea interpretado como contexto previo.
            # 3. task="transcribe" asegura que el modelo se comporte como transcriptor y no traductor.
            segments, info = self._model.transcribe(
                audio_path, 
                beam_size=5, 
                language="es",
                task="transcribe",
                condition_on_previous_text=False
            )
            
            transcription = " ".join([segment.text for segment in segments]).strip()
            
            # Auditoría detallada en logs para diagnóstico
            logger.info(f"--- AUDITORÍA DE VOZ ---")
            logger.info(f"Idioma detectado: {info.language} (Probabilidad: {info.language_probability:.2f})")
            logger.info(f"Texto detectado: '{transcription}'")
            logger.info(f"------------------------")
            
            # Filtro de calidad: si es demasiado corto, se considera ruido de grabación.
            if len(transcription) < 3:
                logger.warning("La transcripción fue ignorada por ser demasiado corta o ruido.")
                return ""
                
            return transcription
        except Exception as e:
            logger.error(f"Error técnico durante la transcripción: {e}")
            return ""