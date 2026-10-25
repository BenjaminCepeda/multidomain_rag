from abc import ABC, abstractmethod


class IAudioService(ABC):
    """Contrato para la captura y reproducción fonética local."""

    @abstractmethod
    def is_stt_available(self) -> bool:
        """Verifica de forma proactiva la salud del hardware de entrada de audio (Micrófono)."""
        pass

    @abstractmethod
    def is_tts_available(self) -> bool:
        """Verifica la inicialización del motor nativo de síntesis de voz."""
        pass

    @abstractmethod
    def listen(self) -> str:
        """
        Captura el flujo del micrófono y aplica transcripción local (Whisper).

        Raises:
            HardwareDegradationError: Si el canal de grabación está bloqueado o ausente.
        """
        pass

    @abstractmethod
    def speak(self, text: str) -> None:
        """
        Sintetiza texto a voz de forma síncrona.

        Raises:
            HardwareDegradationError: Si los hilos de salida de audio del sistema fallan.
        """
        pass
