class DomainBaseError(Exception):
    """Clase base estricta para excepciones del dominio con tipado de mensaje."""

    def __init__(self, message: str, original_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.original_exception: Exception | None = original_exception


class HardwareDegradationError(DomainBaseError):
    """Se lanza cuando un subsistema físico (Micrófono, TTS, Cámara) falla localmente."""

    pass


class LLMInferenceError(DomainBaseError):
    """Se lanza cuando LM Studio agota el tiempo de espera o el modelo colapsa."""

    pass


class VectorStorageError(DomainBaseError):
    """Se lanza ante fallos de conexión, lectura o escritura en el espacio de Qdrant."""

    pass


class DataIncompleteError(DomainBaseError):
    """Se lanza cuando las precondiciones de un caso de uso no se cumplen."""

    pass
