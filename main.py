import logging
import sys

from src.infrastructure.composition.root import CompositionRoot
from src.presentation.web_app import UserInterfaceApplication

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def setup_environment() -> None:
    """Composition root handles settings; this is a no-op placeholder for compatibility."""
    return None


def main() -> None:
    logger = logging.getLogger("MainSystem")
    try:
        logger.info("Iniciando sistema de agente multimodal...")
        ctx = CompositionRoot().build()
        app = UserInterfaceApplication(ctx)
        interface = app.build_interface()
        logger.info(
            f"Sistema listo. Interfaz disponible en http://{ctx.settings.gradio_host}:{ctx.settings.gradio_port}"
        )
        interface.launch(
            server_name=ctx.settings.gradio_host,
            server_port=ctx.settings.gradio_port,
            share=False,
            allowed_paths=app.allowed_paths,
        )
    except Exception as e:
        logger.critical(f"Fallo critico en el arranque del sistema: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
