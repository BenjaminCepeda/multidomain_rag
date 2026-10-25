# main.py
import os
import logging
import sys
from dotenv import load_dotenv
from src.presentation.web_app import UserInterfaceApplication

def setup_environment() -> None:
    """Carga y valida las variables críticas del entorno."""
    if not load_dotenv():
        print("Advertencia: No se encontró archivo .env. Utilizando valores por defecto.")
    
    # Validaciones mínimas de seguridad operativa
    required_vars = ["QDRANT_URL", "LLM_ENDPOINT"]
    for var in required_vars:
        if not os.getenv(var):
            logging.warning(f"Variable de entorno {var} no definida. Se usará valor por defecto.")

def main() -> None:
    """Punto de entrada principal para el despliegue del Agente de Ingeniería."""
    # Configuración de logs globales
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("MainSystem")
    
    try:
        logger.info("Iniciando sistema de agente multimodal...")
        setup_environment()
        
        # Inicialización de la aplicación
        app = UserInterfaceApplication()
        interface = app.build_interface()
        
        # Lanzamiento del servidor de interfaz
        logger.info("Sistema listo. Interfaz disponible en http://127.0.0.1:7860")
        interface.launch(server_name="127.0.0.1", server_port=7860, share=False)
        
    except Exception as e:
        logger.critical(f"Fallo crítico en el arranque del sistema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()