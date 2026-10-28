# Instrucciones para Agentes: multidomain_rag

## Política de Gate (Puerta de Calidad)
El proyecto exige una pasarela de calidad estricta mediante **`just gate`**. Ninguna tarea se da por terminada a menos que todas las verificaciones pasen (`ruff check`, `ruff format`, `mypy`, `pytest`).

## Deuda Técnica y Contexto de Cobertura
- La cobertura de pruebas actual está por debajo del objetivo del 80% (~6%). Si bien la remediación de la cobertura es un hito planificado, cualquier nueva lógica de dominio o adaptador añadido debe respetar los estándares de prueba.

## Protocolo de Ejecución (Nativo de Terminal)
- **Uso Exclusivo de Terminal**: Todas las inspecciones, modificaciones de archivos, parches y validaciones deben ejecutarse exclusivamente mediante comandos nativos de shell (`uv`, `just`, `python`).

## Principios de Ingeniería, Patrones de Diseño y Arquitectura
- **Clean Architecture Estricta**: `interfaces -> application -> domain`. La capa `domain/` debe permanecer 100% pura (sin frameworks, sin ORMs, sin importaciones directas de Qdrant o Gradio).
- **Patrones de Diseño Obligatorios**: Aplicar explícitamente:
  - **Puertos y Adaptadores** (Hexagonal) para desacoplar infraestructura.
  - **Patrón Repositorio** para la persistencia vectorial y de datos.
  - **Patrón Estrategia** para políticas de negocio variables (ej. políticas de ingesta normal vs. extendida).
  - **Composition Root** para la inyección de dependencias centralizada.
- **Buenas Prácticas de Código**: Tipado estricto obligatorio (`mypy --strict`), inmutabilidad en Objetos de Valor (`frozen=True`), separación clara de DTOs en las fronteras y alta cohesión con baja acoplamiento.

## Seguridad y Privacidad
- **Política de Cero Registros (Zero-Log)**: Prohibido registrar en logs contenido de documentos, PII (información personal identificable) o vectores de embedding completos a nivel `INFO`.
- **Validación de Entradas**: Sanitizar y validar rigurosamente rutas de archivos, parámetros de consulta y payloads en los límites de la aplicación.

## Resiliencia y Manejo de Errores
- **Cero Bloques Vacíos**: Prohibido usar bloques `try/except` vacíos. Todo error debe capturarse, registrarse con contexto y traducirse a excepciones tipadas del dominio (`DomainBaseError`).
- **Degradación Granciosa**: Los adaptadores externos (como Qdrant o LLM local) deben implementar tiempos de espera explícitos (`timeouts`) y estrategias de recuperación ante fallos de red o hardware.

## GitFlow y Protocolo de Entrega
- Trabajar exclusivamente en ramas `feature/*`, `fix/*` o `refactor/*` derivadas de `develop`.
- Los nombres de ramas, mensajes de commit (Conventional Commits) y plantillas de PR deben redactarse estrictamente en **inglés**.
