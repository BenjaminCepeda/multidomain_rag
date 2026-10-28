# Motor RAG Multidominio (multidomain_rag)

Motor de Generación Aumentada por Recuperación (RAG) de alto rendimiento y agnóstico al dominio, diseñado para ingestar, indexar y razonar sobre corpora documentales heterogéneos bajo principios estrictos de **Clean Architecture (Hexagonal)**.

---

## 🏗️ Principios Arquitectónicos

- **Clean Architecture Estricta:** Desacoplamiento absoluto entre capas:
  - `domain/`: Lógica de negocio pura, objetos de valor inmutables y puertos (interfaces). Sin frameworks ni dependencias externas.
  - `application/`: Casos de uso y DTOs que orquestan los flujos de la aplicación.
  - `infrastructure/`: Adaptadores concretos (Qdrant para vectores, PyMuPDF para extracción, LLM local) y Composition Root.
  - `presentation/`: Interfaz de usuario basada en Gradio.
- **Patrones de Diseño Obligatorios:** Patrón Repositorio, Patrón Estrategia (para políticas de ingesta) y Composition Root para inyección de dependencias.
- **Persistencia Local First:** Embeddings y metadatos almacenados localmente mediante Qdrant.

---

## 📁 Estructura del Repositorio

```text
├── src/
│   ├── domain/           # Modelos de dominio puros, objetos de valor y puertos
│   ├── application/      # Casos de uso y DTOs
│   ├── infrastructure/   # Adaptadores (Qdrant, LLM, PyMuPDF) y Composition Root
│   └── presentation/     # Adaptador de UI con Gradio
├── tests/                # Suites de pruebas unitarias y de integración
├── multidomain_rag.spec.md                # Documento de Arquitectura de Software
├── AGENTS.md             # Instrucciones de gobernanza para agentes
└── justfile              # Orquestador central de tareas y gate de calidad
