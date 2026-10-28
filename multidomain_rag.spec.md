# Software Architecture Document (SAD)

## Multidomain RAG Engine (`multidomain_rag`)

| Field | Value |
|---|---|
| **Document** | Software Architecture Document (SAD) |
| **Project** | `multidomain_rag` — Multidomain Retrieval-Augmented Generation Engine |
| **Version** | 1.0.0 |
| **Status** | Proposed (Specification-Driven Development baseline) |
| **Date** | 2026-10-24 |
| **Author** | Senior Software Architect |
| **Audience** | Engineering team, reviewers, OpenCode execution agents |
| **Methodology** | Specification-Driven Development (SDD) |
| **Language of implementation** | Python 3.13 |

---

## Table of Contents

1. [Objectives and Technical & Architectural Scope](#1-objectives-and-technical--architectural-scope)
2. [Target Architecture (Clean Architecture)](#2-target-architecture-clean-architecture)
3. [Version Control and Strict GitFlow](#3-version-control-and-strict-gitflow)
4. [Ports and Contracts Redesign](#4-ports-and-contracts-redesign)
5. [Document Ingestion Strategy](#5-document-ingestion-strategy)
6. [Centralized Composition and Configuration](#6-centralized-composition-and-configuration)
7. [Execution and Finalization Protocol for OpenCode](#7-execution-and-finalization-protocol-for-opencode)
8. [Current State vs. Target State (Gap Analysis)](#8-current-state-vs-target-state-gap-analysis)
9. [Refactoring Roadmap](#9-refactoring-roadmap)
10. [Quality, Testing, Observability, Security](#10-quality-testing-observability-security)
11. [Risks and Mitigations](#11-risks-and-mitigations)
12. [Acceptance Criteria](#12-acceptance-criteria)
13. [Glossary](#13-glossary)

---

## 1. Objectives and Technical & Architectural Scope

### 1.1 System Purpose

`multidomain_rag` is a **high-performance, modular, domain-agnostic Retrieval-Augmented Generation (RAG) engine** designed to ingest, index, and reason over **multiple documentary corpora** (technical manuals, engineering documents, scans, diagrams). It decouples retrieval from generation and exposes a clean, framework-agnostic core that can be driven by any presentation layer (web UI, CLI, API, voice).

Primary capabilities:

- Ingest heterogeneous documents (PDFs, images, scans) with both **normal** (≤ 50 pages) and **extended** (> 50 pages) ingestion policies.
- Chunk, embed, and persist document fragments into a local vector store (Qdrant).
- Orchestrate multimodal reasoning: manual search, image/diagram analysis, direct answers.
- Synthesize grounded, citation-backed technical answers using a local LLM.
- Degrade gracefully when hardware or inference subsystems fail.

### 1.2 Scope

**In scope**

- Domain model and business rules of RAG ingestion, retrieval, and orchestration.
- Application use cases: ingest manual, orchestrate agent, search, synthesize.
- Infrastructure adapters: LLM (LM Studio), vector store (Qdrant), document parser (PyMuPDF), STT (Faster-Whisper), embedding service.
- Centralized, strongly-typed configuration and a single Composition Root.
- Presentation layer (Gradio web UI) as a thin adapter.
- Version control governance, GitFlow, dependency management via `uv`.

**Out of scope (this SAD)**

- Production deployment topology (Kubernetes, cloud).
- Multi-tenant authorization model (future).
- Training/fine-tuning of models.
- Billing, quotas, SLA monitoring dashboards.

### 1.3 Technology Stack

| Concern | Technology | Notes |
|---|---|---|
| Language | Python 3.13 | Strict type hinting mandatory |
| Package manager | `uv` (exclusive) | No global installs; `pyproject.toml` is the source of truth |
| Lock file | `uv.lock` | Committed for reproducibility |
| Vector database | Qdrant (local, embedded or remote) | Cosine similarity, persistent storage |
| LLM (chat + vision) | Qwen 3 14B via LM Studio | OpenAI-compatible `/v1` endpoint |
| Embeddings | `BAAI/bge-m3` (1024 dims) | Multilingual, high recall |
| Document parsing | PyMuPDF (`fitz`) | PDF text + layout extraction |
| STT | Faster-Whisper | Local CPU inference |
| Web UI | Gradio 5.x | Presentation adapter only |
| Configuration | `pydantic-settings` (target) | Strongly typed |
| Validation | `pydantic` v2 | DTOs / Value Objects |
| Testing | `pytest` + `pytest-cov` | Unit + integration |
| Linting/typing | `ruff`, `mypy` (strict) | CI gates |

### 1.4 Design Patterns and Quality Standards

Mandatory patterns:

- **Clean Architecture** (Ports & Adapters / Hexagonal).
- **Dependency Inversion** — dependencies point inward.
- **Repository pattern** for vector storage.
- **Strategy pattern** for ingestion policies (`NormalIngestionPolicy`, `ExtendedIngestionPolicy`).
- **Factory / Composition Root** for object graph construction.
- **Value Objects** for domain primitives (`DocumentId`, `ChunkId`, `EmbeddingVector`, `PageRange`).
- **DTOs** at application boundaries.
- **Result / Either** style error handling at use-case boundaries (no bare `except`).

Quality standards:

- Strict Python type hinting on every public and private signature.
- No empty `try/except` blocks; every handler must log with context and re-raise or translate to a typed domain error.
- No `os.getenv` outside the configuration module.
- No framework imports in `domain/`.
- Cyclomatic complexity ≤ 10 per function (target).
- Test coverage ≥ 80% on `domain/` and `application/`.

### 1.5 Constraints and Assumptions

- Single-node, local-first deployment (developer workstation / lab).
- LLM and embedding models run locally; network is optional (Qdrant may be remote but defaults to local).
- GPU optional; CPU fallback must work (int8 compute).
- Documents may be large (hundreds of pages); memory budget is bounded.
- Idempotent ingestion: re-running on the same corpus must not duplicate vectors.

### 1.6 Non-Functional Objectives

| Attribute | Target |
|---|---|
| Ingestion throughput (normal, ≤50 pp) | < 30 s per 50-page PDF on CPU |
| Ingestion memory ceiling (extended) | < 512 MB peak, streaming/batched |
| Query latency (p95) | < 3 s end-to-end (retrieval + synthesis) |
| Resilience | Graceful degradation on LLM/Qdrant/STT failure |
| Idempotency | Re-ingestion yields identical point IDs; no duplicates |
| Observability | Structured logs with correlation IDs; audit trail of searches |
| Security | No secrets in code; `.env` ignored; input validation at boundaries |

---

## 2. Target Architecture (Clean Architecture)

### 2.1 Overview

The system is organized into four concentric layers. Dependencies point **inward**: Presentation → Application → Domain ← Infrastructure (adapters implement domain ports).

```mermaid
flowchart TB
    subgraph Presentation["Presentation Layer (Gradio / CLI / API)"]
        UI[Web UI / Controllers]
    end
    subgraph Application["Application Layer (Use Cases)"]
        UC1[IngestManualUseCase]
        UC2[OrchestrateAgentUseCase]
        UC3[SearchManualsUseCase]
    end
    subgraph Domain["Domain Layer (Core, framework-agnostic)"]
        M[Models / Value Objects]
        P[Ports (Interfaces)]
        E[Domain Errors]
        POL[Ingestion Policies]
    end
    subgraph Infrastructure["Infrastructure Layer (Adapters)"]
        A1[LocalLLMAdapter]
        A2[QdrantAdapter]
        A3[PyMuPDFAdapter]
        A4[EmbeddingAdapter]
        A5[STTAdapter]
        CFG[Settings / Composition Root]
    end

    UI --> UC1
    UI --> UC2
    UC1 --> P
    UC2 --> P
    P -.implemented by.-> A1
    P -.implemented by.-> A2
    P -.implemented by.-> A3
    P -.implemented by.-> A4
    A1 --> CFG
    A2 --> CFG
    M --- P
```

### 2.2 Layer Definitions

#### 2.2.1 Domain Layer (`src/domain/`)

The **heart** of the system. Pure Python, zero framework dependencies. Contains:

- **Models / Value Objects**: `ManualChunk`, `OrchestratorDecision`, `Document`, `ChunkId`, `DocumentId`, `EmbeddingVector`, `PageRange`, `IngestionJob`.
- **Ports (Interfaces)**: `IEmbeddingService`, `ILLMEngine`, `IVectorRepository`, `IDocumentExtractor`, `IChunker`, `IIngestionPolicy`, `IAudioService`.
- **Domain Errors**: `DomainBaseError` hierarchy.
- **Policies (abstract)**: `IIngestionPolicy`.
- **Rules**: chunking thresholds, score thresholds, allowed tools.

Rules:

- No imports from `application`, `infrastructure`, or `presentation`.
- No I/O, no `os`, no `requests`, no `qdrant_client`.
- All state immutable where possible (`frozen=True` dataclasses).

#### 2.2.2 Application Layer (`src/application/`)

Orchestrates domain objects to fulfill use cases. Contains:

- **Use Cases**: `IngestManualUseCase`, `OrchestrateAgentUseCase`, `SearchManualsUseCase`, `SynthesizeResponseUseCase`.
- **DTOs**: `InboundPayloadDTO`, `OutboundResponseDTO`, `IngestionRequestDTO`, `IngestionResultDTO`, `SearchQueryDTO`, `SearchResultDTO`.
- **Application services / mappers**: mapping between domain and DTOs.

Rules:

- Depends only on `domain/`.
- No framework or infrastructure imports.
- Defines transactional boundaries (ingestion as a unit).

#### 2.2.3 Infrastructure Layer (`src/infrastructure/`)

Implements domain ports with concrete technologies. Contains:

- **Adapters**: `LocalLLMAdapter`, `QdrantAdapter`, `PyMuPDFAdapter`, `SentenceTransformerEmbeddingAdapter`, `STTAdapter`.
- **Config**: `Settings` (pydantic-settings), `CompositionRoot`.
- **Persistence helpers**: checkpointing, file storage.

Rules:

- Implements domain ports; never contains business rules.
- All configuration consumed via `Settings`, never `os.getenv`.

#### 2.2.4 Presentation Layer (`src/presentation/`)

Thin delivery mechanism. Contains:

- Gradio UI (`web_app.py`), CLI entry point, (future) FastAPI.
- Controllers delegate to use cases; no business logic.
- Maps DTOs to view models.

### 2.3 Dependency Rule

```
presentation -> application -> domain <- infrastructure
```

- `domain` imports nothing from other layers.
- `application` imports only `domain`.
- `infrastructure` imports `domain` (and `application` DTOs only when strictly needed for mapping — discouraged; prefer domain types).
- `presentation` imports `application` and `domain` DTOs.

Enforced via:

- Import-linter configuration in `pyproject.toml` (target).
- CI check failing on forbidden imports.

### 2.4 Target Directory Structure

```
multidomain_rag/
├── pyproject.toml
├── uv.lock
├── .env.example
├── README.md
├── SAD.md
├── main.py
├── src/
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── document.py          # Document, ManualChunk, ChunkId, DocumentId
│   │   │   ├── agent.py             # OrchestratorDecision, ToolName (enum)
│   │   │   ├── embedding.py         # EmbeddingVector
│   │   │   └── ingestion.py         # IngestionJob, IngestionStatus, PageRange
│   │   ├── ports/
│   │   │   ├── embedding_service.py # IEmbeddingService (NEW)
│   │   │   ├── llm_engine.py       # ILLMEngine (split)
│   │   │   ├── vector_repository.py
│   │   │   ├── document_extractor.py
│   │   │   ├── chunker.py          # IChunker (NEW)
│   │   │   ├── ingestion_policy.py # IIngestionPolicy (NEW)
│   │   │   └── audio_service.py
│   │   ├── exceptions/
│   │   │   └── core_errors.py
│   │   └── services/
│   │       └── id_generator.py     # UUID/hash identity (NEW)
│   ├── application/
│   │   ├── __init__.py
│   │   ├── dtos/
│   │   │   ├── agent_dtos.py
│   │   │   └── ingestion_dtos.py   # NEW
│   │   └── use_cases/
│   │       ├── ingest_manual.py
│   │       ├── orchestrate_agent.py
│   │       ├── search_manuals.py   # NEW (extracted)
│   │       └── synthesize_response.py # NEW (extracted)
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── adapters/
│   │   │   ├── local_llm_adapter.py
│   │   │   ├── qdrant_adapter.py
│   │   │   ├── pymupdf_adapter.py
│   │   │   ├── embedding_adapter.py # NEW (split from qdrant)
│   │   │   ├── chunker_adapter.py  # NEW
│   │   │   └── stt_adapter.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py         # Settings (pydantic-settings)
│   │   ├── composition/
│   │   │   └── root.py             # Composition Root
│   │   └── persistence/
│   │       └── ingestion_checkpoint.py # NEW
│   └── presentation/
│       ├── __init__.py
│       ├── web_app.py
│       └── assets/
│           └── style.css
└── tests/
    ├── unit/
    │   ├── domain/
    │   └── application/
    └── integration/
        └── infrastructure/
```

### 2.5 Key Design Patterns

| Pattern | Where | Purpose |
|---|---|---|
| Ports & Adapters | `domain/ports` + `infrastructure/adapters` | Decouple domain from frameworks |
| Repository | `IVectorRepository` | Abstract vector persistence |
| Strategy | `IIngestionPolicy` | Normal vs extended ingestion |
| Factory / Composition Root | `infrastructure/composition/root.py` | Single object-graph construction site |
| Value Object | `ChunkId`, `DocumentId`, `EmbeddingVector` | Immutability, equality by value, validation |
| DTO | `application/dtos` | Cross-boundary data transfer |
| Template Method | `BaseIngestionPolicy` | Shared skeleton, variant steps |
| Chain of Responsibility | Orchestrator tool routing | Extensible tool dispatch |

### 2.6 Architectural Decision Records (ADRs)

- **ADR-001**: Local-first inference (LM Studio) over cloud APIs — privacy and offline operation.
- **ADR-002**: Qdrant as sole vector store — mature Python client, local embedded mode.
- **ADR-003**: Embedding service extracted from vector repository — reusability and testability.
- **ADR-004**: Idempotent ingestion via deterministic UUID/hash IDs — no duplicate vectors.
- **ADR-005**: Strongly-typed settings via pydantic-settings — single configuration source.
- **ADR-006**: Ingestion policy split at 50 pages — memory safety for large documents.

---

## 3. Version Control and Strict GitFlow

### 3.1 Branch Strategy

- `main` — production-ready, tagged releases only.
- `develop` — integration branch; **protected** on GitHub (PRs required, status checks required).
- `feature/*` — feature work, branched from `develop`.
- `release/*` — release preparation (optional, from `develop`).
- `hotfix/*` — urgent fixes from `main`.

Branch naming: `feature/<scope>-<short-description>` (e.g., `feature/ports-extract-embedding-service`).

### 3.2 Workflow

1. Create `feature/*` from `develop`.
2. Implement in small, atomic commits.
3. Open PR against `develop` (GitHub web UI by the user).
4. CI gates must pass (lint, type check, tests).
5. Squash or merge per team convention; delete branch after merge.
6. `develop` → `main` via release PR.

### 3.3 Conventional Commits (Mandatory)

Format: `<type>(<scope>): <subject>`

Allowed types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `build`, `ci`, `style`.

Scopes: `domain`, `ports`, `application`, `infra`, `config`, `ingestion`, `ui`, `deps`, `tests`.

Examples:

- `feat(ports): add IEmbeddingService and split from vector repository`
- `refactor(ingestion): implement IngestionPolicy for documents over 50 pages`
- `chore(deps): migrate dependency management to uv`
- `docs(sad): add software architecture document`

Rules:

- Subject ≤ 72 chars, imperative mood, lowercase.
- Body explains *why*, not *what*.
- Breaking changes use `!` and `BREAKING CHANGE:` footer.

### 3.4 Dependency Management (uv only)

- All dependencies declared in `pyproject.toml` under `[project.dependencies]` and `[project.optional-dependencies]`.
- Install via `uv add <pkg>` (never `pip install` globally).
- Lock via `uv lock`; commit `uv.lock`.
- No `requirements.txt` as source of truth (may be generated for compatibility but not edited by hand).
- Python version pinned via `.python-version` (3.13).

### 3.5 Branch Protection (GitHub)

- `develop` and `main`: require PR, require 1+ approving review, require status checks (lint, mypy, pytest), no force push, no deletion.
- Signed commits recommended.

### 3.6 Release Tagging

- Semantic Versioning `MAJOR.MINOR.PATCH`.
- Tags on `main` only (`v1.2.3`).

---

## 4. Ports and Contracts Redesign

### 4.1 Principles

- Ports are **interfaces** (ABCs) in `domain/ports/`; they depend only on domain types.
- Contracts accept **Value Objects / DTOs**, not primitive strings where a domain concept exists.
- Each port has a single, well-defined responsibility (ISP).
- Infrastructure services (embedding, inference) are **extracted** into dedicated ports/adapters to avoid rigid coupling inside `QdrantAdapter` / `LocalLLMAdapter`.

### 4.2 Value Objects and DTOs

#### 4.2.1 Domain Value Objects (`domain/models/`)

```python
# src/domain/models/ids.py
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class DocumentId:
    value: UUID

    @classmethod
    def from_source(cls, source_path: str) -> "DocumentId":
        # deterministic UUID5 based on normalized path
        ...


@dataclass(frozen=True)
class ChunkId:
    value: str  # "<doc_uuid>:<section>:<index>"


@dataclass(frozen=True)
class EmbeddingVector:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("EmbeddingVector cannot be empty")
```

```python
# src/domain/models/document.py (revised)
from dataclasses import dataclass, field
from typing import Mapping
from .ids import ChunkId, DocumentId


@dataclass(frozen=True)
class ManualChunk:
    chunk_id: ChunkId
    document_id: DocumentId
    content: str
    page: int
    section: str | None = None
    is_visual: bool = False
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)
```

#### 4.2.2 Application DTOs (`application/dtos/`)

```python
# src/application/dtos/ingestion_dtos.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestionRequestDTO:
    file_paths: tuple[Path, ...]
    force_reindex: bool = False
    policy_override: str | None = None  # "normal" | "extended" | None (auto)


@dataclass(frozen=True)
class IngestionResultDTO:
    documents_processed: int
    chunks_indexed: int
    skipped_duplicates: int
    failed_files: tuple[str, ...]
    duration_seconds: float
```

```python
# src/application/dtos/search_dtos.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchQueryDTO:
    text: str
    limit: int = 5
    min_score: float = 0.20
    source_filter: str | None = None


@dataclass(frozen=True)
class SearchResultDTO:
    chunk_id: str
    content: str
    source: str
    page: int
    score: float
```

### 4.3 Redesigned Ports

#### 4.3.1 Embedding Service (NEW, extracted)

```python
# src/domain/ports/embedding_service.py
from abc import ABC, abstractmethod
from typing import Sequence
from src.domain.models.embedding import EmbeddingVector


class IEmbeddingService(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a single text. Raises EmbeddingError."""
        ...

    @abstractmethod
    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[EmbeddingVector]:
        """Embed a batch efficiently. Raises EmbeddingError."""
        ...

    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def model_name(self) -> str: ...
```

#### 4.3.2 LLM Engine (split by responsibility)

```python
# src/domain/ports/llm_engine.py
from abc import ABC, abstractmethod
from typing import Sequence
from src.domain.models.agent import OrchestratorDecision


class ILLMEngine(ABC):
    @abstractmethod
    def get_orchestrator_decision(
        self, user_input: str, has_image: bool
    ) -> OrchestratorDecision: ...

    @abstractmethod
    def generate_response(self, context: Sequence[str], user_input: str) -> str: ...

    @abstractmethod
    def analyze_image(self, image_path: str) -> str: ...

    @abstractmethod
    def health_check(self) -> bool: ...
```

> Note: vision capability may be split into `IVisionEngine` in a future iteration (ISP), but kept unified for Qwen 3 14B which is multimodal.

#### 4.3.3 Vector Repository (uses embedding service externally)

```python
# src/domain/ports/vector_repository.py
from abc import ABC, abstractmethod
from typing import Sequence
from src.domain.models.document import ManualChunk
from src.domain.models.embedding import EmbeddingVector


class IVectorRepository(ABC):
    @abstractmethod
    def ensure_collection(self, vector_dim: int) -> None: ...

    @abstractmethod
    def upsert(self, chunks: Sequence[ManualChunk], vectors: Sequence[EmbeddingVector]) -> None: ...

    @abstractmethod
    def search(
        self, query_vector: EmbeddingVector, limit: int, min_score: float
    ) -> list[ManualChunk]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def exists_by_id(self, chunk_id: str) -> bool: ...
```

> **Key change**: embedding is no longer performed inside the repository; the use case (or an embedding adapter) produces vectors and passes them in. This removes the rigid coupling between Qdrant and SentenceTransformers.

#### 4.3.4 Document Extractor / Chunker

```python
# src/domain/ports/document_extractor.py
from abc import ABC, abstractmethod
from pathlib import Path
from src.domain.models.document import Document


class IDocumentExtractor(ABC):
    @abstractmethod
    def extract(self, path: Path) -> Document:
        """Extract raw text + structure. Raises DataIncompleteError."""
        ...


# src/domain/ports/chunker.py
from abc import ABC, abstractmethod
from src.domain.models.document import Document, ManualChunk


class IChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document, policy: "IIngestionPolicy") -> list[ManualChunk]: ...
```

#### 4.3.5 Ingestion Policy

```python
# src/domain/ports/ingestion_policy.py
from abc import ABC, abstractmethod
from typing import Sequence
from src.domain.models.document import ManualChunk


class IIngestionPolicy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def should_use(self, page_count: int) -> bool: ...

    @abstractmethod
    def chunk_document(self, document: "Document") -> Sequence[ManualChunk]: ...

    @abstractmethod
    def batch_size(self) -> int: ...
```

#### 4.3.6 Audio Service (unchanged contract, hardened)

Existing `IAudioService` retained; STT adapter must implement it (currently `STTAdapter` does not — see gap analysis).

### 4.4 Separation of Base Services

To avoid rigid coupling in adapters:

- `QdrantAdapter` depends on `IVectorRepository` contract only and receives an `IEmbeddingService` only if needed for query-time embedding; preferred: use case embeds query via `IEmbeddingService` and passes `EmbeddingVector` to `IVectorRepository.search`.
- `LocalLLMAdapter` depends only on HTTP client and `Settings`.
- `SentenceTransformerEmbeddingAdapter` implements `IEmbeddingService`.
- `PyMuPDFAdapter` implements `IDocumentExtractor`; a separate `HierarchicalChunker` implements `IChunker`.

### 4.5 Error Contract

All domain errors inherit from `DomainBaseError` with `message` and `original_exception`. Adapters translate vendor exceptions into domain errors; use cases translate domain errors into `OutboundResponseDTO` degradation flags. No empty `except`.

---

## 5. Document Ingestion Strategy

### 5.1 Classification

| Class | Pages | Policy | Memory strategy |
|---|---|---|---|
| Normal | ≤ 50 | `NormalIngestionPolicy` | In-memory, single pass |
| Extended | > 50 | `ExtendedIngestionPolicy` | Streaming, hierarchical chunking, batched upsert, checkpointing |

### 5.2 Normal Ingestion (≤ 50 pages)

1. Validate file exists and is readable.
2. Extract `Document` (text + page count + section outline).
3. `NormalIngestionPolicy.chunk_document` → paragraph/section chunks.
4. Generate deterministic `ChunkId`s.
5. Filter out already-indexed IDs (idempotency).
6. Embed batch (`IEmbeddingService.embed_batch`).
7. Upsert to Qdrant.
8. Return `IngestionResultDTO`.

### 5.3 Extended Ingestion (> 50 pages)

`ExtendedIngestionPolicy` applies:

- **Hierarchical chunking by sections**: build a section tree (heading → subsections → paragraphs); chunks carry `section` path in metadata.
- **Batch processing**: process N pages per batch (configurable, default 10), embed and upsert per batch to bound memory.
- **Stable identity**: `ChunkId = uuid5(document_id, section_path, page, index)`; re-running skips existing IDs.
- **Checkpointing**: persist progress (`IngestionCheckpoint`) so a failed run can resume from last successful batch.
- **Concurrency control**: single-writer per collection; no parallel upserts to the same collection.
- **Memory hygiene**: release page objects after each batch; no full-document retention.

### 5.4 Ingestion Policy Interface and Implementations

```python
# src/infrastructure/adapters/ingestion/normal_policy.py
class NormalIngestionPolicy(IIngestionPolicy):
    def should_use(self, page_count: int) -> bool:
        return page_count <= 50

    def batch_size(self) -> int:
        return 50  # whole document

    def chunk_document(self, document: Document) -> Sequence[ManualChunk]:
        # paragraph-level chunks with page metadata
        ...
```

```python
# src/infrastructure/adapters/ingestion/extended_policy.py
class ExtendedIngestionPolicy(IIngestionPolicy):
    def should_use(self, page_count: int) -> bool:
        return page_count > 50

    def batch_size(self) -> int:
        return 10  # pages per batch

    def chunk_document(self, document: Document) -> Sequence[ManualChunk]:
        # hierarchical section-aware chunks
        ...
```

### 5.5 Policy Selection

A `IngestionPolicySelector` (domain service) chooses the policy based on `page_count` and optional override from DTO:

```python
class IngestionPolicySelector:
    def select(self, page_count: int, override: str | None = None) -> IIngestionPolicy:
        if override == "normal":
            return NormalIngestionPolicy()
        if override == "extended":
            return ExtendedIngestionPolicy()
        return self._by_page_count(page_count)
```

### 5.6 Idempotency and Deduplication

- `DocumentId = uuid5(normalized_abs_path)` (or content hash for uploaded files).
- `ChunkId = uuid5(document_id, section_path, page, local_index)`.
- Before upsert, `IVectorRepository.exists_by_id` filters duplicates; count reported in `IngestionResultDTO.skipped_duplicates`.
- Re-ingestion of identical content is a no-op at point level.

### 5.7 Error Handling and Retry

- Per-file isolation: one corrupt file does not abort the batch.
- Retry policy: 3 attempts with exponential backoff for transient Qdrant/LLM errors.
- Checkpoint persisted after each successful batch; on failure, resume from last checkpoint.
- Final report includes failed files with reasons.

### 5.8 Ingestion Use Case (revised)

```python
class IngestManualUseCase:
    def __init__(
        self,
        extractor: IDocumentExtractor,
        chunker: IChunker,
        embedding: IEmbeddingService,
        repository: IVectorRepository,
        policy_selector: IngestionPolicySelector,
        checkpoint_store: ICheckpointStore,
    ) -> None: ...

    def execute(
        self, request: IngestionRequestDTO, progress_callback=None
    ) -> IngestionResultDTO: ...
```

---

## 6. Centralized Composition and Configuration

### 6.1 Strongly-Typed Settings

Target file: `src/infrastructure/config/settings.py`

```python
# src/infrastructure/config/settings.py
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    llm_endpoint: str = "http://127.0.0.1:1234/v1"
    llm_model_chat: str = "qwen3-14b"
    llm_model_vision: str = "qwen3-14b"
    llm_timeout_seconds: int = 60

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "technical_manuals"
    qdrant_timeout_seconds: float = 2.0

    # Embeddings
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_vector_dim: int = 1024
    hf_model_cache: str = "./.cache/models"

    # Ingestion
    ingestion_normal_page_limit: int = 50
    ingestion_batch_size_extended: int = 10
    ingestion_max_retries: int = 3

    # STT
    stt_model_size: str = "base"
    stt_device: str = "cpu"

    # App
    log_level: str = "INFO"
    gradio_host: str = "127.0.0.1"
    gradio_port: int = 7860
```

Rules:

- **No `os.getenv` outside this module.**
- Settings loaded once at Composition Root and injected.
- `.env.example` committed; `.env` git-ignored.

### 6.2 Composition Root

Target file: `src/infrastructure/composition/root.py`

```python
# src/infrastructure/composition/root.py
from src.infrastructure.config.settings import Settings
from src.infrastructure.adapters.local_llm_adapter import LocalLLMAdapter
from src.infrastructure.adapters.qdrant_adapter import QdrantAdapter
from src.infrastructure.adapters.embedding_adapter import SentenceTransformerEmbeddingAdapter
from src.infrastructure.adapters.pymupdf_adapter import PyMuPDFAdapter
from src.infrastructure.adapters.chunker_adapter import HierarchicalChunker
from src.infrastructure.adapters.stt_adapter import STTAdapter
from src.application.use_cases.ingest_manual import IngestManualUseCase
from src.application.use_cases.orchestrate_agent import OrchestrateAgentUseCase


class CompositionRoot:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def build(self) -> "ApplicationContext":
        embedding = SentenceTransformerEmbeddingAdapter(
            model_name=self.settings.embedding_model_name,
            cache_folder=self.settings.hf_model_cache,
            expected_dim=self.settings.embedding_vector_dim,
        )
        vector_repo = QdrantAdapter(
            settings=self.settings,
            embedding=embedding,
        )
        llm = LocalLLMAdapter(settings=self.settings)
        extractor = PyMuPDFAdapter()
        chunker = HierarchicalChunker()
        stt = STTAdapter(
            model_size=self.settings.stt_model_size,
            device=self.settings.stt_device,
        )
        ingest_uc = IngestManualUseCase(
            extractor=extractor,
            chunker=chunker,
            embedding=embedding,
            repository=vector_repo,
            settings=self.settings,
        )
        orchestrate_uc = OrchestrateAgentUseCase(
            llm_engine=llm,
            vector_repo=vector_repo,
            embedding=embedding,
        )
        return ApplicationContext(
            settings=self.settings,
            llm=llm,
            vector_repo=vector_repo,
            embedding=embedding,
            extractor=extractor,
            chunker=chunker,
            stt=stt,
            ingest_use_case=ingest_uc,
            orchestrate_use_case=orchestrate_uc,
        )
```

### 6.3 Presentation Wiring

`main.py` and `web_app.py` receive an `ApplicationContext` from the Composition Root; they never construct adapters or read env vars.

```python
# main.py (target)
from src.infrastructure.composition.root import CompositionRoot
from src.presentation.web_app import UserInterfaceApplication


def main() -> None:
    ctx = CompositionRoot().build()
    app = UserInterfaceApplication(ctx)
    interface = app.build_interface()
    interface.launch(server_name=ctx.settings.gradio_host, server_port=ctx.settings.gradio_port)
```

### 6.4 Configuration Environments

- `.env` (local, git-ignored) for dev.
- `.env.test` for tests (loaded via `Settings(_env_file=".env.test")`).
- Production via real environment variables (no file).
- No secrets in repository.

---

## 7. Execution and Finalization Protocol for OpenCode

### 7.1 Autonomy and Completeness

- OpenCode must execute the assigned task **to completion** before yielding control.
- Partial implementations are not acceptable; each task must end in a working, tested state.

### 7.2 Stop Criteria

OpenCode stops early **only** when:

- A critical divergence from this SAD is detected.
- A blocking ambiguity requires user confirmation.

In all other cases, proceed autonomously and document decisions in the PR description.

### 7.3 GitHub Delivery Cycle

- Work happens on `feature/*` branches from `develop`.
- Task is considered successfully completed upon `git push` of the corresponding `feature/*` branch to GitHub.
- User creates the Pull Request in the GitHub web UI and performs the final merge into `develop`.
- OpenCode never merges into `develop` or `main`.

### 7.4 Closure Governance

- Only the user (repository owner) creates the formal PR and executes the final merge.
- OpenCode must provide, at completion:
  1. A **detailed technical summary in Spanish**.
  2. The **PR title and description in English**, following international standards.

### 7.5 PR Template (English)

**Title format**: `<type>(<scope>): <short imperative subject>`

**Description template**:

```markdown
## Summary
<what changed and why>

## Changes
- ...

## Architecture Impact
- Layers touched: ...
- Ports/contracts changed: ...

## Testing
- Unit tests added/updated: ...
- Integration tests: ...
- Manual verification: ...

## Checklist
- [ ] Type hints on all new/changed signatures
- [ ] No empty try/except
- [ ] No os.getenv outside settings
- [ ] Conventional Commits followed
- [ ] Tests pass locally
- [ ] Documentation updated

## Breaking Changes
None | <description>
```

### 7.6 Acceptance Criteria for Any Task

- Code compiles and imports cleanly.
- `ruff` and `mypy --strict` pass on changed files.
- `pytest` passes; coverage does not decrease.
- No regression in existing functionality.
- PR description provided in English.

---

## 8. Current State vs. Target State (Gap Analysis)

| Area | Current State | Target State | Gap |
|---|---|---|---|
| Dependency mgmt | `requirements.txt`, `venv/` | `pyproject.toml` + `uv.lock` | High |
| Configuration | `os.getenv` scattered in `web_app.py`, `qdrant_adapter.py` | `Settings` + Composition Root | High |
| Embedding service | Embedded inside `QdrantAdapter` | Separate `IEmbeddingService` + adapter | High |
| Ports contracts | Primitive `str`/`int`, mutable lists | Value Objects / DTOs, frozen dataclasses | Medium |
| LLM port | `generate_response(context: str, ...)` but adapter takes `List[Dict]` | Align signature (`Sequence[str]` or DTO) | Medium |
| STT adapter | `STTAdapter` does not implement `IAudioService` | Implement port | Medium |
| Ingestion | Single path, integer IDs `idx`, no policy | Normal/Extended policies, UUID IDs, batching, checkpointing | High |
| Chunking | Paragraph split only | Hierarchical section-aware chunking | Medium |
| Idempotency | None (duplicate upserts) | Deterministic IDs + skip existing | High |
| Error handling | Broad `except Exception` in several places (acceptable if logged) but some empty-ish | Typed domain errors, no empty handlers | Medium |
| Tests | None in repo | Unit + integration suite | High |
| CI | None | Lint/type/test gates | High |
| Docs | README only | SAD + ADRs | Low |

### 8.1 Specific Code Smells Identified

- `QdrantAdapter.save_chunks` uses `id=idx` (positional integer) → collisions across runs and documents.
- `QdrantAdapter.__init__` reads `os.getenv("HF_MODEL_CACHE")` directly.
- `UserInterfaceApplication.__init__` reads multiple `os.getenv` calls.
- `LocalLLMAdapter.generate_response` signature (`context: str`) mismatches port (`context: str` ok) but use case passes `List[Dict]`; adapter internally expects list of dicts — type contract is loose (`str` declared, `List[Dict[str, Any]]` used). Must reconcile.
- `IngestManualUseCase` catches `Exception` and continues; acceptable for per-file resilience but must log with stack trace and not swallow silently.
- `STTAdapter.transcribe` returns `""` on error instead of raising `HardwareDegradationError` per port contract.
- `PyMuPDFAdapter` catches and re-raises as `DataIncompleteError` (good) but chunk IDs are non-deterministic-ish (file name + page + idx) and not UUID-based.

---

## 9. Refactoring Roadmap

### Phase 0 — Foundation (no behavior change)

- Add `pyproject.toml`, migrate deps with `uv`.
- Add `.env.example`.
- Add `Settings` and `CompositionRoot`; wire `main.py`/`web_app.py` to consume `ApplicationContext`.
- Remove scattered `os.getenv`.

### Phase 1 — Ports & Contracts

- Introduce Value Objects (`DocumentId`, `ChunkId`, `EmbeddingVector`).
- Extract `IEmbeddingService` + `SentenceTransformerEmbeddingAdapter`.
- Refactor `IVectorRepository` to accept vectors; update `QdrantAdapter`.
- Reconcile `ILLMEngine.generate_response` signature.
- Make `STTAdapter` implement `IAudioService`.

### Phase 2 — Ingestion

- Implement `IIngestionPolicy`, selector, normal & extended policies.
- Hierarchical chunker.
- Deterministic UUID IDs + idempotency.
- Checkpointing for extended ingestion.
- Revise `IngestManualUseCase`.

### Phase 3 — Orchestration & Quality

- Extract `SearchManualsUseCase`, `SynthesizeResponseUseCase`.
- Add typed error translation.
- Add `ruff`, `mypy`, `pytest` configuration and CI.

### Phase 4 — Testing & Hardening

- Unit tests for domain and application.
- Integration tests for adapters (Qdrant in-memory, LM Studio mocked).
- Coverage gate ≥ 80%.

### Phase 5 — Documentation & Release

- ADRs, README update, tag `v1.0.0`.

Each phase is a separate `feature/*` branch and PR into `develop`.

---

## 10. Quality, Testing, Observability, Security

### 10.1 Testing Strategy

- **Unit tests**: domain models, policies, selectors, use cases with mocked ports.
- **Integration tests**: adapters against real Qdrant (embedded) and mocked HTTP LLM.
- **Property-based tests** (optional): chunking invariants (no empty chunks, page ranges valid).
- **Smoke test**: end-to-end ingestion + query on a small fixture PDF.

Test layout: `tests/unit/...`, `tests/integration/...`, fixtures in `tests/fixtures/`.

### 10.2 Linting and Type Checking

`pyproject.toml` (target):

```toml
[tool.ruff]
line-length = 100
target-version = "py313"
select = ["E", "F", "I", "B", "UP", "N", "SIM", "RUF"]

[tool.mypy]
python_version = "3.13"
strict = true
packages = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=src --cov-fail-under=80"
```

### 10.3 Observability

- Structured logging (JSON in prod) with `correlation_id` per request.
- Audit log of searches: query, top-k scores, selected chunks.
- Ingestion progress callbacks report to UI.
- Health checks: `ILLMEngine.health_check`, Qdrant connectivity, embedding model load.

### 10.4 Security

- `.env` git-ignored; `.env.example` committed without secrets.
- Input validation at presentation boundary (file types, size limits).
- No `eval`/`exec`; no shell injection.
- Dependency vulnerability scanning via `uv` + CI.
- Local-only network by default; no data exfiltration.

### 10.5 Resilience

- Graceful degradation returns `OutboundResponseDTO(is_degraded=True)` instead of crashing.
- Timeouts on all HTTP calls (LLM, Qdrant).
- Retry with backoff for transient failures.
- Fallback: Qdrant remote → local embedded.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Large PDFs cause OOM | High | Extended policy batching + streaming |
| LLM hallucination | High | Strict grounding prompt + score threshold + citations |
| Duplicate vectors on re-ingest | Medium | Deterministic IDs + skip existing |
| Model download latency | Medium | HF cache dir, pre-warm in Composition Root |
| STT hardware variability | Medium | Implement `IAudioService`, degrade gracefully |
| Scope creep | Medium | Phased roadmap, one PR per phase |
| Breaking port changes | High | ADRs + migration notes in PRs |

---

## 12. Acceptance Criteria (SAD-level)

The architecture is considered adopted when:

1. `pyproject.toml` + `uv.lock` are the sole dependency source.
2. No `os.getenv` outside `src/infrastructure/config/settings.py`.
3. `domain/` has zero imports from other layers (verified by import-linter).
4. All ports accept Value Objects/DTOs; no primitive leakage for domain concepts.
5. Embedding service is a separate port/adapter.
6. Ingestion supports normal (≤50 pp) and extended (>50 pp) policies with batching, hierarchical chunking, stable IDs, idempotency.
7. Composition Root is the single construction site.
8. Test suite passes with ≥ 80% coverage on `domain/` and `application/`.
9. CI gates (ruff, mypy, pytest) pass on `develop`.

---

## 13. Glossary

- **RAG**: Retrieval-Augmented Generation.
- **Port**: Interface defined in the domain layer (ABC).
- **Adapter**: Concrete implementation of a port in infrastructure.
- **Value Object**: Immutable domain object identified by its value.
- **DTO**: Data Transfer Object crossing layer boundaries.
- **Composition Root**: Single place where the object graph is constructed.
- **Idempotent ingestion**: Re-running ingestion produces no duplicate vectors.
- **Hierarchical chunking**: Splitting documents by section/heading tree rather than flat paragraphs.

---

*End of Software Architecture Document — multidomain_rag v1.0.0*
