# Multidomain RAG Engine Justfile

# Sincroniza el entorno con uv de forma estricta
sync:
	uv sync --frozen --all-extras

# Ejecuta el gate de calidad estructural (linting y tipado estricto)
gate: lint typecheck
	@echo "Gate estructural completo en verde."

# Análisis estático de código con Ruff
lint:
	uv run ruff check --output-format=concise .
	uv run ruff format --check .

# Verificación de tipos estricta con Mypy
typecheck:
	uv run mypy

# Ejecución de pruebas y reporte de cobertura (exigirá >=80% en fases de remediación)
test:
	uv run pytest -q