# src/infrastructure/adapters/local_llm_adapter.py
import base64
import json
import os
import re
from collections.abc import Sequence

import requests

from src.domain.exceptions.core_errors import DataIncompleteError, LLMInferenceError
from src.domain.models.agent import OrchestratorDecision
from src.domain.ports.llm_engine import ILLMEngine
from src.infrastructure.config.settings import Settings


class LocalLLMAdapter(ILLMEngine):
    def __init__(
        self,
        settings: Settings | None = None,
        endpoint: str | None = None,
        model_chat: str | None = None,
        model_vision: str | None = None,
        timeout: int | None = None,
    ) -> None:
        if settings is not None:
            endpoint = endpoint or settings.llm_endpoint
            model_chat = model_chat or settings.llm_model_chat
            model_vision = model_vision or settings.llm_model_vision
            timeout = timeout or settings.llm_timeout_seconds

        self._endpoint = (endpoint or "http://127.0.0.1:1234/v1").rstrip("/")
        self._model_chat = model_chat or "qwen3-14b"
        self._model_vision = model_vision or "qwen3-14b"
        self._timeout = timeout or 60
        self._headers = {"Content-Type": "application/json"}

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self._endpoint}/models", timeout=3)
            return int(response.status_code) == 200
        except Exception:
            return False

    def health_check(self) -> bool:
        return self.check_health()

    def get_orchestrator_decision(self, user_input: str, has_image: bool) -> OrchestratorDecision:
        if not self.check_health():
            raise LLMInferenceError("El motor LLM no esta disponible.")

        url = f"{self._endpoint}/chat/completions"
        prompt_sistema = (
            "Eres un cerebro orquestador. Responde SOLO con un objeto JSON. "
            '{"razonamiento": "motivo", "herramienta": "nombre", "parametro": "texto_o_null"}'
        )
        payload = {
            "model": self._model_chat,
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Operario: {user_input} | Imagen: {has_image}"},
            ],
            "temperature": 0.0,
        }

        try:
            response = requests.post(
                url, json=payload, headers=self._headers, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise LLMInferenceError("Timeout en inferencia del LLM local.", exc) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMInferenceError("No se pudo conectar al LLM local.", exc) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc
        raw_content = str(response.json()["choices"][0]["message"]["content"]).strip()

        clean_content = re.sub(r"^```json\s*|\s*```$", "", raw_content, flags=re.MULTILINE)
        start = clean_content.find("{")
        end = -1
        count = 0
        for i, char in enumerate(clean_content[start:]):
            if char == "{":
                count += 1
            elif char == "}":
                count -= 1
            if count == 0:
                end = start + i + 1
                break

        if start == -1 or end == -1:
            raise LLMInferenceError(
                f"El motor no devolvio un JSON valido. Contenido: {raw_content}"
            )

        data = json.loads(clean_content[start:end])
        return OrchestratorDecision(
            razonamiento=data.get("razonamiento", "Sin descripcion"),
            herramienta=data.get("herramienta", "responder_directo"),
            parametro=data.get("parametro"),
        )

    def generate_response(self, context: Sequence[str], user_input: str) -> str:
        if not self.check_health():
            raise LLMInferenceError("El motor LLM no esta disponible.")

        if not context:
            return "No hay informacion relevante en los documentos para responder."

        url = f"{self._endpoint}/chat/completions"

        context_str = "\n\n".join(context)

        system_prompt = (
            "Eres un asistente tecnico experto para manuales de impresoras Epson. "
            "Tu unica fuente de verdad es el Contexto tecnico que se te proporciona. "
            "Reglas estrictas:\n"
            "- RESPONDE SIEMPRE con base en el contexto. Si hay informacion relevante, "
            "aunque sea parcial, usala para elaborar una respuesta util y precisa.\n"
            "- NO digas 'No dispongo de informacion' si la respuesta se puede inferir "
            "o encontrar parcialmente en los fragmentos recuperados.\n"
            "- Explica paso a paso, procedimientos, rutas de menu, nombres de programas "
            "(Epson Scan 2, Epson Email Print) o configuraciones cuando aparezcan.\n"
            "- Cita la fuente y la pagina al final de las afirmaciones clave usando "
            "el formato [Archivo, Pag].\n"
            "- No inventes pasos, nombres o valores. Solo si tras revisar TODO el contexto "
            "no hay absolutamente nada relacionado, usa exactamente: "
            "'No dispongo de informacion en los manuales'.\n"
            "- Prioriza utilidad practica para el usuario sobre la cautela excesiva."
        )

        user_msg = (
            f"Contexto tecnico:\n{context_str}\n\n"
            f"Pregunta del usuario:\n{user_input}\n\n"
            "Instruccion: Usa UNICAMENTE el contexto anterior. Si existe informacion relacionada, "
            "genera una respuesta completa y accionable con referencias [Archivo, Pag]."
        )
        payload = {
            "model": self._model_chat,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        try:
            response = requests.post(
                url, json=payload, headers=self._headers, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise LLMInferenceError("Timeout en inferencia del LLM local.", exc) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMInferenceError("No se pudo conectar al LLM local.", exc) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc
        return str(response.json()["choices"][0]["message"]["content"]).strip()

    def analyze_image(self, image_path: str) -> str:
        if not self.check_health():
            raise LLMInferenceError("El motor LLM no esta disponible.")

        if not os.path.exists(image_path):
            raise DataIncompleteError(f"Archivo no encontrado: {image_path}")
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        url = f"{self._endpoint}/chat/completions"
        payload = {
            "model": self._model_vision,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe este diagrama tecnico brevemente."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
        }
        try:
            response = requests.post(
                url, json=payload, headers=self._headers, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise LLMInferenceError("Timeout en inferencia del LLM local.", exc) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMInferenceError("No se pudo conectar al LLM local.", exc) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc
        return str(response.json()["choices"][0]["message"]["content"]).strip()
