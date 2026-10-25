# src/infrastructure/adapters/local_llm_adapter.py
import json
import re
import requests
import base64
import os
from typing import List, Dict, Any
from src.domain.ports.llm_engine import ILLMEngine
from src.domain.models.agent import OrchestratorDecision
from src.domain.exceptions.core_errors import LLMInferenceError, DataIncompleteError

class LocalLLMAdapter(ILLMEngine):
    def __init__(self, endpoint: str, model_chat: str, model_vision: str) -> None:
        self._endpoint = endpoint.rstrip('/')
        self._model_chat = model_chat
        self._model_vision = model_vision
        self._headers = {"Content-Type": "application/json"}

    def check_health(self) -> bool:
        """Verifica si el motor LLM está operativo."""
        try:
            response = requests.get(f"{self._endpoint}/models", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def get_orchestrator_decision(self, user_input: str, has_image: bool) -> OrchestratorDecision:
        if not self.check_health():
            raise LLMInferenceError("El motor LLM no está disponible.")

        url = f"{self._endpoint}/chat/completions"
        prompt_sistema = (
            "Eres un cerebro orquestador. Responde SOLO con un objeto JSON. "
            '{"razonamiento": "motivo", "herramienta": "nombre", "parametro": "texto_o_null"}'
        )
        payload = {
            "model": self._model_chat,
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Operario: {user_input} | Imagen: {has_image}"}
            ],
            "temperature": 0.0
        }
        
        response = requests.post(url, json=payload, headers=self._headers, timeout=20)
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"].strip()
        
        clean_content = re.sub(r'^```json\s*|\s*```$', '', raw_content, flags=re.MULTILINE)
        start = clean_content.find('{')
        end = -1
        count = 0
        for i, char in enumerate(clean_content[start:]):
            if char == '{': count += 1
            elif char == '}': count -= 1
            if count == 0:
                end = start + i + 1
                break
        
        if start == -1 or end == -1:
            raise LLMInferenceError(f"El motor no devolvió un JSON válido. Contenido: {raw_content}")
            
        data = json.loads(clean_content[start:end])
        return OrchestratorDecision(
            razonamiento=data.get("razonamiento", "Sin descripción"),
            herramienta=data.get("herramienta", "responder_directo"),
            parametro=data.get("parametro")
        )

    def generate_response(self, context: List[Dict[str, Any]], user_input: str) -> str:
        """Genera respuesta garantizando veracidad mediante contexto estricto."""
        if not self.check_health():
            raise LLMInferenceError("El motor LLM no está disponible.")
            
        if not context:
            return "No hay información relevante en los documentos para responder."

        url = f"{self._endpoint}/chat/completions"
        
        context_str = "\n\n".join([
            f"[FUENTE: {os.path.basename(c.get('metadata', {}).get('source', 'Desconocido'))} | PÁGINA: {c.get('metadata', {}).get('page', 'N/A')}]\n{c.get('content', '')}"
            for c in context
        ])
        
        system_prompt = (
            "Eres un asistente técnico experto. Tu objetivo es la precisión absoluta. "
            "1. Responde ÚNICAMENTE basándote en el 'Contexto técnico' proporcionado. "
            "2. Si la información no está en el contexto, di explícitamente: 'No dispongo de información en los manuales'. "
            "3. DEBES incluir la fuente y página al final de cada frase técnica usando el formato: [Archivo, Pág]. "
            "4. NUNCA inventes información ni alucines referencias. Si no puedes verificarlo en el contexto, no lo incluyas."
        )
        
        payload = {
            "model": self._model_chat,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexto técnico:\n{context_str}\n\nPregunta: {user_input}"}
            ],
            "temperature": 0.1,
            "max_tokens": 800
        }
        
        response = requests.post(url, json=payload, headers=self._headers, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def analyze_image(self, image_path: str) -> str:
        if not self.check_health():
            raise LLMInferenceError("El motor LLM no está disponible.")
            
        if not os.path.exists(image_path):
            raise DataIncompleteError(f"Archivo no encontrado: {image_path}")
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        url = f"{self._endpoint}/chat/completions"
        payload = {
            "model": self._model_vision,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe este diagrama técnico brevemente."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            "temperature": 0.1
        }
        response = requests.post(url, json=payload, headers=self._headers, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()