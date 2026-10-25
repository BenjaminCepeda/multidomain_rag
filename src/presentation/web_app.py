import os
import logging
import gradio as gr
from typing import Optional, List, Tuple
from src.infrastructure.adapters.local_llm_adapter import LocalLLMAdapter
from src.infrastructure.adapters.qdrant_adapter import QdrantAdapter
from src.infrastructure.adapters.pymupdf_adapter import PyMuPDFAdapter
from src.infrastructure.adapters.stt_adapter import STTAdapter
from src.application.use_cases.ingest_manual import IngestManualUseCase
from src.application.use_cases.orchestrate_agent import OrchestrateAgentUseCase
from src.application.dtos.agent_dtos import InboundPayloadDTO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

class UserInterfaceApplication:
    def __init__(self) -> None:
        self._llm_engine = LocalLLMAdapter(
            endpoint=os.getenv("LLM_ENDPOINT", "http://localhost:1234/v1"),
            model_chat=os.getenv("LLM_MODEL_CHAT", "qwen2.5-vl-3b-instruct"),
            model_vision=os.getenv("LLM_MODEL_VISION", "qwen2.5-vl-3b-instruct")
        )
        self._vector_repo = QdrantAdapter(
            location=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "technical_manuals"),
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
            vector_dim=int(os.getenv("EMBEDDING_VECTOR_DIM", "1024"))
        )
        self._extractor = PyMuPDFAdapter()
        self._stt_engine = STTAdapter(model_size="base")
        self._ingest_use_case = IngestManualUseCase(extractor=self._extractor, repository=self._vector_repo)
        self._orchestrate_use_case = OrchestrateAgentUseCase(llm_engine=self._llm_engine, vector_repo=self._vector_repo)

    def transcribe_audio(self, audio_path: str) -> str:
        if not audio_path: return ""
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return "❌ Error: Archivo de audio vacío o corrupto."
            
            try:
                from pydub import AudioSegment
                sound = AudioSegment.from_file(audio_path)
                sound = sound.set_frame_rate(16000).set_channels(1)
                normalized_path = audio_path + "_converted.wav"
                sound.export(normalized_path, format="wav")
                audio_path = normalized_path
            except Exception as pydub_err:
                logging.warning(f"No se pudo normalizar con pydub: {pydub_err}")

            return self._stt_engine.transcribe(audio_path)
        except Exception as e:
            return f"❌ Error STT: {str(e)}"

    def process_user_interaction(self, message: str, staged_image: Optional[str], history: List) -> Tuple[str, Optional[str], List]:
        text_content = (message or "").strip()
        if not text_content and not staged_image: return "", None, history
        try:
            outbound_dto = self._orchestrate_use_case.execute(InboundPayloadDTO(user_input=text_content, image_path=staged_image))
            history.append([text_content, outbound_dto.response_text])
        except Exception as e:
            history.append([text_content, f"❌ Error: {str(e)}"])
        return "", None, history

    def build_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Edge-AI Agent", css="src/presentation/assets/style.css") as interface:
            staged_image = gr.State(None)
            menu_visible = gr.State(False)
            is_recording = gr.State(False) 
            
            with gr.Tab("🎯 Operación"):
                chatbot = gr.Chatbot(label="Historial", height=450)
                
                with gr.Column(elem_classes="input-container"):
                    with gr.Row(visible=False, elem_classes="preview-container") as preview_row:
                        preview_img = gr.Image(label="", interactive=False, elem_classes="preview-thumb", visible=False)
                        btn_remove = gr.Button("✕", size="sm", elem_classes="minimal-btn")
                    
                    # El componente de audio conserva su ID para la inyección de JavaScript nativo
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", visible=False, elem_id="hidden-mic")
                    
                    with gr.Row(elem_classes="input-row"):
                        btn_plus = gr.Button("➕", elem_classes="minimal-btn")
                        txt_input = gr.Textbox(placeholder="Escribe tu consulta...", show_label=False, scale=5)
                        btn_mic = gr.Button("🎤", elem_classes="minimal-btn")
                        btn_send = gr.Button("➤", elem_classes="minimal-btn")

                    with gr.Group(visible=False, elem_classes="menu-popup") as action_menu:
                        btn_upload = gr.UploadButton("📁 Subir Archivo", file_types=["image"])
                        btn_open_cam = gr.Button("📷 Abrir Cámara")

                with gr.Group(visible=False, elem_classes="camera-modal") as cam_modal:
                    gr.Markdown("### 📷 Captura de Imagen desde Webcam")
                    cam_input = gr.Image(sources=["webcam"], label=None, type="filepath")
                    btn_close_cam = gr.Button("Cerrar Ventana", variant="stop")

                # --- INTERACCIÓN DE AUDIO MEJORADA (RESET INFINITO Y REEMPLAZO) ---
                def toggle_menu(is_visible):
                    return not is_visible, gr.update(visible=not is_visible)
                
                btn_plus.click(toggle_menu, inputs=[menu_visible], outputs=[menu_visible, action_menu])
                
                toggle_mic_js = """
                () => {
                    const recordBtn = document.querySelector("#hidden-mic button[aria-label='Record audio'], #hidden-mic button[class*='record']");
                    const stopBtn = document.querySelector("#hidden-mic button[aria-label='Stop recording'], #hidden-mic button[class*='stop']");
                    
                    if (recordBtn) {
                        recordBtn.click();
                    } else if (stopBtn) {
                        stopBtn.click();
                    }
                }
                """

                def handle_mic_state(recording):
                    if not recording:
                        return True, gr.update(value="🛑", variant="stop")
                    else:
                        return False, gr.update(value="🎤", variant="secondary")

                btn_mic.click(
                    handle_mic_state, 
                    inputs=[is_recording], 
                    outputs=[is_recording, btn_mic]
                ).then(None, js=toggle_mic_js)
                
                def post_transcription_reset(path):
                    if not path:
                        # Si el cambio fue provocado por el limpiador automático, ignorar para evitar bucles
                        return gr.update(), gr.update(), gr.update(), gr.update()
                    
                    # Genera la transcripción en limpio
                    texto_detectado = self.transcribe_audio(path)
                    
                    # RETORNO CLAVE: 
                    # 1. texto_detectado reemplaza por completo el valor en txt_input.
                    # 2. Desactiva el flag lógico (False).
                    # 3. Restaura el icono original del botón ("🎤").
                    # 4. Inyecta un None directo sobre el propio audio_input para resetear el buffer físico del navegador.
                    return texto_detectado, False, gr.update(value="🎤", variant="secondary"), None

                audio_input.change(
                    post_transcription_reset,
                    inputs=[audio_input],
                    outputs=[txt_input, is_recording, btn_mic, audio_input]
                )
                
                # Manejadores de Imágenes
                def handle_file_input(file_obj):
                    if file_obj is None: 
                        return None, False, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                    return file_obj.name, False, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True, value=file_obj.name)
                
                def handle_cam_input(img_path):
                    if img_path is None: 
                        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                    return img_path, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True, value=img_path), gr.update(visible=False)

                btn_upload.upload(
                    handle_file_input, 
                    inputs=[btn_upload], 
                    outputs=[staged_image, menu_visible, action_menu, preview_row, preview_img]
                )
                
                cam_input.change(
                    handle_cam_input, 
                    inputs=[cam_input], 
                    outputs=[staged_image, cam_modal, preview_row, preview_img, action_menu]
                )
                
                btn_open_cam.click(lambda: (False, gr.update(visible=False), gr.update(visible=True)), None, [menu_visible, action_menu, cam_modal])
                btn_close_cam.click(lambda: gr.update(visible=False), None, [cam_modal])
                
                btn_remove.click(
                    lambda: (None, gr.update(visible=False), gr.update(visible=False)),
                    None,
                    [staged_image, preview_row, preview_img]
                )
                
                btn_send.click(self.process_user_interaction, [txt_input, staged_image, chatbot], [txt_input, staged_image, chatbot])
        return interface

if __name__ == "__main__":
    UserInterfaceApplication().build_interface().launch(server_name="127.0.0.1", server_port=7860)