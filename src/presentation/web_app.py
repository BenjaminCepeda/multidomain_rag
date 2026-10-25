import contextlib
import logging
import os
from pathlib import Path
from typing import Any

import gradio as gr

from src.application.dtos.agent_dtos import InboundPayloadDTO
from src.application.dtos.ingestion_dtos import IngestionRequestDTO
from src.domain.exceptions.core_errors import DomainBaseError
from src.infrastructure.composition.root import ApplicationContext, CompositionRoot
from src.presentation.ui_state_presenter import UIStatePresenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)  # noqa: E501


class UserInterfaceApplication:
    def __init__(self, context: ApplicationContext | None = None) -> None:
        self._ctx = context or CompositionRoot().build()
        self._llm_engine = self._ctx.llm
        self._vector_repo = self._ctx.vector_repo
        self._extractor = self._ctx.extractor
        self._stt_engine = self._ctx.stt
        self._ingest_use_case = self._ctx.ingest_use_case
        self._orchestrate_use_case = self._ctx.orchestrate_use_case
        self._presenter = UIStatePresenter(self._ctx)

    def transcribe_audio(self, audio_path: str) -> str:
        if not audio_path:
            return ""
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return "Error: Archivo de audio vacio o corrupto."

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
            return f"Error STT: {e!s}"

    def process_user_interaction(
        self, message: str, staged_image: str | None, history: list[Any]
    ) -> tuple[str, str | None, list[Any], str]:
        text_content = (message or "").strip()
        if not text_content and not staged_image:
            return "", None, history, ""

        status_text = ""
        try:
            outbound_dto = self._orchestrate_use_case.execute(
                InboundPayloadDTO(user_input=text_content, image_path=staged_image)
            )
            history.append([text_content, outbound_dto.response_text])
            status_text = (
                f"Herramienta: {outbound_dto.used_tool} | "
                f"Degradado: {'Si' if outbound_dto.is_degraded else 'No'}"
            )
            if outbound_dto.is_degraded and outbound_dto.error_log:
                status_text += f" | Error: {outbound_dto.error_log}"
        except DomainBaseError as e:
            history.append([text_content, f"Error del dominio: {e.message}"])
            status_text = f"Error: {e.message}"
        except Exception as e:
            history.append([text_content, f"Error: {e!s}"])
            status_text = f"Error inesperado: {e!s}"
        return "", None, history, status_text

    def refresh_status(self) -> str:
        return self._presenter.build_status()

    def run_ingestion(
        self, files: Any, policy: str, progress: gr.Progress = gr.Progress()  # noqa: B008
    ) -> str:
        if not files:
            return "No se seleccionaron archivos."

        file_list = files if isinstance(files, list) else [files]
        paths: list[Path] = []
        for f in file_list:
            path_str = getattr(f, "name", None) or str(f)
            paths.append(Path(path_str))

        progress(0, desc="Preparando ingesta...")

        def progress_callback(value: float, desc: str) -> None:
            with contextlib.suppress(Exception):
                progress(value, desc=desc)

        policy_override = None if policy == "auto" else policy
        request = IngestionRequestDTO(
            file_paths=tuple(paths),
            policy_override=policy_override,
        )

        try:
            result = self._ingest_use_case.execute(request, progress_callback=progress_callback)
        except Exception as e:
            return f"Error durante la ingesta: {e!s}"

        lines = [
            f"Documentos procesados: {result.documents_processed}",
            f"Fragmentos indexados: {result.chunks_indexed}",
            f"Duplicados omitidos: {result.skipped_duplicates}",
            f"Archivos fallidos: {len(result.failed_files)}",
        ]
        if result.failed_files:
            lines.append("")
            lines.append("Errores por archivo:")
            for failed in result.failed_files:
                lines.append(f"- {failed}")
        return "\n".join(lines)

    def build_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Edge-AI Agent", css="src/presentation/assets/style.css") as interface:
            staged_image = gr.State(None)
            menu_visible = gr.State(False)
            is_recording = gr.State(False)

            with gr.Tab("Estado"):
                status_box = gr.Textbox(label="Estado de servicios", lines=6, interactive=False)
                refresh_btn = gr.Button("Refrescar estado")
                refresh_btn.click(self.refresh_status, None, [status_box])
                interface.load(self.refresh_status, None, [status_box])

            with gr.Tab("Operacion"):
                chatbot = gr.Chatbot(label="Historial", height=450)
                last_status = gr.Textbox(label="Estado del turno", lines=2, interactive=False)

                with gr.Column(elem_classes="input-container"):
                    with gr.Row(visible=False, elem_classes="preview-container") as preview_row:
                        preview_img = gr.Image(
                            label="", interactive=False, elem_classes="preview-thumb", visible=False
                        )
                        btn_remove = gr.Button("X", size="sm", elem_classes="minimal-btn")

                    audio_input = gr.Audio(sources=["microphone"], type="filepath", visible=False, elem_id="hidden-mic")  # noqa: E501

                    with gr.Row(elem_classes="input-row"):
                        btn_plus = gr.Button("+", elem_classes="minimal-btn")
                        txt_input = gr.Textbox(placeholder="Escribe tu consulta...", show_label=False, scale=5)  # noqa: E501
                        btn_mic = gr.Button("Mic", elem_classes="minimal-btn")
                        btn_send = gr.Button("Send", elem_classes="minimal-btn")

                    with gr.Group(visible=False, elem_classes="menu-popup") as action_menu:
                        btn_upload = gr.UploadButton("Subir Archivo", file_types=["image"])
                        btn_open_cam = gr.Button("Abrir Camara")

                with gr.Group(visible=False, elem_classes="camera-modal") as cam_modal:
                    gr.Markdown("### Captura de Imagen desde Webcam")
                    cam_input = gr.Image(sources=["webcam"], label=None, type="filepath")
                    btn_close_cam = gr.Button("Cerrar Ventana", variant="stop")

                def toggle_menu(is_visible: bool) -> tuple[bool, Any]:
                    return not is_visible, gr.update(visible=not is_visible)

                btn_plus.click(toggle_menu, inputs=[menu_visible], outputs=[menu_visible, action_menu])  # noqa: E501

                toggle_mic_js = """
                () => {
                    const recordBtn = document.querySelector(
                        "#hidden-mic button[aria-label='Record audio'], " +
                        "#hidden-mic button[class*='record']"
                    );
                    const stopBtn = document.querySelector(
                        "#hidden-mic button[aria-label='Stop recording'], " +
                        "#hidden-mic button[class*='stop']"
                    );
                    if (recordBtn) {
                        recordBtn.click();
                    } else if (stopBtn) {
                        stopBtn.click();
                    }
                }
                """

                def handle_mic_state(recording: bool) -> tuple[bool, Any]:
                    if not recording:
                        return True, gr.update(value="Stop", variant="stop")
                    return False, gr.update(value="Mic", variant="secondary")

                btn_mic.click(
                    handle_mic_state, inputs=[is_recording], outputs=[is_recording, btn_mic]
                ).then(None, js=toggle_mic_js)

                def post_transcription_reset(path: str | None) -> tuple[Any, Any, Any, Any]:
                    if not path:
                        return gr.update(), gr.update(), gr.update(), gr.update()
                    texto_detectado = self.transcribe_audio(path)
                    return texto_detectado, False, gr.update(value="Mic", variant="secondary"), None

                audio_input.change(
                    post_transcription_reset,
                    inputs=[audio_input],
                    outputs=[txt_input, is_recording, btn_mic, audio_input],
                )

                def handle_file_input(file_obj: Any) -> tuple[Any, bool, Any, Any, Any]:
                    if file_obj is None:
                        return (
                            None,
                            False,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )
                    return (
                        file_obj.name,
                        False,
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True, value=file_obj.name),
                    )

                def handle_cam_input(img_path: Any) -> tuple[Any, Any, Any, Any, Any]:
                    if img_path is None:
                        return (
                            None,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )
                    return (
                        img_path,
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True, value=img_path),
                        gr.update(visible=False),
                    )

                btn_upload.upload(
                    handle_file_input,
                    inputs=[btn_upload],
                    outputs=[staged_image, menu_visible, action_menu, preview_row, preview_img],
                )

                cam_input.change(
                    handle_cam_input,
                    inputs=[cam_input],
                    outputs=[staged_image, cam_modal, preview_row, preview_img, action_menu],
                )

                btn_open_cam.click(
                    lambda: (False, gr.update(visible=False), gr.update(visible=True)),
                    None,
                    [menu_visible, action_menu, cam_modal],
                )
                btn_close_cam.click(lambda: gr.update(visible=False), None, [cam_modal])

                btn_remove.click(
                    lambda: (None, gr.update(visible=False), gr.update(visible=False)),
                    None,
                    [staged_image, preview_row, preview_img],
                )

                btn_send.click(
                    self.process_user_interaction,
                    [txt_input, staged_image, chatbot],
                    [txt_input, staged_image, chatbot, last_status],
                )

            with gr.Tab("Ingesta"):
                ingest_files = gr.File(
                    label="Documentos (PDF/TXT)", file_count="multiple", type="filepath"
                )
                policy_dropdown = gr.Dropdown(
                    choices=["auto", "normal", "extended"],
                    value="auto",
                    label="Politica de ingesta",
                )
                ingest_btn = gr.Button("Ingestar documentos")
                ingest_result = gr.Textbox(label="Resultado de ingesta", lines=10, interactive=False)  # noqa: E501

                ingest_btn.click(
                    self.run_ingestion,
                    inputs=[ingest_files, policy_dropdown],
                    outputs=[ingest_result],
                )

        return interface  # type: ignore[no-any-return]

    def __del__(self) -> None:  # pragma: no cover
        pass


if __name__ == "__main__":
    UserInterfaceApplication().build_interface().launch(server_name="127.0.0.1", server_port=7860)
