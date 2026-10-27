# src/infrastructure/hardware/device_factory.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    provider: str
    device: str
    available: bool
    reason: str | None = None


class DeviceFactory:
    """Factory for hardware execution providers with automatic CPU fallback."""

    @staticmethod
    def get_preferred_providers() -> list[tuple[str, str]]:
        providers: list[tuple[str, str]] = []
        # NPU/GPU first, CPU last - prefer DirectML for AMD/Intel integrated NPU+GPU
        providers.append(("DmlExecutionProvider", "directml"))
        providers.append(("DirectMLExecutionProvider", "directml"))
        providers.append(("OpenVINOExecutionProvider", "openvino"))
        providers.append(("CUDAExecutionProvider", "cuda"))
        providers.append(("ROCMExecutionProvider", "rocm"))
        providers.append(("AzureExecutionProvider", "azure"))
        providers.append(("CPUExecutionProvider", "cpu"))
        return providers

    @staticmethod
    def detect_available_providers() -> list[DeviceInfo]:
        devices: list[DeviceInfo] = []
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            logger.info("ORT_AVAILABLE_PROVIDERS: %s", available)
            for provider, dev in DeviceFactory.get_preferred_providers():
                if provider in available:
                    devices.append(DeviceInfo(provider=provider, device=dev, available=True))
                else:
                    devices.append(
                        DeviceInfo(
                            provider=provider,
                            device=dev,
                            available=False,
                            reason="not_available_in_ort",
                        )
                    )
        except Exception as exc:
            logger.warning("Device detection via onnxruntime failed: %s", exc)
            devices.append(
                DeviceInfo(
                    provider="CPUExecutionProvider", device="cpu", available=True, reason=None
                )
            )
        return devices

    @classmethod
    def select_device(cls) -> tuple[str, str]:
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            logger.info("HARDWARE_AUDIT | ORT_AVAILABLE_PROVIDERS=%s", available)
            for provider, dev in cls.get_preferred_providers():
                if provider != "CPUExecutionProvider" and provider in available:
                    logger.info(
                        "HARDWARE_AUDIT | SELECTED_PROVIDER=%s | DEVICE=%s | PRIORITY=accelerated",
                        provider,
                        dev,
                    )
                    return provider, dev
            logger.info(
                "HARDWARE_AUDIT | SELECTED_PROVIDER=CPUExecutionProvider | "
                "DEVICE=cpu | PRIORITY=fallback"
            )
        except Exception as exc:
            logger.warning("Falling back to CPU due to device detection error: %s", exc)
        return "CPUExecutionProvider", "cpu"

    @classmethod
    def configure_sentence_transformers_device(cls) -> str:
        """Return device string for sentence-transformers (cpu/cuda/directml)."""
        try:
            provider, dev = cls.select_device()
            if provider == "CPUExecutionProvider":
                return "cpu"
            if dev == "cuda":
                return "cuda"
            if dev == "directml":
                return "cpu"  # sentence-transformers uses PyTorch; try cpu safe path
        except Exception as exc:
            logger.warning("SentenceTransformer device selection failed, using cpu: %s", exc)
        return "cpu"

    @classmethod
    def get_ort_session_options(cls) -> Any:
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.getenv("ORT_INTRA_THREADS", "0")) or 0
            opts.inter_op_num_threads = int(os.getenv("ORT_INTER_THREADS", "0")) or 0
            return opts
        except Exception:
            return None
