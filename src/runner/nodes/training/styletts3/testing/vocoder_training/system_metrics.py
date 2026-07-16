from __future__ import annotations

import psutil
import pynvml

BYTES_PER_GIBIBYTE = 1024**3


class SystemMetricsSampler:
    """Sample host, process, and GPU utilization at a wall-clock interval."""

    def __init__(self, device_index: int, interval_seconds: float, started_at: float) -> None:
        pynvml.nvmlInit()
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.last_sample_at = started_at
        psutil.cpu_percent()

    def sample(self, now: float) -> dict[str, float]:
        if now - self.last_sample_at < self.interval_seconds:
            return {}
        self.last_sample_at = now
        system_memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu)
        return {
            "cpu_utilization_percent": psutil.cpu_percent(),
            "memory_utilization_percent": system_memory.percent,
            "process_rss_gb": process_memory.rss / BYTES_PER_GIBIBYTE,
            "gpu_utilization_percent": float(gpu_utilization.gpu),
            "gpu_memory_utilization_percent": float(gpu_utilization.memory),
            "gpu_memory_used_gb": gpu_memory.used / BYTES_PER_GIBIBYTE,
            "gpu_temperature_celsius": float(
                pynvml.nvmlDeviceGetTemperature(self.gpu, pynvml.NVML_TEMPERATURE_GPU)
            ),
            "gpu_power_watts": pynvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000,
        }
