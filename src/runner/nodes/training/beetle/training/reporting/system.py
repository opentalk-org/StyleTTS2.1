import psutil
import pynvml

from .metrics import TrainingMetric

_BYTES_PER_GIBIBYTE = 1024**3


class SystemMetricsSampler:
    def __init__(self, device_index: int) -> None:
        if device_index < 0:
            raise ValueError("GPU device index must be non-negative")
        pynvml.nvmlInit()
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.process = psutil.Process()
        psutil.cpu_percent()

    def sample(self) -> tuple[TrainingMetric, ...]:
        system_memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu)
        return (
            TrainingMetric(
                "system/cpu_utilization_percent",
                psutil.cpu_percent(),
            ),
            TrainingMetric(
                "system/memory_utilization_percent",
                system_memory.percent,
            ),
            TrainingMetric(
                "system/process_rss_gb",
                process_memory.rss / _BYTES_PER_GIBIBYTE,
            ),
            TrainingMetric(
                "system/gpu_utilization_percent",
                float(gpu_utilization.gpu),
            ),
            TrainingMetric(
                "system/gpu_memory_utilization_percent",
                float(gpu_utilization.memory),
            ),
            TrainingMetric(
                "system/gpu_memory_used_gb",
                gpu_memory.used / _BYTES_PER_GIBIBYTE,
            ),
            TrainingMetric(
                "system/gpu_temperature_celsius",
                float(
                    pynvml.nvmlDeviceGetTemperature(
                        self.gpu,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                ),
            ),
            TrainingMetric(
                "system/gpu_power_watts",
                pynvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000,
            ),
        )
