from prometheus_client import Histogram, Counter, Gauge
import psutil
import time
import threading

INFERENCE_TIME_HISTOGRAM = Histogram(
    "inference_latency_seconds",
    "Time spent processing prediction requests"
)

REQUEST_COUNT = Counter("api_request_count", "Total number of API requests")
REQUEST_ERRORS = Counter("api_request_errors", "Number of failed API predictions")

CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage percentage")
RAM_USAGE = Gauge("ram_usage_percent", "RAM usage percentage")

UPTIME = Gauge("app_uptime_seconds", "App uptime in seconds")
START_TIME = time.time()

def update_system_metrics():
    while True:
        try:
            CPU_USAGE.set(psutil.cpu_percent(interval=1))
            RAM_USAGE.set(psutil.virtual_memory().percent)
            UPTIME.set(time.time() - START_TIME)
            time.sleep(5)
        except Exception:
            continue

def start_metrics_collection():
    thread = threading.Thread(target=update_system_metrics, daemon=True)
    thread.start()
