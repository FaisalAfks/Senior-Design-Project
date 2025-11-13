"""Jetson power logging utilities backed by jetson-stats (jtop)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from jtop import jtop as _jtop_context  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _jtop_context = None

SOC_RAIL_TOKENS = ("CPU", "GPU", "SOC", "CV")


def _scaled(value: Any, scale: float) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / scale
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ChannelReading:
    rail: str
    power_w: Optional[float]
    avg_power_w: Optional[float]
    voltage_v: Optional[float]
    current_a: Optional[float]
    online: Optional[bool]
    sensor_type: Optional[str]
    status: Optional[str]


@dataclass
class ProcessPowerBreakdown:
    pid: Optional[int]
    name: Optional[str]
    user: Optional[str]
    cpu_percent: Optional[float]
    ram_mb: Optional[float]
    gpu_mem_mb: Optional[float]
    cpu_share: Optional[float]
    gpu_share: Optional[float]
    estimated_power_w: Optional[float]
    method: str


@dataclass
class PowerSample:
    timestamp: float
    activity: str
    elapsed_s: float
    total_name: Optional[str]
    total_power_w: Optional[float]
    total_avg_power_w: Optional[float]
    soc_power_w: Optional[float] = None
    process: Optional[ProcessPowerBreakdown] = None
    rails: List[ChannelReading] = field(default_factory=list)


class JetsonPowerLogger:
    """Background sampler that records Jetson power telemetry via jtop."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_path: Path,
        sample_interval: float = 1.0,
        verbose: bool = True,
        process_pid: Optional[int] = None,
        process_name: Optional[str] = None,
        process_cpu_weight: float = 0.6,
        process_gpu_weight: float = 0.4,
    ) -> None:
        self._requested = enabled
        self.available = _jtop_context is not None
        self.enabled = self._requested and self.available
        self.log_path = log_path
        self.sample_interval = max(sample_interval, 0.1)
        self.verbose = verbose
        self._last_error: Optional[str] = None
        self.process_pid = process_pid if process_pid is not None else os.getpid()
        script_name = sys.argv[0] if sys.argv else ""
        default_process_name = Path(script_name).stem if script_name else "process"
        self.process_name = process_name or default_process_name or "process"
        self.process_cpu_weight = max(0.0, process_cpu_weight)
        self.process_gpu_weight = max(0.0, process_gpu_weight)
        if self.process_cpu_weight == 0.0 and self.process_gpu_weight == 0.0:
            self.process_cpu_weight = 1.0
        self._process_error_reported = False

        if self._requested and not self.enabled and self.verbose:
            if _jtop_context is None:
                msg = "jetson-stats (jtop) is not installed"
            else:
                msg = "jtop backend is unavailable"
            print(f"[Power] Jetson power logging disabled: {msg}.")

        self._lock = threading.Lock()
        self._activity = "initializing"
        self._activity_events: List[dict[str, float | str]] = []
        self._samples: List[PowerSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_timestamp: Optional[float] = None
        self._end_timestamp: Optional[float] = None
        self._metadata_context: Dict[str, Any] = {}

    def __enter__(self) -> "JetsonPowerLogger":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._start_timestamp = time.time()
        self._record_activity_event("initializing", self._start_timestamp)
        self._thread = threading.Thread(target=self._run_loop, name="JetsonPowerLogger", daemon=True)
        self._thread.start()
        if self.verbose:
            print(f"[Power] Logging enabled at {self.sample_interval:.2f}s intervals -> {self.log_path}")

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._end_timestamp = time.time()
        self._record_activity_event("shutdown", self._end_timestamp)
        self._write_log()
        if self.verbose:
            if self._last_error:
                print(f"[Power] Logging finished with {len(self._samples)} samples (warning: {self._last_error}).")
            else:
                print(f"[Power] Logging finished with {len(self._samples)} samples.")

    def set_activity(self, activity: str) -> None:
        if not self.enabled:
            return
        timestamp = time.time()
        with self._lock:
            if activity == self._activity:
                return
            self._activity = activity
        self._record_activity_event(activity, timestamp)
        if self.verbose:
            print(f"[Power] Activity -> {activity}")

    def set_process_target(self, pid: Optional[int], name: Optional[str] = None) -> None:
        """Update the process that should be tracked for power attribution."""
        self.process_pid = pid
        if name:
            self.process_name = name

    def update_metadata(self, **fields: Any) -> None:
        """Merge custom fields into the metadata block for the final log."""
        if not fields:
            return
        sanitized: Dict[str, Any] = {}
        for key, value in fields.items():
            if value is None:
                continue
            sanitized[key] = value
        if not sanitized:
            return
        with self._lock:
            self._metadata_context.update(sanitized)

    def set_resolution(
        self,
        width: int,
        height: int,
        *,
        fps: Optional[float] = None,
        source: Optional[str] = None,
    ) -> None:
        """Record the resolved capture resolution (and optional FPS) in metadata."""
        try:
            width_int = int(width)
            height_int = int(height)
        except (TypeError, ValueError):
            return
        if width_int <= 0 or height_int <= 0:
            return
        resolution: Dict[str, Any] = {
            "width": width_int,
            "height": height_int,
            "label": f"{width_int}x{height_int}",
        }
        if fps is not None:
            try:
                fps_val = float(fps)
            except (TypeError, ValueError):
                fps_val = None
            if fps_val is not None and fps_val > 0:
                resolution["fps"] = fps_val
        if source is not None:
            resolution["source"] = str(source)
        self.update_metadata(resolution=resolution)

    # Internal helpers -----------------------------------------------------------------

    def _record_activity_event(self, activity: str, timestamp: float) -> None:
        with self._lock:
            self._activity_events.append({"timestamp": timestamp, "activity": activity})

    def _run_loop(self) -> None:
        if _jtop_context is None:
            self._last_error = "jetson-stats (jtop) Python module not installed"
            return

        try:
            with _jtop_context(interval=max(self.sample_interval, 0.1)) as jetson:
                if not jetson.ok():
                    self._last_error = "unable to communicate with jtop service (jetson not ok)"
                    return
                while not self._stop_event.is_set() and jetson.ok():
                    sample = self._collect_sample(jetson)
                    if sample is not None:
                        with self._lock:
                            self._samples.append(sample)
                        if self.verbose and sample.total_power_w is not None:
                            print(f"[Power] {sample.total_power_w:.2f} W ({sample.activity})")
                    # jtop already waits for the interval internally, but sleep lightly to honour stop events.
                    self._stop_event.wait(self.sample_interval)
        except Exception as exc:  # pragma: no cover - device interaction
            self._last_error = str(exc)

    def _collect_sample(self, jetson) -> Optional[PowerSample]:
        snapshot = self._read_jtop_power(jetson)
        if snapshot is None:
            return None
        timestamp = time.time()
        with self._lock:
            activity = self._activity
            start_timestamp = self._start_timestamp or timestamp
        elapsed = max(0.0, timestamp - start_timestamp)
        snapshot.timestamp = timestamp
        snapshot.activity = activity
        snapshot.elapsed_s = elapsed
        snapshot.soc_power_w = self._estimate_soc_power(snapshot.rails)
        snapshot.process = self._build_process_breakdown(jetson, snapshot.soc_power_w)
        return snapshot

    def _estimate_soc_power(self, rails: List[ChannelReading]) -> Optional[float]:
        if not rails:
            return None
        total = 0.0
        matched = False
        for reading in rails:
            if reading.power_w is None or not reading.rail:
                continue
            label = reading.rail.upper()
            if any(token in label for token in SOC_RAIL_TOKENS):
                total += reading.power_w
                matched = True
        return total if matched else None

    def _build_process_breakdown(
        self,
        jetson,
        soc_power_w: Optional[float],
    ) -> Optional[ProcessPowerBreakdown]:
        pid = self.process_pid
        if pid is None:
            return None
        breakdown = ProcessPowerBreakdown(
            pid=pid,
            name=self.process_name,
            user=None,
            cpu_percent=None,
            ram_mb=None,
            gpu_mem_mb=None,
            cpu_share=None,
            gpu_share=None,
            estimated_power_w=None,
            method="no-process-data",
        )
        try:
            processes = list(jetson.processes or [])
            if self._process_error_reported:
                self._process_error_reported = False
        except Exception as exc:  # pragma: no cover - requires hardware
            if not self._process_error_reported and self.verbose:
                print(f"[Power] Unable to query jtop processes: {exc}")
                self._process_error_reported = True
            breakdown.method = "process-read-error"
            return breakdown

        if not processes:
            return breakdown

        target_entry = None
        for proc in processes:
            if not proc:
                continue
            try:
                proc_pid = int(proc[0])
            except (TypeError, ValueError):
                continue
            if proc_pid == pid:
                target_entry = proc
                break

        if target_entry is None:
            breakdown.method = "process-not-found"
            return breakdown

        breakdown.method = "stats-only"
        breakdown.user = target_entry[1] if len(target_entry) > 1 else None
        cpu_percent = _to_float(target_entry[6]) if len(target_entry) > 6 else None
        ram_kb = _to_float(target_entry[7]) if len(target_entry) > 7 else None
        gpu_kb = _to_float(target_entry[8]) if len(target_entry) > 8 else None
        breakdown.cpu_percent = cpu_percent
        breakdown.ram_mb = (ram_kb / 1024.0) if ram_kb is not None else None
        breakdown.gpu_mem_mb = (gpu_kb / 1024.0) if gpu_kb is not None else None

        cpu_total = 0.0
        gpu_total_kb = 0.0
        for proc in processes:
            if len(proc) > 6:
                value = _to_float(proc[6])
                if value is not None:
                    cpu_total += max(0.0, value)
            if len(proc) > 8:
                value = _to_float(proc[8])
                if value is not None:
                    gpu_total_kb += max(0.0, value)

        cpu_share = None
        if cpu_percent is not None and cpu_total > 0:
            cpu_share = max(0.0, min(1.0, cpu_percent / cpu_total))
        gpu_share = None
        if gpu_kb is not None and gpu_total_kb > 0:
            gpu_share = max(0.0, min(1.0, gpu_kb / gpu_total_kb))

        breakdown.cpu_share = cpu_share
        breakdown.gpu_share = gpu_share

        weighted_components: List[Tuple[float, float]] = []
        if cpu_share is not None and self.process_cpu_weight > 0:
            weighted_components.append((cpu_share, self.process_cpu_weight))
        if gpu_share is not None and self.process_gpu_weight > 0:
            weighted_components.append((gpu_share, self.process_gpu_weight))

        if not weighted_components:
            breakdown.method = "insufficient-metrics"
            return breakdown

        if soc_power_w is None:
            breakdown.method = "no-soc-power"
            return breakdown

        denom = sum(weight for _, weight in weighted_components)
        if denom <= 0:
            breakdown.method = "insufficient-metrics"
            return breakdown

        combined_share = sum(share * weight for share, weight in weighted_components) / denom
        combined_share = max(0.0, min(1.0, combined_share))
        breakdown.estimated_power_w = soc_power_w * combined_share
        breakdown.method = "cpu-gpu-weighted"
        return breakdown

    def _read_jtop_power(self, jetson) -> Optional[PowerSample]:
        stats = jetson.power
        if not stats:
            return None

        rails: List[ChannelReading] = []
        rail_stats = stats.get("rail") or {}
        for name, values in sorted(rail_stats.items()):
            rails.append(
                ChannelReading(
                    rail=name,
                    power_w=_scaled(values.get("power"), 1000.0),
                    avg_power_w=_scaled(values.get("avg"), 1000.0),
                    voltage_v=_scaled(values.get("volt"), 1000.0),
                    current_a=_scaled(values.get("curr"), 1000.0),
                    online=values.get("online"),
                    sensor_type=values.get("type"),
                    status=values.get("status"),
                )
            )

        total = stats.get("tot") or {}
        total_name = total.get("name")
        total_power = _scaled(total.get("power"), 1000.0)
        total_avg = _scaled(total.get("avg"), 1000.0)

        return PowerSample(
            timestamp=time.time(),
            activity="",
            elapsed_s=0.0,
            total_name=total_name,
            total_power_w=total_power,
            total_avg_power_w=total_avg,
            rails=rails,
        )

    def _write_log(self) -> None:
        if not self.enabled or not self.log_path:
            return
        with self._lock:
            activity_events = list(self._activity_events)
            metadata_context = dict(self._metadata_context)
        payload = {
            "metadata": {
                "backend": "jtop",
                "start_timestamp": self._start_timestamp,
                "end_timestamp": self._end_timestamp,
                "sample_interval_s": self.sample_interval,
                "activity_events": activity_events,
                "error": self._last_error,
            },
            "samples": [asdict(sample) for sample in self._samples],
        }
        if metadata_context:
            payload["metadata"].update(metadata_context)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def jetson_power_available() -> bool:
    """Return True when jetson-stats (jtop) is importable."""
    return _jtop_context is not None
