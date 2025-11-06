"""Jetson power logging utilities backed by jetson-stats (jtop)."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from jtop import jtop as _jtop_context  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _jtop_context = None


def _scaled(value: Any, scale: float) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / scale
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
class PowerSample:
    timestamp: float
    activity: str
    elapsed_s: float
    total_name: Optional[str]
    total_power_w: Optional[float]
    total_avg_power_w: Optional[float]
    rails: List[ChannelReading]


class JetsonPowerLogger:
    """Background sampler that records Jetson power telemetry via jtop."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_path: Path,
        sample_interval: float = 1.0,
        verbose: bool = True,
    ) -> None:
        self._requested = enabled
        self.available = _jtop_context is not None
        self.enabled = self._requested and self.available
        self.log_path = log_path
        self.sample_interval = max(sample_interval, 0.1)
        self.verbose = verbose
        self._last_error: Optional[str] = None

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
        return snapshot

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
