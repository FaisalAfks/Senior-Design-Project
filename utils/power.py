"""Jetson power logging utilities backed by jetson-stats (jtop)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field, replace
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


def _compact_mapping(values: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of values without None entries."""
    return {key: value for key, value in values.items() if value is not None}


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
        self._resolution_label: str = "unresolved"
        self._sample_event = threading.Event()
        self._activity_sample_counts: Dict[str, int] = {}
        self._activity_condition = threading.Condition(self._lock)

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
        self._sample_event.set()
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
        self._sample_event.set()
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
        self._resolution_label = self._sanitize_resolution_label(resolution["label"])

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
                        self._register_sample(sample)
                        if self.verbose and sample.total_power_w is not None:
                            print(f"[Power] {sample.total_power_w:.2f} W ({sample.activity})")
                    if self._await_next_sample():
                        break
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

    def _register_sample(self, sample: PowerSample) -> None:
        with self._lock:
            self._samples.append(sample)
            count = self._activity_sample_counts.get(sample.activity, 0) + 1
            self._activity_sample_counts[sample.activity] = count
            self._activity_condition.notify_all()

    def activity_sample_count(self, activity: str) -> int:
        if not self.enabled:
            return 0
        with self._lock:
            return self._activity_sample_counts.get(activity, 0)

    def wait_for_activity_sample(self, activity: str, baseline: Optional[int], *, timeout: float = 1.5) -> bool:
        if not self.enabled or baseline is None:
            return True
        deadline = time.time() + max(timeout, 0.0)
        with self._activity_condition:
            while self._activity_sample_counts.get(activity, 0) <= baseline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._activity_condition.wait(timeout=remaining)
        return True

    def force_activity_sample(self, activity: str) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if not self._samples:
                return False
            source = self._samples[-1]
            elapsed = max(0.0, time.time() - (self._start_timestamp or time.time()))
            clone = replace(
                source,
                timestamp=time.time(),
                activity=activity,
                elapsed_s=elapsed,
            )
            self._samples.append(clone)
            count = self._activity_sample_counts.get(activity, 0) + 1
            self._activity_sample_counts[activity] = count
            self._activity_condition.notify_all()
            return True

    def _await_next_sample(self) -> bool:
        """Wait for the next sampling window or an activity-triggered request.

        Returns True when the logger should exit (stop requested), False to continue sampling.
        """
        deadline = time.time() + self.sample_interval
        while True:
            if self._stop_event.is_set():
                return True
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            if self._sample_event.wait(timeout=remaining):
                self._sample_event.clear()
                break
        return self._stop_event.is_set()

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
        samples = self._serialize_samples()
        metadata = {
            "backend": "jtop",
            "start_timestamp": self._start_timestamp,
            "end_timestamp": self._end_timestamp,
            "sample_interval_s": self.sample_interval,
            "activity_events": activity_events,
            "error": self._last_error,
        }
        if metadata_context:
            metadata.update(metadata_context)
        metadata = _compact_mapping(metadata)
        written_ts = time.time()
        metadata["log_variant"] = "board"
        metadata["written_at"] = written_ts
        payload = {
            "metadata": metadata,
            "samples": samples,
        }
        pid_metadata = dict(metadata)
        pid_metadata["log_variant"] = "process"
        pid_payload = self._build_pid_payload(pid_metadata, samples)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        combined_payload = self._load_combined_log()
        merged_payload = self._merge_combined_log(combined_payload, payload, pid_payload)
        self.log_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")

    def _build_pid_payload(self, metadata: Dict[str, Any], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        metadata = dict(metadata)
        metadata["log_variant"] = "process"
        metadata = _compact_mapping(metadata)
        process_samples: List[Dict[str, Any]] = []
        for sample in samples:
            process = sample.get("process")
            if not isinstance(process, dict):
                continue
            entry = {
                "timestamp": sample.get("timestamp"),
                "activity": sample.get("activity"),
                "elapsed_s": sample.get("elapsed_s"),
                "total_name": sample.get("total_name"),
                "total_power_w": sample.get("total_power_w"),
                "total_avg_power_w": sample.get("total_avg_power_w"),
                "soc_power_w": sample.get("soc_power_w"),
                "process": process,
            }
            process_samples.append(_compact_mapping(entry))
        return {
            "metadata": metadata,
            "process": {
                "pid": self.process_pid,
                "name": self.process_name,
            },
            "samples": process_samples,
        }

    def _serialize_samples(self) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for sample in self._samples:
            entry: Dict[str, Any] = {
                "timestamp": sample.timestamp,
                "elapsed_s": sample.elapsed_s,
            }
            activity = (sample.activity or "").strip()
            if activity:
                entry["activity"] = activity
            if sample.total_name:
                entry["total_name"] = sample.total_name
            if sample.total_power_w is not None:
                entry["total_power_w"] = sample.total_power_w
            if sample.total_avg_power_w is not None:
                entry["total_avg_power_w"] = sample.total_avg_power_w
            if sample.soc_power_w is not None:
                entry["soc_power_w"] = sample.soc_power_w
            process_entry = self._serialize_process_breakdown(sample.process)
            if process_entry:
                entry["process"] = process_entry
            rails = [self._serialize_rail(reading) for reading in sample.rails]
            rails = [rail for rail in rails if rail]
            if rails:
                entry["rails"] = rails
            serialized.append(entry)
        return serialized

    @staticmethod
    def _serialize_process_breakdown(breakdown: Optional[ProcessPowerBreakdown]) -> Optional[Dict[str, Any]]:
        if breakdown is None:
            return None
        entry = {
            "pid": breakdown.pid,
            "name": breakdown.name,
            "user": breakdown.user,
            "cpu_percent": breakdown.cpu_percent,
            "ram_mb": breakdown.ram_mb,
            "gpu_mem_mb": breakdown.gpu_mem_mb,
            "cpu_share": breakdown.cpu_share,
            "gpu_share": breakdown.gpu_share,
            "estimated_power_w": breakdown.estimated_power_w,
        }
        compact = _compact_mapping(entry)
        if breakdown.method:
            compact["method"] = breakdown.method
        return compact or None

    @staticmethod
    def _serialize_rail(reading: ChannelReading) -> Dict[str, Any]:
        entry: Dict[str, Any] = {}
        if reading.rail is not None:
            entry["rail"] = reading.rail
        extras = _compact_mapping(
            {
                "power_w": reading.power_w,
                "avg_power_w": reading.avg_power_w,
                "voltage_v": reading.voltage_v,
                "current_a": reading.current_a,
                "online": reading.online,
                "sensor_type": reading.sensor_type,
                "status": reading.status,
            }
        )
        entry.update(extras)
        return entry

    @staticmethod
    def _sanitize_resolution_label(label: str) -> str:
        text = (label or "").strip()
        if not text:
            return "unresolved"
        cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-", "x", "X") else "_" for ch in text)
        cleaned = cleaned.strip("_")
        return cleaned.lower() or "unresolved"

    def _empty_combined_log(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "logs": {},
            "plots": {},
        }

    def _load_combined_log(self) -> Dict[str, Any]:
        try:
            raw = json.loads(self.log_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._empty_combined_log()
        except (OSError, json.JSONDecodeError):
            return self._empty_combined_log()
        if not isinstance(raw, dict):
            return self._empty_combined_log()
        if "logs" not in raw or not isinstance(raw["logs"], dict):
            raw["logs"] = {}
        if "plots" not in raw or not isinstance(raw["plots"], dict):
            raw["plots"] = {}
        return self._normalize_combined_log(raw)

    def _normalize_combined_log(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logs = payload.get("logs")
        if not isinstance(logs, dict):
            payload["logs"] = {}
            logs = payload["logs"]
        for key, entry in list(logs.items()):
            if not isinstance(entry, dict):
                continue
            variants = entry.get("variants")
            if not isinstance(variants, dict):
                variants = {}
            board_payload = entry.pop("board", None)
            process_payload = entry.pop("process", None)
            if board_payload and "board" not in variants:
                variants["board"] = board_payload
            if process_payload and process_payload.get("samples"):
                variants["process"] = process_payload
            entry["variants"] = variants
        plots = payload.get("plots")
        if not isinstance(plots, dict):
            payload["plots"] = {}
            plots = payload["plots"]
        for name, config in plots.items():
            if not isinstance(config, dict):
                continue
            if config.get("type") != "power":
                continue
            parsed_key, parsed_variant = self._parse_plot_reference(name)
            if "log_key" not in config and parsed_key:
                config["log_key"] = parsed_key
            if "variant" not in config and parsed_variant:
                config["variant"] = parsed_variant
            if "payload" in config and config.get("log_key"):
                config.pop("payload", None)
        payload["version"] = 3
        if "latest_board" in payload and isinstance(payload["latest_board"], dict):
            latest = payload["latest_board"]
            if "key" in latest and "variant" in latest:
                pass
            else:
                payload["latest_board"] = None
        if "latest_process" in payload and isinstance(payload["latest_process"], dict):
            latest_proc = payload["latest_process"]
            if "key" in latest_proc and "variant" in latest_proc:
                pass
            else:
                payload["latest_process"] = None
        return payload

    @staticmethod
    def _parse_plot_reference(name: str) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(name, str):
            return None, None
        if not name.startswith("power_"):
            return None, None
        remainder = name[len("power_") :]
        if "_" not in remainder:
            return None, None
        prefix, suffix = remainder.rsplit("_", 1)
        variant = suffix if suffix in ("board", "process") else None
        key = prefix or None
        return key, variant

    def _merge_combined_log(
        self,
        combined_payload: Dict[str, Any],
        board_payload: Dict[str, Any],
        process_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata = board_payload.get("metadata") or {}
        resolution_meta = metadata.get("resolution") or {}
        label = resolution_meta.get("label") or self._resolution_label or "unresolved"
        label_text = label.strip() if isinstance(label, str) else "unresolved"
        key = self._sanitize_resolution_label(label_text)
        timestamp = metadata.get("written_at", time.time())
        combined_payload["version"] = 3
        combined_payload["updated_at"] = timestamp
        logs_section = combined_payload.setdefault("logs", {})
        entry: Dict[str, Any] = {
            "label": label_text or key,
            "key": key,
            "updated_at": timestamp,
            "resolution": resolution_meta,
            "variants": {
                "board": board_payload,
            },
        }
        if process_payload.get("samples"):
            entry["variants"]["process"] = process_payload
        logs_section[key] = entry

        plots_section = combined_payload.setdefault("plots", {})
        board_plot_key = self._plot_key(key, "board")
        plots_section[board_plot_key] = {
            "type": "power",
            "title": self._plot_title(label_text or key, variant="board"),
            "log_key": key,
            "variant": "board",
            "use_process_power": False,
        }

        process_plot_key = self._plot_key(key, "process")
        if process_payload.get("samples"):
            plots_section[process_plot_key] = {
                "type": "power",
                "title": self._plot_title(label_text or key, variant="process"),
                "log_key": key,
                "variant": "process",
                "use_process_power": True,
            }
        else:
            plots_section.pop(process_plot_key, None)

        combined_payload["latest_board"] = {
            "key": key,
            "variant": "board",
            "updated_at": timestamp,
        }
        if process_payload.get("samples"):
            combined_payload["latest_process"] = {
                "key": key,
                "variant": "process",
                "updated_at": timestamp,
            }
        else:
            combined_payload["latest_process"] = None
        return combined_payload

    @staticmethod
    def _plot_key(resolution_key: str, variant: str) -> str:
        variant_clean = "process" if variant.lower().startswith("process") else "board"
        return f"power_{resolution_key}_{variant_clean}"

    @staticmethod
    def _plot_title(label: str, variant: str) -> str:
        descriptor = "Process" if variant == "process" else "Board"
        return f"{label} • {descriptor} power"


def jetson_power_available() -> bool:
    """Return True when jetson-stats (jtop) is importable."""
    return _jtop_context is not None
