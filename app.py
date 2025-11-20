#!/usr/bin/env python3
"""Modernised attendance demo app with a dashboard-style layout."""
import atexit
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from dataclasses import replace

import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from MobileFaceNet import MobileFaceNetService
from pipelines.attendance import (
    SessionCallbacks,
    DEFAULT_FACEBANK,
)
from dashboard.configuration import (
    DemoConfig,
)
from dashboard.utils import select_log_dialog, METRICS_PANEL_STYLE, _sanitize_identity_name
from dashboard.controllers import AttendanceSessionController, RegistrationSession
from dashboard.services import AttendanceLogBook
from dashboard.services.errors import format_exception, show_error_dialog
from dashboard.settings import AppSettings, SettingsStore
from dashboard.theme import ThemeManager, ThemeConfig
from dashboard.widgets import (
    ControlPanel,
    FacebankPanel,
    FrameDisplay,
    LogbookPanel,
    RegistrationPanel,
    SessionButtons,
    StatusPanel,
)
from utils.device import select_device
from utils.overlay import draw_text_panel
from utils.paths import logs_path
from utils.logging import CSV_FIELDS


SETTINGS_PATH = Path("app_settings.json")


EXPORTS_DIR = logs_path("exports")
LOG_BOOK_PATH = logs_path("logbook.json")

class DashboardApp:
    def __init__(self) -> None:
        self.settings_store = SettingsStore(SETTINGS_PATH)
        self.app_settings = self.settings_store.load()

        large_text_theme = ThemeConfig()
        self.theme = ThemeManager(config=large_text_theme)
        self.root = tk.Tk()
        self.root.title("Attendance Dashboard")
        self.root.geometry("1400x800")
        self._geometry_job: Optional[str] = None
        self._apply_initial_geometry()
        self.root.bind("<Configure>", self._on_root_configure)
        self.theme.apply(self.root)
        self.controller: Optional[AttendanceSessionController] = None
        self.log_book = AttendanceLogBook(LOG_BOOK_PATH)
        self._current_log_id: Optional[str] = None
        self._active_log_id: Optional[str] = None
        self._next_session_log_id: Optional[str] = None
        self._cleanup_done = False
        self._facebank_refresh_lock = threading.Lock()
        self._facebank_refresh_active = False
        self._pending_facebank_refresh: Optional[tuple[DemoConfig, str, str]] = None
        self._facebank_locked = False
        atexit.register(self._cleanup_resources)

        status_pad = self.theme.pad("outer")
        self.status_panel = StatusPanel(self.root, theme=self.theme)
        self.status_panel.pack(fill="x", padx=status_pad, pady=(status_pad, status_pad // 2))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.notebook = notebook

        self.session_tab = ttk.Frame(notebook)
        self.session_tab.columnconfigure(0, weight=1)
        self.session_tab.rowconfigure(0, weight=1)
        notebook.add(self.session_tab, text="Live Session")

        pad = self.theme.pad("outer")
        video_frame = ttk.LabelFrame(self.session_tab, text="Live Feed")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=pad, pady=(pad, pad))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        live_w, live_h = self.theme.live_feed_size
        feed_container = ttk.Frame(video_frame, width=live_w, height=live_h)
        feed_container.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        feed_container.grid_propagate(False)
        feed_container.rowconfigure(0, weight=1)
        feed_container.columnconfigure(0, weight=1)
        video_label = ttk.Label(feed_container, anchor="center")
        video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_display = FrameDisplay(video_label, target_size=(live_w, live_h))
        self.video_display.start()

        session_footer = ttk.Frame(self.session_tab)
        session_footer.grid(row=1, column=0, sticky="ew", pady=(0, pad), padx=pad)
        session_footer.columnconfigure(0, weight=1)
        self.session_buttons = SessionButtons(session_footer, theme=self.theme)
        self.session_buttons.grid(row=0, column=0)
        self.session_buttons.configure_start(text="Start Session", command=self._start_session_from_controls)
        self.session_buttons.configure_stop(self._stop_session)
        self.session_buttons.configure_resume(self._resume_session_from_log)

        self.registration_panel = RegistrationPanel(
            session_footer,
            on_capture=self._capture_registration_sample,
            on_save=self._save_registration_samples,
            on_cancel=self._cancel_registration,
            theme=self.theme,
        )
        self.registration_panel.grid(row=0, column=0, sticky="ew")
        self.registration_panel.grid_remove()

        facebank_tab = ttk.Frame(notebook)
        notebook.add(facebank_tab, text="Facebank")
        self.facebank_panel = FacebankPanel(
            facebank_tab,
            on_register=self._start_registration,
            on_refresh=self._on_facebank_refresh,
            show_error_dialog=show_error_dialog,
            theme=self.theme,
        )
        facebank_pad = self.theme.pad("outer")
        self.facebank_panel.pack(fill="both", expand=True, padx=facebank_pad, pady=facebank_pad)

        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="Attendance Logs")

        self.log_panel = LogbookPanel(
            log_tab,
            on_prev=self._select_prev_log,
            on_next=self._select_next_log,
            on_new=self._create_log,
            on_rename=self._rename_log,
            on_delete=self._delete_log,
            on_export=self._export_log,
            theme=self.theme,
        )
        self.log_panel.pack(fill="both", expand=True)

        settings_tab = ttk.Frame(notebook)
        settings_tab.columnconfigure(0, weight=1)
        notebook.add(settings_tab, text="Settings")
        self.settings_tab = settings_tab
        self.control_panel = ControlPanel(
            settings_tab,
            settings_store=self.settings_store,
            initial_settings=self.app_settings,
            on_overlay_change=self._on_overlay_change,
            on_settings_saved=self._handle_settings_saved,
            show_error_dialog=show_error_dialog,
        )
        self.control_panel.pack(fill="both", expand=True, padx=12, pady=12)

        self._latest_metrics: Dict[str, float] = {}
        self._show_metrics_overlay = self.control_panel.show_metrics_var.get()
        self._display_scores = self.control_panel.display_scores_var.get()
        self._session_active = False
        self._session_identities: set[str] = set()
        self._spoof_attempts: dict[str, int] = {}
        self._blocked_identities: set[str] = set()
        self._max_spoof_attempts = 3
        self._next_person_event = threading.Event()
        self._awaiting_next = False
        self.registration_session: Optional[RegistrationSession] = None
        self._session_button_mode = "start"
        self._set_start_button_mode("start")
        self._initialize_logbook_state()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Ensure embeddings reflect the latest facebank on startup.
        self.root.after(
            100,
            lambda: self._on_facebank_refresh(
                status_message="Preparing facebank...",
                success_message="Facebank ready.",
            ),
        )

    def _handle_settings_saved(self, settings: AppSettings) -> None:
        self.app_settings = settings

    def _apply_initial_geometry(self) -> None:
        geometry = self.app_settings.window_geometry
        if geometry:
            try:
                self.root.geometry(geometry)
            except tk.TclError:
                pass
        state = (self.app_settings.window_state or "normal").lower()
        try:
            self.root.state(state)
        except tk.TclError:
            pass

    def _on_root_configure(self, event) -> None:
        if event.widget is not self.root:
            return
        state = self.root.state()
        if state not in ("normal", "zoomed"):
            return
        self._schedule_geometry_snapshot()

    def _schedule_geometry_snapshot(self) -> None:
        if self._geometry_job is not None:
            try:
                self.root.after_cancel(self._geometry_job)
            except tk.TclError:
                pass
        self._geometry_job = self.root.after(750, self._persist_geometry_settings)

    def _persist_geometry_settings(self) -> None:
        self._geometry_job = None
        try:
            geometry = self.root.geometry()
            state = self.root.state()
        except tk.TclError:
            return
        if geometry == self.app_settings.window_geometry and state == self.app_settings.window_state:
            return
        updated = replace(self.app_settings, window_geometry=geometry, window_state=state)
        self.settings_store.save(updated)
        self.app_settings = updated
        control_panel = getattr(self, "control_panel", None)
        if control_panel is not None:
            control_panel.update_window_settings(geometry=geometry, state=state)

    def run(self) -> None:
        self.root.mainloop()

    # ------------------------------------------------------------------ logbook helpers
    def _initialize_logbook_state(self) -> None:
        selected = self.log_book.selected_page_id
        if selected and self.log_book.get_page(selected):
            self._current_log_id = selected
        elif not self.log_book.is_empty:
            first = self.log_book.list_pages()[0]
            self._current_log_id = first["id"]
            self.log_book.set_selected_page(self._current_log_id)
        else:
            self._current_log_id = None
        self._active_log_id = None
        self._refresh_log_display()

    def _refresh_log_display(self) -> None:
        if self._current_log_id is None:
            self.log_panel.clear_entries()
        else:
            entries = self.log_book.combined_entries(self._current_log_id)
            self.log_panel.load_entries(entries)
        self.log_panel.set_title(self._format_log_label())
        self._update_log_nav_state()
        keep_ids = {pid for pid in (self._current_log_id, self._active_log_id) if pid}
        self.log_book.release_entries(keep=keep_ids or None)

    def _format_log_label(self) -> str:
        pages = self.log_book.list_pages()
        if not pages or not self._current_log_id:
            return "No logs yet"
        index = self.log_book.page_index(self._current_log_id)
        if index < 0:
            return "No logs yet"
        page = pages[index]
        badges: list[str] = []
        if self._session_active and self._active_log_id == page["id"]:
            badges.append("live")
        suffix = f" ({', '.join(badges)})" if badges else ""
        return f"Log {index + 1}/{len(pages)}: {page['name']}{suffix}"

    def _update_log_nav_state(self) -> None:
        pages = self.log_book.list_pages()
        index = self.log_book.page_index(self._current_log_id)
        self.log_panel.set_prev_enabled(not (len(pages) <= 1 or index <= 0))
        self.log_panel.set_next_enabled(not (len(pages) <= 1 or index in (-1, len(pages) - 1)))

    def _set_current_log(self, page_id: Optional[str]) -> None:
        if not page_id:
            return
        if self.log_book.get_page(page_id) is None:
            return
        self._current_log_id = page_id
        self.log_book.set_selected_page(page_id)
        self._refresh_log_display()

    def _select_prev_log(self) -> None:
        pages = self.log_book.list_pages()
        index = self.log_book.page_index(self._current_log_id)
        if index > 0:
            self._set_current_log(pages[index - 1]["id"])

    def _select_next_log(self) -> None:
        pages = self.log_book.list_pages()
        index = self.log_book.page_index(self._current_log_id)
        if 0 <= index < len(pages) - 1:
            self._set_current_log(pages[index + 1]["id"])

    def _generate_default_log_name(self) -> str:
        return datetime.now().strftime("Session %Y-%m-%d %H:%M")


    def _require_active_log_id(self) -> str:
        target_page_id = self._active_log_id or self._current_log_id
        if target_page_id is None:
            fallback_page = self.log_book.create_page(self._generate_default_log_name())
            target_page_id = fallback_page["id"]
            self._active_log_id = target_page_id
            self._set_current_log(target_page_id)
        return target_page_id

    def _create_log(self) -> None:
        suggested = self._generate_default_log_name()
        name = simpledialog.askstring(
            "New attendance log",
            "Name this log:",
            parent=self.root,
            initialvalue=suggested,
        )
        if name is None:
            return
        final_name = name.strip() or suggested
        page = self.log_book.create_page(final_name)
        self._set_current_log(page["id"])
        self.status_panel.set_status(f"Created log '{final_name}'.")

    def _rename_log(self) -> None:
        page = self.log_book.get_page(self._current_log_id)
        if page is None:
            messagebox.showinfo("No log", "Create a log before renaming.", parent=self.root)
            return
        new_name = simpledialog.askstring(
            "Rename log",
            "Enter a new name for this log:",
            parent=self.root,
            initialvalue=page["name"],
        )
        if new_name is None:
            return
        final_name = new_name.strip()
        if not final_name:
            return
        self.log_book.rename_page(page["id"], final_name)
        self._refresh_log_display()
        self.status_panel.set_status(f"Renamed log to '{final_name}'.")

    def _resume_session_from_log(self) -> None:
        if self._session_active:
            messagebox.showinfo("Resume session", "Stop the current session before resuming another log.", parent=self.root)
            return
        pages = self.log_book.list_pages()
        if not pages:
            messagebox.showinfo("Resume session", "No logs available to resume.", parent=self.root)
            return
        current_index = self.log_book.page_index(self._current_log_id)
        selection = select_log_dialog(self.root, self.theme, pages, current_index)
        if selection is None:
            return
        page = self.log_book.get_page(selection)
        if page is None:
            show_error_dialog("Resume session", "Unable to load the selected log.", parent=self.root)
            return
        if self._session_active and self._active_log_id == page["id"]:
            messagebox.showinfo("Resume session", f"'{page['name']}' is already live.", parent=self.root)
            return
        self._next_session_log_id = page["id"]
        self._set_current_log(page["id"])
        self.status_panel.set_status(f"Next session will resume '{page['name']}'.")
        self._refresh_log_display()

    def _prepare_log_for_new_session(self) -> bool:
        if self._next_session_log_id:
            page = self.log_book.get_page(self._next_session_log_id)
            if page is not None:
                self._active_log_id = page["id"]
                self._set_current_log(page["id"])
                self.status_panel.set_status(f"Resuming log '{page['name']}'.")
                self._next_session_log_id = None
                return True
            self._next_session_log_id = None
        suggested = self._generate_default_log_name()
        prompt = (
            "Name this session's attendance log.\n"
            "A new log will be created so you can revisit it later."
        )
        name = simpledialog.askstring(
            "New session log",
            prompt,
            parent=self.root,
            initialvalue=suggested,
        )
        if name is None:
            return False
        final_name = name.strip() or suggested
        page = self.log_book.create_page(final_name)
        self._active_log_id = page["id"]
        self._set_current_log(page["id"])
        self.status_panel.set_status(f"Logging session '{final_name}'.")
        return True

    def _seed_session_identities(self) -> None:
        """Populate the duplicate tracker using existing accepted entries in the active log."""
        self._session_identities.clear()
        target_page_id = self._active_log_id or self._current_log_id
        if not target_page_id:
            return
        entries = self.log_book.entries_for(target_page_id)
        for entry in entries:
            identity = entry.get("identity")
            if entry.get("accepted") and identity and identity != "Unknown":
                self._session_identities.add(str(identity))

    def _reset_spoof_tracking(self) -> None:
        self._spoof_attempts.clear()
        self._blocked_identities.clear()

    def _is_identity_blocked(self, identity: str) -> bool:
        if not identity:
            return False
        return identity in self._blocked_identities

    def _delete_log(self) -> None:
        page = self.log_book.get_page(self._current_log_id)
        if page is None:
            messagebox.showinfo("Delete log", "Select a log to delete.", parent=self.root)
            return
        if self._session_active and self._active_log_id == page["id"]:
            messagebox.showinfo("Delete log", "Stop the active session before deleting its log.", parent=self.root)
            return
        confirm = messagebox.askyesno(
            "Delete log",
            f"Delete '{page['name']}' and all attendance entries?",
            parent=self.root,
        )
        if not confirm:
            return
        index = self.log_book.page_index(page["id"])
        self.log_book.delete_page(page["id"])
        if self._next_session_log_id == page["id"]:
            self._next_session_log_id = None
        if self._active_log_id == page["id"]:
            self._active_log_id = None
        pages = self.log_book.list_pages()
        if pages:
            new_index = min(max(index, 0), len(pages) - 1)
            self._set_current_log(pages[new_index]["id"])
        else:
            self._current_log_id = None
            self._refresh_log_display()
        self.status_panel.set_status(f"Deleted log '{page['name']}'.")

    def _start_session_from_controls(self) -> None:
        if self._facebank_locked:
            messagebox.showinfo("Facebank busy", "Facebank is refreshing; wait until it completes before starting a session.", parent=self.root)
            return
        if self.registration_session is not None:
            messagebox.showinfo("Registration active", "Finish or cancel registration before starting a session.", parent=self.root)
            return
        if self._session_active:
            return
        config = self.control_panel.build_demo_config()
        if config is None:
            self.notebook.select(self.settings_tab)
            return
        try:
            select_device(config.device)
        except Exception as exc:
            show_error_dialog("Device error", str(exc), parent=self.root)
            return
        if not self._prepare_log_for_new_session():
            return
        self._start_session(config)

    def _start_session(self, config: DemoConfig) -> None:
        self._stop_session(preserve_active_log=True, dispose=False)
        controller = self.controller
        if controller is None:
            controller = AttendanceSessionController(config)
            self.controller = controller
        elif controller.config != config:
            controller.shutdown()
            controller = AttendanceSessionController(config)
            self.controller = controller
        else:
            controller.config = config
        controller.set_blocked_identity_checker(self._is_identity_blocked)
        self._show_metrics_overlay = config.show_metrics
        self._display_scores = config.display_scores
        self._seed_session_identities()
        self._reset_spoof_tracking()
        self._next_person_event = threading.Event()
        self._awaiting_next = False
        self._session_active = True
        self._update_facebank_panel_busy_state()
        self.log_panel.set_title(self._format_log_label())
        self._set_start_button_mode("disabled")
        self.session_buttons.set_pause_enabled(True)
        self._show_session_controls()
        callbacks = SessionCallbacks(
            on_guidance_frame=self._submit_frame,
            on_verification_frame=self._submit_frame,
            on_final_frame=self._submit_frame,
            poll_cancel=lambda: False,
            wait_for_next_person=self._wait_for_next,
            on_summary=self._handle_summary,
            on_status=lambda text: self._ui_call(self.status_panel.set_status, text),
            on_stage_change=lambda text: self._ui_call(self.status_panel.set_stage, text),
            on_metrics=self._handle_metrics,
        )
        try:
            controller.start(callbacks, on_error=self._handle_session_error)
            self.status_panel.set_status("Align face to begin verification.")
        except Exception as exc:
            self._stop_session(dispose=True)
            show_error_dialog("Session error", format_exception(exc), parent=self.root)

    def _shutdown_controller(self) -> None:
        controller = self.controller
        if controller is None:
            return
        self.controller = None
        try:
            controller.shutdown()
        except Exception as exc:
            print(f"[Dashboard] Warning: Failed to stop session controller: {exc}")

    def _stop_session(self, *, preserve_active_log: bool = False, dispose: bool = False) -> None:
        retained_log = self._active_log_id if preserve_active_log else None
        controller = self.controller
        if controller is not None:
            try:
                if dispose:
                    controller.shutdown()
                    self.controller = None
                else:
                    controller.stop()
            except Exception as exc:
                print(f"[Dashboard] Warning: Failed to stop session controller: {exc}")
                if dispose:
                    self.controller = None
        self._session_active = False
        self._active_log_id = retained_log
        self._awaiting_next = False
        self._next_person_event.set()
        self.video_display.clear()
        self.status_panel.set_stage("Idle")
        self.status_panel.set_status("Session stopped.")
        self.session_buttons.set_pause_enabled(False)
        self._set_start_button_mode("start")
        self._clear_metrics_overlay()
        self._session_identities.clear()
        self._reset_spoof_tracking()
        self._update_facebank_panel_busy_state()
        keep_ids = {pid for pid in (self._current_log_id, self._active_log_id) if pid}
        self.log_book.release_entries(keep=keep_ids or None)
        self.log_panel.set_title(self._format_log_label())

    def _handle_session_error(self, exc: BaseException) -> None:
        self.root.after(0, lambda: self._report_session_error(exc))

    def _report_session_error(self, exc: BaseException) -> None:
        self._stop_session(dispose=True)
        self.status_panel.set_status("Session ended due to an error.")
        show_error_dialog("Session error", format_exception(exc), parent=self.root)

    def _wait_for_next(self) -> bool:
        self._awaiting_next = True
        while self.controller is not None and not self.controller._stop.is_set() and self._session_active:
            if self._next_person_event.wait(0.2):
                self._next_person_event.clear()
                self._awaiting_next = False
                return True
        return False

    def _handle_summary(self, summary: Dict[str, object]) -> None:
        self.root.after(0, lambda: self._process_summary(summary))

    def _process_summary(self, summary: Dict[str, object]) -> None:
        timestamp = summary.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        identity = summary.get("identity") or "Unknown"
        accepted = bool(summary.get("accepted"))
        blocked = bool(summary.get("blocked"))
        is_real = summary.get("is_real")
        recognized = bool(summary.get("recognized"))
        source = summary.get("source") or self.control_panel.source_var.get()
        attempts = self._spoof_attempts.get(identity, 0)
        if accepted:
            self._spoof_attempts.pop(identity, None)
            self._blocked_identities.discard(identity)
            attempts = 0
        elif (
            identity != "Unknown"
            and recognized
            and is_real is False
            and not blocked
        ):
            attempts = self._spoof_attempts.get(identity, 0) + 1
            self._spoof_attempts[identity] = attempts
            if attempts >= self._max_spoof_attempts:
                self._blocked_identities.add(identity)
                blocked = True
        elif blocked:
            attempts = max(attempts, self._max_spoof_attempts)
        identity_has_accept = identity != "Unknown" and identity in self._session_identities
        duplicate = accepted and identity_has_accept
        if accepted and not identity_has_accept and identity != "Unknown":
            self._session_identities.add(identity)
        if not duplicate:
            entry = dict(summary)
            entry.update(
                {
                    "timestamp": timestamp,
                    "identity": identity,
                    "accepted": accepted,
                    "source": str(source),
                    "blocked": blocked,
                }
            )
            target_page_id = self._require_active_log_id()
            if accepted:
                self.log_book.add_entry(target_page_id, entry)
                removed_failure = self.log_book.remove_rejection(target_page_id, identity, str(source))
                if self._current_log_id == target_page_id:
                    if removed_failure:
                        self._refresh_log_display()
                    else:
                        self.log_panel.add_entry(
                            timestamp=str(entry.get("timestamp", "")),
                            identity=str(entry.get("identity", "")),
                            accepted=True,
                            source=str(entry.get("source", "")),
                        )
            else:
                if identity_has_accept:
                    removed_failure = self.log_book.remove_rejection(target_page_id, identity, str(source))
                    if removed_failure and self._current_log_id == target_page_id:
                        self._refresh_log_display()
                else:
                    self.log_book.upsert_rejection(target_page_id, entry)
                    if self._current_log_id == target_page_id:
                        self._refresh_log_display()
        result_text: str
        if duplicate:
            result_text = f"{identity}: Already attended"
        elif not accepted and identity_has_accept:
            result_text = f"{identity}: Already attended"
        elif blocked and identity != "Unknown":
            result_text = f"{identity}: Blocked after {self._max_spoof_attempts} spoof attempts"
        elif accepted:
            result_text = f"{identity}: Attended"
        elif identity != "Unknown" and recognized and is_real is False:
            progress = f" ({attempts}/{self._max_spoof_attempts})" if attempts else ""
            result_text = f"{identity}: Spoof detected{progress}"
        else:
            result_text = f"{identity}: Try again"
        self.status_panel.set_status(result_text)
        self._awaiting_next = True
        self._set_start_button_mode("next")

    def _handle_metrics(self, metrics: Dict[str, float]) -> None:
        self._latest_metrics = metrics

    def _clear_metrics_overlay(self) -> None:
        self._latest_metrics = {}

    def _submit_frame(self, frame: np.ndarray) -> None:
        if self.registration_session is not None:
            self.video_display.submit(frame)
            return
        annotated = frame
        if self._show_metrics_overlay:
            annotated = self._apply_metrics_overlay(annotated)
        self.video_display.submit(annotated)

    def _signal_next_person(self) -> None:
        if not self._awaiting_next:
            return
        self._set_start_button_mode("disabled")
        self._awaiting_next = False
        self._clear_metrics_overlay()
        self._next_person_event.set()

    def _start_registration(self) -> None:
        if self.controller is not None:
            self._stop_session()
        if self.registration_session is not None:
            return
        config = self.control_panel.build_demo_config()
        if config is None:
            self.notebook.select(self.settings_tab)
            return
        try:
            session = RegistrationSession(config, self._submit_frame)
            session.start()
        except Exception as exc:
            show_error_dialog("Registration", f"Unable to start registration: {exc}", parent=self.root)
            return
        self.registration_session = session
        self.registration_panel.set_name("")
        self.registration_panel.set_status("Capture samples using the controls buttons.")
        self._show_registration_controls()
        self.notebook.select(self.session_tab)

    def _capture_registration_sample(self) -> None:
        session = self.registration_session
        if session is None:
            return
        try:
            count = session.capture_sample()
        except RuntimeError as exc:
            show_error_dialog("Capture sample", str(exc), parent=self.root)
            return
        self.registration_panel.set_status(f"Captured {count} sample(s).")

    def _save_registration_samples(self) -> None:
        session = self.registration_session
        if session is None:
            return
        identity_raw = self.registration_panel.get_name()
        if not identity_raw:
            messagebox.showinfo("Registration", "Enter an identity name before saving.", parent=self.root)
            return
        sanitized = _sanitize_identity_name(identity_raw)
        if not sanitized:
            show_error_dialog("Registration", "Identity name is invalid.", parent=self.root)
            return
        if session.sample_count == 0:
            messagebox.showinfo("Registration", "Capture at least one sample before saving.", parent=self.root)
            return
        face_dir = DEFAULT_FACEBANK / sanitized
        if face_dir.exists():
            proceed = messagebox.askyesno(
                "Append samples",
                f"The identity '{sanitized}' exists. Append new samples?",
                parent=self.root,
            )
            if not proceed:
                return
        try:
            saved = session.save_samples(sanitized)
        except RuntimeError as exc:
            show_error_dialog("Registration", str(exc), parent=self.root)
            return
        self.registration_panel.set_status(f"Saved {saved} samples for {sanitized}.")
        messagebox.showinfo("Registration complete", f"Saved {saved} samples for {sanitized}.", parent=self.root)
        self.status_panel.set_status("Updating facebank with new registration...")
        self._rebuild_facebank_async(
            session.config,
            success_message=f"Facebank refreshed with new samples for {sanitized}.",
        )
        self._end_registration()
        self._refresh_facebank()

    def _cancel_registration(self) -> None:
        if self.registration_session is None:
            return
        self._end_registration(message="Registration cancelled.")

    def _end_registration(self, message: Optional[str] = None) -> None:
        self._shutdown_registration_session()
        if message:
            self.registration_panel.set_status(message)
        else:
            self.registration_panel.set_status("Registration idle.")
        self.registration_panel.set_name("")
        self.video_display.clear()
        self._show_session_controls()

    def _apply_metrics_overlay(self, frame: np.ndarray) -> np.ndarray:
        metrics = self._latest_metrics
        if not metrics:
            return frame
        lines: list[tuple[str, tuple[int, int, int]]] = []
        fps = metrics.get("avg_fps") or metrics.get("fps")
        if fps is not None:
            lines.append((f"FPS: {fps:5.1f}", (255, 255, 0)))
        detector = metrics.get("avg_detector_ms") or metrics.get("detector_ms")
        if detector is not None:
            lines.append((f"Detector: {detector:.1f} ms", (0, 255, 255)))
        recog = metrics.get("avg_recognition_ms") or metrics.get("recognition_ms")
        if recog is not None:
            lines.append((f"Recogniser: {recog:.1f} ms", (0, 255, 255)))
        spoof = metrics.get("avg_spoof_ms") or metrics.get("spoof_ms")
        if spoof is not None:
            lines.append((f"Spoof: {spoof:.1f} ms", (0, 255, 255)))
        if not lines:
            return frame
        return draw_text_panel(
            frame,
            lines,
            anchor="top-right",
            title="Performance",
            style=METRICS_PANEL_STYLE,
        )

    def _ui_call(self, func, *args, **kwargs) -> None:
        if func is None or self.root is None:
            return
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return
        try:
            self.root.after(0, lambda f=func, a=args, kw=kwargs: f(*a, **kw))
        except tk.TclError:
            pass

    def _set_start_button_mode(self, mode: str) -> None:
        self._session_button_mode = mode
        locked = self._facebank_locked and not self._session_active
        if mode == "start":
            self.session_buttons.configure_start(text="Start Session", command=self._start_session_from_controls)
            self.session_buttons.set_start_enabled(not locked)
        elif mode == "next":
            self.session_buttons.configure_start(text="Next Person", command=self._signal_next_person)
            self.session_buttons.set_start_enabled(True)
        elif mode == "disabled":
            self.session_buttons.set_start_enabled(False)
        self._update_resume_button_state()

    def _update_resume_button_state(self) -> None:
        self.session_buttons.set_resume_enabled(not (self._facebank_locked or self._session_active))

    def _show_session_controls(self) -> None:
        self.registration_panel.grid_remove()
        self.session_buttons.grid()
        if not self._session_active:
            self._set_start_button_mode("start")
            self.session_buttons.set_pause_enabled(False)

    def _show_registration_controls(self) -> None:
        self.session_buttons.grid_remove()
        self.registration_panel.grid()
        self.session_buttons.set_pause_enabled(False)
        self._set_start_button_mode("start")

    def _on_overlay_change(self) -> None:
        new_metrics = self.control_panel.show_metrics_var.get()
        self._show_metrics_overlay = new_metrics
        new_scores = self.control_panel.display_scores_var.get()
        self._display_scores = new_scores
        if self.controller is not None:
            self.controller.set_display_scores(new_scores)

    def _on_register(self) -> None:
        self._start_registration()

    def _export_log(self) -> None:
        page = self.log_book.get_page(self._current_log_id)
        if page is None:
            messagebox.showinfo("Export attendance", "Select a log to export.", parent=self.root)
            return
        page_id = page["id"]
        detail_fields = [field for field in CSV_FIELDS if field not in ("timestamp", "identity", "source")]
        export_dir = EXPORTS_DIR
        export_dir.mkdir(parents=True, exist_ok=True)
        page_slug = _sanitize_identity_name(page["name"])
        date_stamp = datetime.now().strftime("%Y%m%d")
        export_path = export_dir / f"{page_slug}_{date_stamp}.csv"
        try:
            success = self.log_book.export_page_to_csv(
                page_id,
                export_path,
                detail_fields=detail_fields,
            )
        except Exception as exc:
            show_error_dialog("Export failed", f"Unable to export log: {exc}", parent=self.root)
            return
        if not success:
            messagebox.showinfo("Export attendance", "No entries to export for this log.", parent=self.root)
            return
        messagebox.showinfo("Export attendance", f"CSV for '{page['name']}' written to {export_path}", parent=self.root)

    def _refresh_facebank(self) -> None:
        if getattr(self, "facebank_panel", None) is not None:
            self.facebank_panel.refresh()

    def _on_facebank_refresh(
        self,
        *,
        status_message: Optional[str] = None,
        success_message: Optional[str] = None,
    ) -> None:
        config = self.control_panel.build_demo_config()
        if config is None:
            self.notebook.select(self.settings_tab)
            return
        final_status = status_message or "Refreshing facebank..."
        final_success = success_message or "Facebank refreshed."
        self._schedule_facebank_refresh(config, status_message=final_status, success_message=final_success)

    def _schedule_facebank_refresh(self, config: DemoConfig, *, status_message: str, success_message: str) -> None:
        with self._facebank_refresh_lock:
            if self._facebank_refresh_active:
                self._pending_facebank_refresh = (config, status_message, success_message)
                self.status_panel.set_status("Facebank refresh running; queuing another update.")
                return
            self._facebank_refresh_active = True
        self._set_facebank_busy(True)
        self.status_panel.set_status(status_message)
        self._rebuild_facebank_async(config, success_message=success_message)

    def _rebuild_facebank_async(self, config: DemoConfig, *, success_message: str = "Facebank refreshed.") -> None:
        def worker() -> None:
            try:
                MobileFaceNetService(
                    weights_path=config.weights_path,
                    facebank_dir=config.facebank_dir,
                    recognition_threshold=config.identity_threshold,
                    refresh_facebank=True,
                )
                controller = self.controller
                reload_warning: Optional[str] = None
                if controller is not None:
                    try:
                        controller.refresh_facebank()
                    except Exception as refresh_exc:  # pragma: no cover - UI thread handles message
                        reload_warning = format_exception(refresh_exc)

                def notify() -> None:
                    if reload_warning:
                        messagebox.showwarning(
                            "Facebank refresh",
                            "Facebank files were rebuilt, but the live session could not reload them automatically.\n"
                            "Restart the session to apply the latest embeddings.\n\n"
                            f"Details:\n{reload_warning}",
                            parent=self.root,
                        )
                        self._facebank_refresh_complete("Facebank refreshed (restart session to apply).")
                    else:
                        self._facebank_refresh_complete(success_message)

                self._ui_call(notify)
            except Exception as exc:
                err = format_exception(exc)
                self._ui_call(
                    lambda: show_error_dialog(
                        "Facebank refresh failed",
                        err,
                        parent=self.root,
                    )
                )
            finally:
                self._handle_facebank_refresh_finished()

        threading.Thread(target=worker, daemon=True).start()

    def _handle_facebank_refresh_finished(self) -> None:
        pending: Optional[tuple[DemoConfig, str, str]] = None
        with self._facebank_refresh_lock:
            self._facebank_refresh_active = False
            pending = self._pending_facebank_refresh
            self._pending_facebank_refresh = None
        if pending:
            config, status_message, success_message = pending
            self._ui_call(
                lambda: self._schedule_facebank_refresh(
                    config,
                    status_message=status_message,
                    success_message=success_message,
                )
            )
            return
        self._ui_call(lambda: self._set_facebank_busy(False))

    def _set_facebank_busy(self, busy: bool) -> None:
        self._facebank_locked = busy
        self._update_facebank_panel_busy_state()
        if not self._session_active:
            self._set_start_button_mode(self._session_button_mode)
        self._update_resume_button_state()

    def _update_facebank_panel_busy_state(self) -> None:
        panel = getattr(self, "facebank_panel", None)
        if panel is None:
            return
        if self._facebank_refresh_active:
            panel.set_busy(True, message="Facebank is refreshing, please wait for it to finish.")
        elif self._session_active:
            panel.set_busy(True, message="Facebank editing is disabled while a session is running.")
        else:
            panel.set_busy(False)

    def _facebank_refresh_complete(self, message: str) -> None:
        self.status_panel.set_status(message)
        self._refresh_facebank()

    def _on_close(self) -> None:
        self._stop_session(dispose=True)
        if self.registration_session is not None:
            self._end_registration()
        self._cleanup_resources()
        self.root.destroy()

    def _shutdown_registration_session(self) -> None:
        session = self.registration_session
        if session is None:
            return
        self.registration_session = None
        try:
            session.stop()
        except Exception as exc:
            print(f"[Dashboard] Warning: Failed to stop registration session: {exc}")

    def _cleanup_resources(self) -> None:
        if self._cleanup_done:
            return
        self._cleanup_done = True
        if self._geometry_job is not None:
            try:
                self.root.after_cancel(self._geometry_job)
            except tk.TclError:
                pass
            self._geometry_job = None
        self._shutdown_controller()
        self._shutdown_registration_session()
        display = getattr(self, "video_display", None)
        if display is not None:
            try:
                display.stop()
            except Exception:
                pass


def main() -> None:
    DashboardApp().run()


if __name__ == "__main__":
    main()
