# Attendance Dashboard Refactor Plan

This roadmap is now split into two tracks so we can tackle pure refactoring work first and layer new functionality afterward without mixing concerns.

## Goals & Tracks
- **Refactor track**: shrink `app.py` into a composable package, isolate widgets/controllers/services, and make the dashboard easier to test or extend. These steps are prerequisite for feature work.
- **Feature track**: once the structure is in place, add the centralized theme system plus persisted window geometry/settings without fighting the monolith.

> Begin with the refactor phases; the feature phases assume the new module layout already exists.

---

## Refactor Track (Start Here)

### Phase R1 – Baseline mapping & package skeleton
1. Create a `dashboard/` package (or `ui/` if we prefer) with `__init__.py` plus subpackages for `widgets`, `controllers`, and `services`.
2. Move purely structural helpers out of `app.py` without changing behavior:
   - `_resolve_source`, `_format_display_timestamp`, `_next_facebank_index`, `METRICS_PANEL_STYLE`, etc. go into `dashboard/utils.py` (or split by concern).
   - `FrameDisplay` becomes `dashboard/widgets/frame_display.py`.
   - `DemoConfig` moves to `dashboard/configuration.py`, still producing the same `SimpleNamespace` for `AttendancePipeline`.
3. Update import sites inside `app.py` to consume the new modules but keep UI/layout logic untouched. This shrinks the file while ensuring subsequent phases can focus on one concern at a time.

**Deliverables**
- `dashboard/__init__.py`, `dashboard/configuration.py`, `dashboard/widgets/frame_display.py`, `dashboard/utils.py`.
- Updated `app.py` that imports from these modules (no behavioral change yet).

### Phase R2 – Widget/module extraction
1. Split each major UI region into its own module under `dashboard/widgets/`:
   - `status_panel.py` (currently `StatusPanel`).
   - `control_panel.py` (settings form + `DemoConfig` building, still using today’s look and feel).
   - `facebank_panel.py` (facebank list, thumbnails, register/delete buttons).
   - `logbook_panel.py` (log view + navigation).
   - `session_toolbar.py` (Start/Stop/Next/Resume buttons + registration controls) to declutter `DashboardApp`.
2. Define compact interfaces for each widget so `DashboardApp` mainly wires callbacks (`on_start`, `on_stop`, `on_next_person`, etc.) rather than containing logic.
3. Move registration-specific controls into `dashboard/widgets/registration_panel.py` to isolate sample capture UI.

**Deliverables**
- New widget modules plus exports from `dashboard/widgets/__init__.py`.
- `DashboardApp` trimmed down by delegating layout/state to the new widgets (still using existing styling).
- Status: ✅ Completed (status/control/facebank/log/session widgets extracted; `app.py` now only composes them).

### Phase R3 – Controllers & services cleanup
1. Move `AttendanceSessionController`, `RegistrationSession`, and `AttendanceLogBook` into `dashboard/controllers/attendance.py`, `dashboard/controllers/registration.py`, and `dashboard/services/logbook.py`.
2. Add narrow interfaces for dependencies (controllers expose `start(config, callbacks)`, `stop()`, `set_display_scores(bool)`), enabling easier unit tests/mocking.
3. Relocate helper functions like `_format_exception`, `_show_error_dialog`, `_sanitize_identity_name` into `dashboard/services/errors.py` or `dashboard/utils/strings.py`.
4. Update widgets/controllers to import from the new modules, deleting redundant definitions from `app.py`.

**Deliverables**
- Controller/service modules with docstrings and unit-test hooks.
- `app.py` reduced to something like:
  ```python
  from dashboard.app import DashboardApp

  if __name__ == "__main__":
      DashboardApp().run()
  ```
- Status: ✅ Completed (controllers relocated to `dashboard/controllers/`, logbook persistence under `dashboard/services/`).

### Phase R4 – Integration polish & testing (post-refactor)
1. Revisit module boundaries to ensure no circular imports (Theme -> Widgets -> Controllers) before introducing new features.
2. Add smoke tests or scripts (e.g., `python -m dashboard.app --headless`) that instantiate the app in a non-GUI environment to fail fast if imports break.
3. Document the new structure in `README.md` (brief section about `dashboard/` package layout and responsibilities).
4. QA checklist (still using existing UI/theme):
   - App launches, session start/stop works, facebank register/delete works.
   - No regressions in attendance logging or logbook navigation.
   - Code style/linting still passes.

---

## Feature Track (After Refactor)

### Phase F1 – Settings service + window state persistence
1. Introduce `dashboard/settings.py` containing:
   - `AppSettings` dataclass representing everything currently read from `app_settings.json` plus new `window_geometry` (string) and `window_state` (e.g., "normal", "zoomed") fields.
   - `SettingsStore` class with `load()`, `save(AppSettings)`, and `update(dict)` helpers, handling default values, schema validation, and thread-safe writes.
2. Replace the ad-hoc `_load_settings` / `_handle_save_settings` logic in `ControlPanel` so that it receives an `AppSettings` instance and writes back via `SettingsStore`.
3. Wire Tk window events in `DashboardApp`:
   - When the root window is created, read `settings.window_geometry` and, if present, call `root.geometry()` and `root.state()` accordingly.
   - Bind `<Configure>` and `<Unmap>/<Map>` (or `WM_STATE` changes) to record width/height/position, debounced via `after`, and persist to `app_settings.json` through `SettingsStore`.
4. Ensure settings persistence remains backward compatible: if `window_geometry` is missing we fall back to `"1250x720"`.

**Deliverables**
- `dashboard/settings.py` with tests for JSON read/write (optional but recommended).
- `app_settings.json` schema documentation noting the new window fields.
- `DashboardApp` updated to auto-restore/save geometry without duplicating JSON logic.
- Status: ✅ Completed (`dashboard/settings.py`, ControlPanel refactor, and window geometry persistence).

### Phase F2 – Centralized theme & asset management
1. Create `dashboard/theme.py` containing:
   - `ThemeConfig` dataclass (fonts, base colors, padding constants, icon paths).
   - `ThemeManager` with methods `apply(root: tk.Tk)`, `get_font(name, size, weight)`, `icon(name)`, and `metrics` (for consistent spacing values such as `UI_PAD`, `LIVE_FEED_SIZE`).
   - Default theme values matching the current look: e.g., fonts `("TkDefaultFont", 10, "bold")`, `LIVE_FEED_SIZE`, `UI_PAD`, `METRICS_PANEL_STYLE`.
2. Convert widgets to consume the theme instead of hard-coded tuples. Example: `StatusPanel` asks the theme for `font_small_bold` and `pad.medium` instead of embedding `(12, "bold")` and `UI_PAD`.
3. Centralize icon loading (if/when we add icons) so `ThemeManager` can lazily load `ImageTk.PhotoImage` instances and reuse them.
4. Update `DashboardApp` to create a single `ThemeManager` instance and supply it to every widget/controller needing fonts, icons, or colors.

**Deliverables**
- `dashboard/theme.py` with documented API.
- Widgets updated to use theme constants rather than module-level globals.
- Removal of duplicated font/style literals from the codebase.
- Status: ✅ Completed (theme manager wired through `DashboardApp`, widgets use `theme.font()/pad()` for layout, and icon hooks exist for future assets).

### Feature QA & Notes
- **Theme**: `ThemeManager` becomes the single source for fonts (`heading`, `body`, `mono`), color palette (panels, borders, warnings), icons (play, stop, register), and spacing constants (pad sizes, `LIVE_FEED_SIZE`). Widgets receive the manager via constructor injection to avoid global state.
- **Window size persistence**: `SettingsStore` keeps `window_geometry` (e.g., `"1280x800+50+30"`) and `window_state`. `DashboardApp` binds to `<Configure>` to debounce writes (e.g., update every 750 ms when dimensions change) to avoid hammering the disk.

---

## Tracking Progress
- Update this file as phases complete (checkboxes, notes, open issues). Keep the refactor track in sync before beginning any feature-phase work so future contributors know exactly where to resume.
