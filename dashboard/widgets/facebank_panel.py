from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageOps, ImageTk

from pipelines.attendance import DEFAULT_FACEBANK
from dashboard.theme import ThemeManager
from dashboard.utils import _sanitize_identity_name



class FacebankPanel(ttk.Frame):
    """Display and manage facebank identities."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_register: Callable[[], None],
        on_refresh: Callable[..., None],
        show_error_dialog: Callable[[str, str], None],
        theme: ThemeManager,
    ) -> None:
        super().__init__(master)
        self.on_register = on_register
        self.on_refresh = on_refresh
        self._show_error_dialog = show_error_dialog
        self.theme = theme
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._active_identity: Optional[str] = None
        self._thumb_labels: dict[Path, tk.Label] = {}
        self._selected_samples: set[Path] = set()
        self._busy = False
        self._busy_message = "Facebank is refreshing, please wait for it to finish."

        pad = theme.pad()
        half_pad = max(1, pad // 2)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=pad, pady=(pad, half_pad))
        header.columnconfigure(0, weight=1)

        self.count_var = tk.StringVar(value="Identities: 0")
        ttk.Label(
            header,
            textvariable=self.count_var,
            font=theme.font("heading"),
            anchor="center",
        ).grid(row=0, column=0, sticky="ew")

        content = ttk.Frame(self)
        content.grid(row=1, column=0, sticky="nsew", padx=pad, pady=(pad, half_pad))
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        list_frame = ttk.Frame(content)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, half_pad))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self._list_frame = list_frame
        self._list_max_width = 440
        self._list_frame.grid_propagate(False)

        self.listbox = tk.Listbox(list_frame, height=18)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_identity_select)
        content.bind("<Configure>", self._handle_content_resize)

        thumbs_frame = ttk.LabelFrame(content, text="Samples")
        thumbs_frame.grid(row=0, column=1, sticky="nsew", padx=(half_pad, 0))
        thumbs_frame.columnconfigure(0, weight=1)
        thumbs_frame.rowconfigure(0, weight=1)
        self.thumbs_canvas = tk.Canvas(thumbs_frame, highlightthickness=0)
        self.thumbs_canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(thumbs_frame, orient="vertical", command=self.thumbs_canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self.thumbs_canvas.configure(yscrollcommand=vscroll.set)
        thumbs_inner = ttk.Frame(self.thumbs_canvas)
        self._thumb_window = self.thumbs_canvas.create_window((0, 0), window=thumbs_inner, anchor="nw")
        self.thumbs_inner = thumbs_inner
        self._thumb_images: list[ImageTk.PhotoImage] = []
        self._thumb_labels = {}
        self._selected_samples = set()
        temp_label = tk.Label(self)
        self._sample_default_bg = temp_label.cget("background")
        temp_label.destroy()
        self.thumbs_canvas.bind("<Configure>", lambda event: self._relayout_thumbnails())
        self.thumbs_canvas.bind_all("<MouseWheel>", self._on_thumb_scroll)
        self.thumbs_inner.bind("<Configure>", lambda event: self._relayout_thumbnails())

        footer = ttk.Frame(self)
        footer.grid(row=2, column=0, sticky="ew", padx=pad, pady=(pad, pad))
        footer.columnconfigure(0, weight=1)
        button_row = ttk.Frame(footer)
        button_row.pack(side="left")
        self.register_button = ttk.Button(button_row, text="Register User", command=self.on_register, width=18)
        self.register_button.grid(row=0, column=0, padx=4)
        self.rename_button = ttk.Button(button_row, text="Rename User", command=self._rename_selected, width=18)
        self.rename_button.grid(row=0, column=1, padx=4)
        self.delete_button = ttk.Button(button_row, text="Delete User", command=self._delete_selected, width=18)
        self.delete_button.grid(row=0, column=2, padx=4)
        samples_row = ttk.Frame(footer)
        samples_row.pack(side="right")
        self.delete_samples_button = ttk.Button(
            samples_row,
            text="Delete Selected",
            width=18,
            command=self._delete_selected_samples,
            state="disabled",
        )
        self.delete_samples_button.grid(row=0, column=0, padx=(0, 6))
        self.clear_selection_button = ttk.Button(
            samples_row,
            text="Cancel Selection",
            width=18,
            command=self._clear_sample_selection,
            state="disabled",
        )
        self.clear_selection_button.grid(row=0, column=1)

        self.after_idle(self._update_list_width)
        self.refresh()

    def set_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self._busy = busy
        if busy and message:
            self._busy_message = message
        elif not busy:
            self._busy_message = "Facebank is refreshing, please wait for it to finish."
        for button in (
            getattr(self, "register_button", None),
            getattr(self, "rename_button", None),
            getattr(self, "delete_button", None),
            getattr(self, "delete_samples_button", None),
            getattr(self, "clear_selection_button", None),
        ):
            if button is None:
                continue
            if busy:
                button.state(["disabled"])
            else:
                button.state(["!disabled"])
        self._update_sample_action_state()
        try:
            self.listbox.configure(state="disabled" if busy else "normal")
        except tk.TclError:
            pass

    def _ensure_ready(self) -> bool:
        if not self._busy:
            return True
        messagebox.showinfo("Facebank busy", self._busy_message, parent=self)
        return False

    def refresh(self) -> None:
        names: list[str] = []
        DEFAULT_FACEBANK.mkdir(parents=True, exist_ok=True)
        try:
            entries = sorted(DEFAULT_FACEBANK.iterdir())
        except OSError as exc:
            self._show_error_dialog("Facebank access error", f"Unable to read facebank directory: {exc}", parent=self)
            entries = []
        for entry in entries:
            try:
                if entry.is_dir() and not entry.name.startswith("."):
                    names.append(entry.name)
            except OSError:
                continue
        self.listbox.delete(0, tk.END)
        for name in names:
            self.listbox.insert(tk.END, name)
        self.count_var.set(f"Identities: {len(names)}")
        if names:
            if self._active_identity in names:
                target = self._active_identity
            else:
                target = names[0]
                self._active_identity = target
            idx = names.index(target)
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
        else:
            self._active_identity = None
        self._load_thumbnails()

    def _handle_content_resize(self, event: tk.Event) -> None:
        self._update_list_width(event.width)

    def _update_list_width(self, available_width: Optional[int] = None) -> None:
        if not hasattr(self, "_list_frame") or not self._list_frame.winfo_exists():
            return
        if available_width is None or available_width <= 0:
            try:
                available_width = self._list_frame.master.winfo_width()
            except tk.TclError:
                return
        target = min(self._list_max_width, max(available_width, 0))
        self._list_frame.configure(width=target)

    def _selected_names(self) -> list[str]:
        indices = self.listbox.curselection()
        return [self.listbox.get(idx) for idx in indices]

    def _on_identity_select(self, event=None) -> None:
        names = self._selected_names()
        self._active_identity = names[0] if names else None
        self._load_thumbnails()

    def _load_thumbnails(self) -> None:
        for widget in self.thumbs_inner.winfo_children():
            widget.destroy()
        self._thumb_images.clear()
        self._thumb_labels = {}
        self._selected_samples.clear()
        selection = self._selected_names()
        if len(selection) != 1:
            self._update_sample_action_state()
            self._relayout_thumbnails()
            return
        person = selection[0]
        person_dir = DEFAULT_FACEBANK / person
        if not person_dir.exists():
            self._update_sample_action_state()
            self._relayout_thumbnails()
            return
        image_paths = sorted([p for p in person_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not image_paths:
            self._update_sample_action_state()
            self._relayout_thumbnails()
            return
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.fit(img, (120, 120), method=Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._thumb_images.append(photo)
                label = tk.Label(self.thumbs_inner, image=photo, borderwidth=2, relief="flat", cursor="hand2")
                label.grid(row=0, column=0, padx=4, pady=4)
                label.bind("<Button-1>", lambda event, p=img_path: self._toggle_sample_selection(p))
                self._thumb_labels[img_path] = label
            except Exception:
                continue
        self._relayout_thumbnails()
        self._update_sample_action_state()

    def _update_thumb_scrollregion(self) -> None:
        self.thumbs_canvas.configure(scrollregion=self.thumbs_canvas.bbox("all"))
        self.thumbs_canvas.itemconfigure(self._thumb_window, width=self.thumbs_canvas.winfo_width())

    def _relayout_thumbnails(self) -> None:
        children = self.thumbs_inner.winfo_children()
        if not children:
            self._update_thumb_scrollregion()
            return
        width = max(self.thumbs_canvas.winfo_width(), 1)
        thumb_w = 120 + 8
        cols = max(1, int(width / thumb_w))
        for idx, child in enumerate(children):
            row = idx // cols
            col = idx % cols
            child.grid_configure(row=row, column=col, padx=4, pady=4)
        self.thumbs_inner.update_idletasks()
        self._update_thumb_scrollregion()

    def _on_thumb_scroll(self, event) -> None:
        widget_path = str(event.widget)
        if widget_path.startswith(str(self.thumbs_canvas)) or widget_path.startswith(str(self.thumbs_inner)):
            self.thumbs_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _toggle_sample_selection(self, path: Path) -> None:
        label = self._thumb_labels.get(path)
        if label is None:
            return
        if path in self._selected_samples:
            self._selected_samples.remove(path)
            label.config(relief="flat", background=self._sample_default_bg)
        else:
            self._selected_samples.add(path)
            label.config(relief="solid", background="#ffedc2")
        self._update_sample_action_state()

    def _clear_sample_selection(self) -> None:
        for path in list(self._selected_samples):
            label = self._thumb_labels.get(path)
            if label:
                label.config(relief="flat", background=self._sample_default_bg)
        self._selected_samples.clear()
        self._update_sample_action_state()

    def _update_sample_action_state(self) -> None:
        if self._busy or not self._selected_samples:
            self.delete_samples_button.state(("disabled",))
            self.clear_selection_button.state(("disabled",))
            return
        self.delete_samples_button.state(("!disabled",))
        self.clear_selection_button.state(("!disabled",))

    def _delete_selected_samples(self) -> None:
        if not self._ensure_ready():
            return
        if not self._selected_samples:
            messagebox.showinfo("Delete samples", "Select at least one sample.", parent=self)
            return
        count = len(self._selected_samples)
        person = self._active_identity or "identity"
        if not messagebox.askyesno(
            "Delete samples",
            f"Delete {count} sample(s) for '{person}'? This cannot be undone.",
            parent=self,
        ):
            return
        errors: list[str] = []
        for path in list(self._selected_samples):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"{Path(path).name}: {exc}")
        if errors:
            self._show_error_dialog("Delete samples", "\n".join(errors), parent=self)
        self._selected_samples.clear()
        self._load_thumbnails()
        self._request_refresh(
            status_message="Deleting samples... refreshing facebank.",
            success_message="Facebank refreshed after deleting samples.",
        )

    def _rename_selected(self) -> None:
        if not self._ensure_ready():
            return
        names = self._selected_names()
        if len(names) != 1:
            messagebox.showinfo("Rename identity", "Select a single identity to rename.", parent=self)
            return
        current = names[0]
        new_name = simpledialog.askstring("Rename identity", f"Enter new name for '{current}':", parent=self)
        if not new_name:
            return
        sanitized = _sanitize_identity_name(new_name)
        if not sanitized:
            self._show_error_dialog("Rename identity", "Name cannot be empty.", parent=self)
            return
        if sanitized == current:
            return
        src = DEFAULT_FACEBANK / current
        dst = DEFAULT_FACEBANK / sanitized
        if dst.exists():
            self._show_error_dialog("Rename identity", f"'{sanitized}' already exists.", parent=self)
            return
        try:
            src.rename(dst)
        except OSError as exc:
            self._show_error_dialog("Rename identity", f"Unable to rename: {exc}", parent=self)
            return
        self.refresh()
        self._request_refresh(
            status_message=f"Renaming '{current}'... refreshing facebank.",
            success_message=f"Facebank refreshed after renaming '{current}' to '{sanitized}'.",
        )

    def _delete_selected(self) -> None:
        if not self._ensure_ready():
            return
        names = self._selected_names()
        if not names:
            messagebox.showinfo("Delete identity", "Select at least one identity to delete.", parent=self)
            return
        count = len(names)
        plural = "ies" if count != 1 else "y"
        confirm = messagebox.askyesno(
            "Delete identities",
            f"Delete {count} identit{plural}? This cannot be undone.",
            parent=self,
        )
        if not confirm:
            return
        for name in names:
            target = DEFAULT_FACEBANK / name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        self.refresh()
        word = "identity" if count == 1 else "identities"
        self._request_refresh(
            status_message=f"Deleting {count} {word}... refreshing facebank.",
            success_message=f"Facebank refreshed after deleting {count} {word}.",
        )

    def _request_refresh(self, *, status_message: str, success_message: str) -> None:
        if callable(self.on_refresh):
            try:
                self.on_refresh(status_message=status_message, success_message=success_message)
            except TypeError:
                self.on_refresh()
