from __future__ import annotations

from tkinter import ttk

BACKGROUND = "#f5f7fb"
SURFACE = "#ffffff"
PRIMARY = "#214f8b"
SECONDARY = "#5c7aa3"
ACCENT = "#d96c3f"
TEXT = "#1b2430"
MUTED = "#6d7a8c"
GRID = "#dfe6f1"


def configure_styles(style: ttk.Style) -> None:
    style.theme_use("clam")

    style.configure("App.TFrame", background=BACKGROUND)
    style.configure("Card.TFrame", background=SURFACE, relief="flat")
    style.configure("Metric.TFrame", background="#eef3fa")
    style.configure("Title.TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 20, "bold"))
    style.configure("Subtitle.TLabel", background=BACKGROUND, foreground=MUTED, font=("Segoe UI", 10))
    style.configure("HeroTitle.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI", 20, "bold"))
    style.configure("HeroSubtitle.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 10))
    style.configure("Status.TLabel", background=BACKGROUND, foreground=PRIMARY, font=("Segoe UI", 10, "italic"))
    style.configure("Body.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("CardCaption.TLabel", background="#eef3fa", foreground=MUTED, font=("Segoe UI", 9))
    style.configure("MetricValue.TLabel", background="#eef3fa", foreground=TEXT, font=("Segoe UI", 16, "bold"))

    style.configure("TLabelFrame", background=SURFACE, foreground=TEXT, font=("Segoe UI", 10, "bold"))
    style.configure("TLabelFrame.Label", background=SURFACE, foreground=TEXT)
    style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"))
    style.map("TNotebook.Tab", background=[("selected", SURFACE), ("active", "#eaf0f8")])
    style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=TEXT, rowheight=28)
    style.configure("Treeview.Heading", background="#e8eef8", foreground=TEXT, font=("Segoe UI", 9, "bold"))
    style.map("Treeview", background=[("selected", "#d8e4f6")], foreground=[("selected", TEXT)])
    style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
