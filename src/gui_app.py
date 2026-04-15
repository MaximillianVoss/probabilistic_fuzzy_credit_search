from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.credit_card_default_search import analyze_dataset as analyze_credit_card_default
from src.german_credit_search import analyze_dataset as analyze_german_credit
from src.search_analysis import DatasetAnalysis

BACKGROUND = "#f5f7fb"
SURFACE = "#ffffff"
PRIMARY = "#214f8b"
SECONDARY = "#5c7aa3"
ACCENT = "#d96c3f"
TEXT = "#1b2430"
MUTED = "#6d7a8c"
GRID = "#dfe6f1"


def format_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: (
                    f"{float(value):,.4f}"
                    if isinstance(value, (float, np.floating))
                    else f"{int(value):,}"
                )
            )
    return formatted


class DatasetTab(ttk.Frame):
    def __init__(
        self,
        master: ttk.Notebook,
        title: str,
        subtitle: str,
        loader: Callable[[], DatasetAnalysis],
    ) -> None:
        super().__init__(master, padding=18, style="App.TFrame")
        self.loader = loader
        self.title = title
        self.subtitle = subtitle
        self.analysis: DatasetAnalysis | None = None
        self.chart_canvas: FigureCanvasTkAgg | None = None

        self.status_var = tk.StringVar(value="Нажмите кнопку, чтобы загрузить данные и построить формы.")
        self.query_var = tk.StringVar(value="Запрос ещё не сформирован.")
        self.source_var = tk.StringVar(value="Источник: не загружен")
        self.raw_shape_var = tk.StringVar(value="Строки/столбцы: -")
        self.numeric_shape_var = tk.StringVar(value="Рабочая выборка: -")
        self.query_row_var = tk.StringVar(value="Индекс запроса: -")
        self.candidates_var = tk.StringVar(value="Кандидатов: -")
        self.baseline_time_var = tk.StringVar(value="-")
        self.proposed_time_var = tk.StringVar(value="-")

        self._build_layout()

    def _build_layout(self) -> None:
        header = ttk.Frame(self, style="App.TFrame")
        header.pack(fill="x", pady=(0, 12))

        title_label = ttk.Label(header, text=self.title, style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky="w")
        subtitle_label = ttk.Label(header, text=self.subtitle, style="Subtitle.TLabel")
        subtitle_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.run_button = ttk.Button(header, text="Построить анализ", command=self.run_analysis)
        self.run_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))
        header.columnconfigure(0, weight=1)

        status_label = ttk.Label(self, textvariable=self.status_var, style="Status.TLabel")
        status_label.pack(fill="x", pady=(0, 12))

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, style="Card.TFrame", padding=14)
        content = ttk.Frame(body, style="App.TFrame")
        body.add(sidebar, weight=1)
        body.add(content, weight=4)

        self._build_sidebar(sidebar)
        self._build_content(content)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        meta_frame = ttk.LabelFrame(parent, text="Сводка", padding=12)
        meta_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(meta_frame, textvariable=self.source_var, style="Body.TLabel").pack(anchor="w")
        ttk.Label(meta_frame, textvariable=self.raw_shape_var, style="Body.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(meta_frame, textvariable=self.numeric_shape_var, style="Body.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(meta_frame, textvariable=self.query_row_var, style="Body.TLabel").pack(anchor="w", pady=(6, 0))

        cards = ttk.Frame(parent, style="Card.TFrame")
        cards.pack(fill="x", pady=(0, 12))
        cards.columnconfigure((0, 1), weight=1)

        self._metric_card(cards, "Кандидаты", self.candidates_var, 0, 0)
        self._metric_card(cards, "Базовый поиск", self.baseline_time_var, 0, 1)
        self._metric_card(cards, "Ускоренный поиск", self.proposed_time_var, 1, 0, colspan=2)

        query_frame = ttk.LabelFrame(parent, text="Неточный запрос", padding=12)
        query_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(query_frame, textvariable=self.query_var, style="Body.TLabel", wraplength=280, justify="left").pack(
            fill="x"
        )

        self.target_frame = ttk.LabelFrame(parent, text="Целевой признак", padding=10)
        self.target_frame.pack(fill="both", expand=True)
        self.target_tree = self._create_tree(self.target_frame, ("value", "count"), [("value", "Значение"), ("count", "Частота")])

    def _metric_card(
        self,
        parent: ttk.Frame,
        title: str,
        variable: tk.StringVar,
        row: int,
        column: int,
        colspan: int = 1,
    ) -> None:
        card = ttk.Frame(parent, style="Metric.TFrame", padding=12)
        card.grid(row=row, column=column, columnspan=colspan, sticky="nsew", padx=4, pady=4)
        ttk.Label(card, text=title, style="CardCaption.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=variable, style="MetricValue.TLabel").pack(anchor="w", pady=(8, 0))

    def _build_content(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        tables_tab = ttk.Frame(notebook, padding=8, style="App.TFrame")
        charts_tab = ttk.Frame(notebook, padding=8, style="App.TFrame")
        notebook.add(tables_tab, text="Таблицы")
        notebook.add(charts_tab, text="Графики")

        self._build_tables_tab(tables_tab)
        self._build_charts_tab(charts_tab)

    def _build_tables_tab(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, style="App.TFrame")
        bottom = ttk.Frame(parent, style="App.TFrame")
        top.pack(fill="both", expand=True)
        bottom.pack(fill="both", expand=True, pady=(12, 0))

        left_top = ttk.LabelFrame(top, text="Статистика признаков", padding=10)
        right_top = ttk.LabelFrame(top, text="Сравнение времени", padding=10)
        left_top.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right_top.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.stats_tree = self._create_tree(
            left_top,
            ("feature", "min", "max", "mean", "std"),
            [
                ("feature", "Признак"),
                ("min", "Min"),
                ("max", "Max"),
                ("mean", "Mean"),
                ("std", "Std"),
            ],
        )

        self.time_tree = self._create_tree(
            right_top,
            ("method", "mean_seconds", "std_seconds", "processed_rows"),
            [
                ("method", "Метод"),
                ("mean_seconds", "Среднее, с"),
                ("std_seconds", "Std, с"),
                ("processed_rows", "Строки"),
            ],
        )

        left_bottom = ttk.LabelFrame(bottom, text="Top-k: базовый метод", padding=10)
        right_bottom = ttk.LabelFrame(bottom, text="Top-k: ускоренный метод", padding=10)
        left_bottom.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right_bottom.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.baseline_tree = self._create_dynamic_tree(left_bottom)
        self.proposed_tree = self._create_dynamic_tree(right_bottom)

    def _build_charts_tab(self, parent: ttk.Frame) -> None:
        self.chart_host = ttk.Frame(parent, style="Card.TFrame", padding=8)
        self.chart_host.pack(fill="both", expand=True)
        ttk.Label(
            self.chart_host,
            text="Графики появятся после запуска анализа.",
            style="Body.TLabel",
        ).pack(expand=True)

    def _create_tree(
        self,
        parent: ttk.Widget,
        columns: tuple[str, ...],
        headings: list[tuple[str, str]],
    ) -> ttk.Treeview:
        container = ttk.Frame(parent, style="App.TFrame")
        container.pack(fill="both", expand=True)

        tree = ttk.Treeview(container, columns=columns, show="headings", height=8)
        y_scroll = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        for column, title in headings:
            tree.heading(column, text=title)
            tree.column(column, width=110, anchor="center")

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        return tree

    def _create_dynamic_tree(self, parent: ttk.Widget) -> ttk.Treeview:
        container = ttk.Frame(parent, style="App.TFrame")
        container.pack(fill="both", expand=True)

        tree = ttk.Treeview(container, show="headings", height=8)
        y_scroll = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        return tree

    def run_analysis(self) -> None:
        self.run_button.configure(state="disabled")
        self.status_var.set("Загрузка датасета и построение графиков...")
        thread = threading.Thread(target=self._run_analysis_worker, daemon=True)
        thread.start()

    def _run_analysis_worker(self) -> None:
        try:
            analysis = self.loader()
        except Exception as exc:
            self.after(0, self._handle_error, exc)
            return
        self.after(0, self._apply_analysis, analysis)

    def _handle_error(self, exc: Exception) -> None:
        self.run_button.configure(state="normal")
        self.status_var.set("Ошибка при построении анализа.")
        messagebox.showerror("Ошибка", str(exc))

    def _apply_analysis(self, analysis: DatasetAnalysis) -> None:
        self.analysis = analysis
        self.run_button.configure(state="normal")
        self.status_var.set("Анализ готов. Таблицы и графики обновлены.")

        self.source_var.set(f"Источник: {analysis.source_name}")
        self.raw_shape_var.set(
            f"Строки/столбцы: {analysis.raw_shape[0]:,} / {analysis.raw_shape[1]:,}".replace(",", " ")
        )
        self.numeric_shape_var.set(
            f"Рабочая выборка: {analysis.numeric_shape[0]:,} x {analysis.numeric_shape[1]:,}".replace(",", " ")
        )
        self.query_row_var.set(f"Индекс запроса: {analysis.row_index}")
        self.candidates_var.set(f"{analysis.candidate_count:,}".replace(",", " "))

        baseline_time = float(analysis.time_comparison.loc[analysis.time_comparison["method"] == "baseline", "mean_seconds"].iloc[0])
        proposed_time = float(analysis.time_comparison.loc[analysis.time_comparison["method"] == "proposed", "mean_seconds"].iloc[0])
        self.baseline_time_var.set(f"{baseline_time:.5f} c")
        self.proposed_time_var.set(f"{proposed_time:.5f} c")

        query_lines = [f"{key}: {value:,.3f}".replace(",", " ") for key, value in analysis.query.items()]
        self.query_var.set("\n".join(query_lines))

        self._fill_tree(self.stats_tree, format_dataframe(analysis.feature_summary))
        self._fill_tree(self.time_tree, format_dataframe(analysis.time_comparison))
        self._fill_dynamic_tree(self.baseline_tree, format_dataframe(analysis.baseline_result))
        self._fill_dynamic_tree(self.proposed_tree, format_dataframe(analysis.proposed_result))

        if analysis.target_distribution is not None:
            self._fill_tree(self.target_tree, format_dataframe(analysis.target_distribution))
        else:
            self._fill_tree(self.target_tree, pd.DataFrame(columns=["value", "count"]))

        self._render_charts(analysis)

    def _fill_tree(self, tree: ttk.Treeview, frame: pd.DataFrame) -> None:
        for item in tree.get_children():
            tree.delete(item)
        for row in frame.itertuples(index=False):
            tree.insert("", "end", values=list(row))

    def _fill_dynamic_tree(self, tree: ttk.Treeview, frame: pd.DataFrame) -> None:
        columns = tuple(frame.columns)
        tree.configure(columns=columns)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=110, anchor="center")
        for item in tree.get_children():
            tree.delete(item)
        for row in frame.itertuples(index=False):
            tree.insert("", "end", values=list(row))

    def _render_charts(self, analysis: DatasetAnalysis) -> None:
        for child in self.chart_host.winfo_children():
            child.destroy()

        figure = Figure(figsize=(12, 7), dpi=100, facecolor=SURFACE)
        axes = figure.subplots(2, 3)
        figure.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, hspace=0.35, wspace=0.28)
        figure.suptitle(analysis.dataset_name, fontsize=15, fontweight="bold", color=TEXT)

        for index, feature in enumerate(analysis.feature_columns):
            axis = axes[0][index]
            values = analysis.numeric_frame[feature]
            axis.hist(values, bins=24, color=PRIMARY, alpha=0.85, edgecolor="white")
            axis.axvline(analysis.query[feature], color=ACCENT, linewidth=2.0, linestyle="--")
            axis.set_title(f"Распределение: {feature}", color=TEXT, fontsize=10)
            axis.set_facecolor("#fbfcfe")
            axis.grid(color=GRID, alpha=0.8, linestyle="--", linewidth=0.7)

        time_axis = axes[1][0]
        time_axis.bar(
            analysis.time_comparison["method"],
            analysis.time_comparison["mean_seconds"],
            color=[SECONDARY, ACCENT],
            width=0.55,
        )
        time_axis.set_title("Среднее время поиска", color=TEXT, fontsize=10)
        time_axis.grid(color=GRID, alpha=0.8, axis="y", linestyle="--", linewidth=0.7)
        time_axis.set_facecolor("#fbfcfe")

        rows_axis = axes[1][1]
        rows_axis.bar(
            analysis.time_comparison["method"],
            analysis.time_comparison["processed_rows"],
            color=[SECONDARY, ACCENT],
            width=0.55,
        )
        rows_axis.set_title("Обработано записей", color=TEXT, fontsize=10)
        rows_axis.grid(color=GRID, alpha=0.8, axis="y", linestyle="--", linewidth=0.7)
        rows_axis.set_facecolor("#fbfcfe")

        target_axis = axes[1][2]
        if analysis.target_distribution is not None and not analysis.target_distribution.empty:
            target_frame = analysis.target_distribution
            categories = target_frame.iloc[:, 0].astype(str)
            counts = target_frame.iloc[:, 1]
            target_axis.bar(categories, counts, color=PRIMARY, width=0.55)
            target_axis.set_title("Целевой признак", color=TEXT, fontsize=10)
        else:
            normalized_means = analysis.normalized_frame.mean()
            target_axis.bar(normalized_means.index, normalized_means.values, color=PRIMARY, width=0.55)
            target_axis.set_title("Средние после нормализации", color=TEXT, fontsize=10)
        target_axis.grid(color=GRID, alpha=0.8, axis="y", linestyle="--", linewidth=0.7)
        target_axis.set_facecolor("#fbfcfe")

        self.chart_canvas = FigureCanvasTkAgg(figure, master=self.chart_host)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)


class CreditSearchApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Probabilistic Fuzzy Credit Search")
        self.root.geometry("1520x920")
        self.root.minsize(1280, 780)
        self.root.configure(bg=BACKGROUND)

        self._configure_style()
        self._build_root()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
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

    def _build_root(self) -> None:
        wrapper = ttk.Frame(self.root, style="App.TFrame", padding=18)
        wrapper.pack(fill="both", expand=True)

        hero = ttk.Frame(wrapper, style="Card.TFrame", padding=18)
        hero.pack(fill="x", pady=(0, 16))
        ttk.Label(hero, text="Система анализа неточного поиска в кредитных данных", style="HeroTitle.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            hero,
            text="Интерфейс под PyCharm: отдельные вкладки по датасетам, таблицы результатов и графики без вывода в консоль.",
            style="HeroSubtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        notebook = ttk.Notebook(wrapper)
        notebook.pack(fill="both", expand=True)

        notebook.add(
            DatasetTab(
                notebook,
                title="German Credit Dataset",
                subtitle="Набор для сравнения базового и ускоренного поиска по возрасту, сумме кредита и сроку.",
                loader=analyze_german_credit,
            ),
            text="German Credit",
        )
        notebook.add(
            DatasetTab(
                notebook,
                title="Credit Card Default Dataset",
                subtitle="Вкладка для анализа кредитных лимитов, возраста и первого платежа по карте.",
                loader=analyze_credit_card_default,
            ),
            text="Credit Card Default",
        )


def launch_app() -> None:
    root = tk.Tk()
    CreditSearchApp(root)
    root.mainloop()
