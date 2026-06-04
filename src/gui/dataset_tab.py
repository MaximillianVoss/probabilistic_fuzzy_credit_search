from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable
from xml.etree import ElementTree

import matplotlib

matplotlib.use("TkAgg")

import pandas as pd
from pandas.api.types import is_numeric_dtype
from matplotlib.figure import Figure

from ..datasets.base import DatasetAnalysis
from .backend import FigureCanvasTkAgg
from .theme import ACCENT, GRID, PRIMARY, SECONDARY, SURFACE, TEXT


def format_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        if is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(_format_numeric)
    return formatted.fillna("")


def _format_numeric(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return f"{int(value):,}".replace(",", " ")
    return f"{float(value):,.4f}".replace(",", " ")


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
        self.chart_canvases: dict[str, FigureCanvasTkAgg | None] = {}

        self.status_var = tk.StringVar(value="Нажмите кнопку, чтобы загрузить данные и построить формы.")
        self.query_var = tk.StringVar(value="Запрос ещё не сформирован.")
        self.source_var = tk.StringVar(value="Источник: не загружен")
        self.raw_shape_var = tk.StringVar(value="Строки/столбцы: -")
        self.numeric_shape_var = tk.StringVar(value="Рабочая выборка: -")
        self.query_row_var = tk.StringVar(value="Индекс запроса: -")
        self.candidates_var = tk.StringVar(value="-")
        self.baseline_time_var = tk.StringVar(value="-")
        self.proposed_time_var = tk.StringVar(value="-")
        self.speedup_var = tk.StringVar(value="-")
        self.export_format_var = tk.StringVar(value="CSV")

        self._build_layout()

    def _build_layout(self) -> None:
        header = ttk.Frame(self, style="App.TFrame")
        header.pack(fill="x", pady=(0, 12))

        ttk.Label(header, text=self.title, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text=self.subtitle, style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.run_button = ttk.Button(header, text="Построить анализ", command=self.run_analysis)
        self.run_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))
        header.columnconfigure(0, weight=1)

        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel").pack(fill="x", pady=(0, 12))

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, style="Card.TFrame", padding=14)
        content = ttk.Frame(body, style="App.TFrame")
        body.add(sidebar, weight=1)
        body.add(content, weight=5)

        self._build_sidebar(sidebar)
        self._build_content(content)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        meta_frame = ttk.LabelFrame(parent, text="Сводка", padding=12)
        meta_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(meta_frame, textvariable=self.source_var, style="Body.TLabel", wraplength=280, justify="left").pack(
            anchor="w"
        )
        ttk.Label(meta_frame, textvariable=self.raw_shape_var, style="Body.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(meta_frame, textvariable=self.numeric_shape_var, style="Body.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(meta_frame, textvariable=self.query_row_var, style="Body.TLabel").pack(anchor="w", pady=(6, 0))

        cards = ttk.Frame(parent, style="Card.TFrame")
        cards.pack(fill="x", pady=(0, 12))
        cards.columnconfigure((0, 1), weight=1)

        self._metric_card(cards, "Кандидаты", self.candidates_var, 0, 0)
        self._metric_card(cards, "Базовый, с", self.baseline_time_var, 0, 1)
        self._metric_card(cards, "Предлагаемый, с", self.proposed_time_var, 1, 0)
        self._metric_card(cards, "Ускорение, %", self.speedup_var, 1, 1)

        query_frame = ttk.LabelFrame(parent, text="Неточный запрос", padding=12)
        query_frame.pack(fill="both", expand=True)
        ttk.Label(query_frame, textvariable=self.query_var, style="Body.TLabel", wraplength=280, justify="left").pack(
            fill="x"
        )

    def _metric_card(
        self,
        parent: ttk.Frame,
        title: str,
        variable: tk.StringVar,
        row: int,
        column: int,
    ) -> None:
        card = ttk.Frame(parent, style="Metric.TFrame", padding=12)
        card.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
        ttk.Label(card, text=title, style="CardCaption.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=variable, style="MetricValue.TLabel").pack(anchor="w", pady=(8, 0))

    def _build_content(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        overview_tab = ttk.Frame(notebook, padding=8, style="App.TFrame")
        quantization_tab = ttk.Frame(notebook, padding=8, style="App.TFrame")
        search_tab = ttk.Frame(notebook, padding=8, style="App.TFrame")
        experiment_tab = ttk.Frame(notebook, padding=8, style="App.TFrame")

        notebook.add(overview_tab, text="Обзор")
        notebook.add(quantization_tab, text="Квантование")
        notebook.add(search_tab, text="Поиск")
        notebook.add(experiment_tab, text="Эксперимент")

        self._build_overview_tab(overview_tab)
        self._build_quantization_tab(quantization_tab)
        self._build_search_tab(search_tab)
        self._build_experiment_tab(experiment_tab)

    def _build_overview_tab(self, parent: ttk.Frame) -> None:
        tables, charts = self._create_table_chart_tabs(parent)

        left = ttk.LabelFrame(tables, text="Статистика признаков", padding=10)
        right = ttk.LabelFrame(tables, text="Целевой признак", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.stats_tree = self._create_tree(
            left,
            ("feature", "min", "max", "mean", "std"),
            [
                ("feature", "Признак"),
                ("min", "Min"),
                ("max", "Max"),
                ("mean", "Mean"),
                ("std", "Std"),
            ],
        )
        self.target_tree = self._create_dynamic_tree(right)

        self.overview_chart_host = self._create_chart_host(charts, "overview")

    def _build_quantization_tab(self, parent: ttk.Frame) -> None:
        tables, charts = self._create_table_chart_tabs(parent)
        top = ttk.Frame(tables, style="App.TFrame")
        middle = ttk.LabelFrame(tables, text="Распределение по группам", padding=10)
        top.pack(fill="both", expand=True)
        middle.pack(fill="both", expand=True, pady=(12, 0))

        left = ttk.LabelFrame(top, text="Квартильные границы", padding=10)
        right = ttk.LabelFrame(top, text="Запрос и группы", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.quantile_tree = self._create_tree(
            left,
            ("feature", "min", "q1", "median", "q3", "max"),
            [
                ("feature", "Признак"),
                ("min", "Min"),
                ("q1", "Q1"),
                ("median", "Median"),
                ("q3", "Q3"),
                ("max", "Max"),
            ],
        )
        self.query_tree = self._create_tree(
            right,
            ("feature", "query_value", "query_group", "allowed_groups"),
            [
                ("feature", "Признак"),
                ("query_value", "Значение"),
                ("query_group", "Группа"),
                ("allowed_groups", "Допустимые"),
            ],
        )
        self.group_distribution_tree = self._create_tree(
            middle,
            ("feature", "group", "count", "share"),
            [
                ("feature", "Признак"),
                ("group", "Группа"),
                ("count", "Частота"),
                ("share", "Доля"),
            ],
        )

        self.quantization_chart_host = self._create_chart_host(charts, "quantization")

    def _build_search_tab(self, parent: ttk.Frame) -> None:
        top = ttk.LabelFrame(parent, text="Шаги фильтра", padding=10)
        bottom = ttk.Frame(parent, style="App.TFrame")
        top.pack(fill="both", expand=True)
        bottom.pack(fill="both", expand=True, pady=(12, 0))

        self.filter_steps_tree = self._create_tree(
            top,
            (
                "stage",
                "feature",
                "query_value",
                "query_group",
                "allowed_groups",
                "window_left",
                "window_right",
                "matched_rows",
            ),
            [
                ("stage", "Этап"),
                ("feature", "Признак"),
                ("query_value", "Запрос"),
                ("query_group", "Группа"),
                ("allowed_groups", "Допустимые"),
                ("window_left", "Левая граница"),
                ("window_right", "Правая граница"),
                ("matched_rows", "Осталось"),
            ],
        )

        left = ttk.LabelFrame(bottom, text="Top-k: базовый метод", padding=10)
        right = ttk.LabelFrame(bottom, text="Top-k: предлагаемый метод", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.baseline_tree = self._create_dynamic_tree(left)
        self.proposed_tree = self._create_dynamic_tree(right)

    def _build_experiment_tab(self, parent: ttk.Frame) -> None:
        tables, charts = self._create_table_chart_tabs(parent)

        actions = ttk.Frame(tables, style="App.TFrame")
        actions.pack(fill="x", pady=(0, 8))

        self.export_button = ttk.Button(
            actions,
            text="Экспорт таблицы сравнения",
            command=self.export_comparison_table,
            state="disabled",
        )
        self.export_button.pack(side="right")
        self.export_format_box = ttk.Combobox(
            actions,
            textvariable=self.export_format_var,
            values=("CSV", "JSON", "XML"),
            width=7,
            state="readonly",
        )
        self.export_format_box.pack(side="right", padx=(0, 8))

        top = ttk.Frame(tables, style="App.TFrame")
        top.pack(fill="both", expand=True)

        left = ttk.LabelFrame(top, text="Сравнение времени", padding=10)
        right = ttk.LabelFrame(top, text="Итоговые метрики", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.time_tree = self._create_tree(
            left,
            ("method", "mean_seconds", "std_seconds", "processed_rows"),
            [
                ("method", "Метод"),
                ("mean_seconds", "Среднее, с"),
                ("std_seconds", "Std, с"),
                ("processed_rows", "Строки"),
            ],
        )
        self.experiment_tree = self._create_tree(
            right,
            ("metric", "value"),
            [
                ("metric", "Метрика"),
                ("value", "Значение"),
            ],
        )

        self.experiment_chart_host = self._create_chart_host(charts, "experiment")

    def _create_table_chart_tabs(self, parent: ttk.Widget) -> tuple[ttk.Frame, ttk.Frame]:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        tables = ttk.Frame(notebook, padding=8, style="App.TFrame")
        charts = ttk.Frame(notebook, padding=8, style="App.TFrame")
        notebook.add(tables, text="Таблицы")
        notebook.add(charts, text="Графики")
        return tables, charts

    def _create_chart_host(self, parent: ttk.Widget, chart_key: str) -> ttk.Frame:
        host = ttk.Frame(parent, style="Card.TFrame", padding=8)
        host.pack(fill="both", expand=True)
        ttk.Label(host, text="Графики появятся после запуска анализа.", style="Body.TLabel").pack(expand=True)
        self.chart_canvases[chart_key] = None
        return host

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
            tree.column(column, width=128, anchor="center")

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
        self.export_button.configure(state="disabled")
        self.status_var.set("Загрузка датасета, квантование и построение форм...")
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
        self.export_button.configure(state="disabled")
        self.status_var.set("Ошибка при построении анализа.")
        messagebox.showerror("Ошибка", str(exc))

    def _apply_analysis(self, analysis: DatasetAnalysis) -> None:
        self.analysis = analysis
        self.run_button.configure(state="normal")
        self.export_button.configure(state="normal")
        self.status_var.set("Анализ готов. Все разделы обновлены.")

        self.source_var.set(f"Источник: {analysis.source_name}")
        self.raw_shape_var.set(
            f"Строки/столбцы: {analysis.raw_shape[0]:,} / {analysis.raw_shape[1]:,}".replace(",", " ")
        )
        self.numeric_shape_var.set(
            f"Рабочая выборка: {analysis.numeric_shape[0]:,} x {analysis.numeric_shape[1]:,}".replace(",", " ")
        )
        self.query_row_var.set(f"Индекс запроса: {analysis.row_index}")
        self.candidates_var.set(f"{analysis.candidate_count:,}".replace(",", " "))

        baseline_time = float(
            analysis.time_comparison.loc[
                analysis.time_comparison["method"] == "baseline",
                "mean_seconds",
            ].iloc[0]
        )
        proposed_time = float(
            analysis.time_comparison.loc[
                analysis.time_comparison["method"] == "proposed",
                "mean_seconds",
            ].iloc[0]
        )
        speedup = float(
            analysis.experiment_summary.loc[
                analysis.experiment_summary["metric"] == "Ускорение, %",
                "value",
            ].iloc[0]
        )
        self.baseline_time_var.set(f"{baseline_time:.5f}")
        self.proposed_time_var.set(f"{proposed_time:.5f}")
        self.speedup_var.set(f"{speedup:.2f}")

        query_lines = [
            (
                f"{feature}: {value:,.3f} | группа {analysis.quantized_query[feature]}"
            ).replace(",", " ")
            for feature, value in analysis.query.items()
        ]
        self.query_var.set("\n".join(query_lines))

        self._fill_tree(self.stats_tree, format_dataframe(analysis.feature_summary))
        self._fill_dynamic_tree(
            self.target_tree,
            format_dataframe(analysis.target_distribution)
            if analysis.target_distribution is not None
            else pd.DataFrame(columns=["value", "count"]),
        )
        self._fill_tree(self.quantile_tree, format_dataframe(analysis.quantile_summary))
        self._fill_tree(self.query_tree, format_dataframe(analysis.query_summary))
        self._fill_tree(self.group_distribution_tree, format_dataframe(analysis.group_distribution))
        self._fill_tree(self.filter_steps_tree, format_dataframe(analysis.filter_steps))
        self._fill_dynamic_tree(self.baseline_tree, format_dataframe(analysis.baseline_result))
        self._fill_dynamic_tree(self.proposed_tree, format_dataframe(analysis.proposed_result))
        self._fill_tree(self.time_tree, format_dataframe(analysis.time_comparison))
        self._fill_tree(self.experiment_tree, format_dataframe(analysis.experiment_summary))

        self._render_overview_charts(analysis)
        self._render_quantization_charts(analysis)
        self._render_experiment_charts(analysis)

    def export_comparison_table(self) -> None:
        if self.analysis is None:
            messagebox.showwarning("Экспорт", "Сначала постройте анализ.")
            return

        selected_format = self._selected_export_format()
        extension, filetypes = self._export_dialog_options(selected_format)
        initial_name = self._default_export_filename(extension)
        file_path = filedialog.asksaveasfilename(
            title="Экспорт таблицы сравнения",
            defaultextension=extension,
            initialfile=initial_name,
            filetypes=filetypes,
        )
        if not file_path:
            return

        try:
            export_frame = self._build_comparison_export_frame(self.analysis)
            self._write_comparison_export(file_path, export_frame, selected_format)
        except OSError as exc:
            messagebox.showerror("Экспорт", f"Не удалось сохранить файл:\n{exc}")
            return

        self.status_var.set(f"Таблица сравнения экспортирована: {Path(file_path).name}")

    def _selected_export_format(self) -> str:
        selected = self.export_format_var.get().strip().upper()
        if selected in {"CSV", "JSON", "XML"}:
            return selected
        return "CSV"

    def _export_dialog_options(self, export_format: str) -> tuple[str, list[tuple[str, str]]]:
        if export_format == "JSON":
            return ".json", [("JSON", "*.json"), ("Все файлы", "*.*")]
        if export_format == "XML":
            return ".xml", [("XML", "*.xml"), ("Все файлы", "*.*")]
        return ".csv", [("CSV для Excel", "*.csv"), ("Все файлы", "*.*")]

    def _default_export_filename(self, extension: str | None = None) -> str:
        if extension is None:
            extension, _ = self._export_dialog_options(self._selected_export_format())
        dataset_name = self.analysis.dataset_name if self.analysis is not None else self.title
        safe_name = "".join(char if char.isalnum() else "_" for char in dataset_name.lower()).strip("_")
        return f"{safe_name}_comparison{extension}"

    def _write_comparison_export(
        self,
        file_path: str,
        export_frame: pd.DataFrame,
        export_format: str,
    ) -> None:
        path = Path(file_path)
        if export_format == "JSON":
            records = export_frame.to_dict(orient="records")
            path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        if export_format == "XML":
            path.write_text(self._comparison_frame_to_xml(export_frame), encoding="utf-8")
            return
        export_frame.to_csv(path, index=False, sep=";", encoding="utf-8-sig")

    def _comparison_frame_to_xml(self, export_frame: pd.DataFrame) -> str:
        root = ElementTree.Element("comparison")
        for row in export_frame.to_dict(orient="records"):
            item = ElementTree.SubElement(root, "row")
            for column, value in row.items():
                tag = self._xml_tag_from_column(column)
                ElementTree.SubElement(item, tag).text = str(value)
        ElementTree.indent(root, space="  ")
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ElementTree.tostring(
            root,
            encoding="unicode",
        )

    def _xml_tag_from_column(self, column: str) -> str:
        tags = {
            "Показатель": "metric",
            "Базовый метод": "baseline_method",
            "Предложенный метод": "proposed_method",
        }
        return tags[column]

    def _build_comparison_export_frame(self, analysis: DatasetAnalysis) -> pd.DataFrame:
        time_frame = analysis.time_comparison.set_index("method")
        metrics = analysis.experiment_summary.set_index("metric")["value"]

        baseline_rows = int(time_frame.loc["baseline", "processed_rows"])
        proposed_rows = int(time_frame.loc["proposed", "processed_rows"])
        baseline_time = float(time_frame.loc["baseline", "mean_seconds"])
        proposed_time = float(time_frame.loc["proposed", "mean_seconds"])
        candidate_reduction = float(metrics.get("Сокращение кандидатов, %", 0.0))
        speedup = float(metrics.get("Ускорение, %", 0.0))
        top_k_overlap = float(metrics.get("Совпадение top-k, %", 0.0))

        return pd.DataFrame(
            [
                {
                    "Показатель": "Обработано записей",
                    "Базовый метод": f"{baseline_rows:,}".replace(",", " "),
                    "Предложенный метод": f"{proposed_rows:,}".replace(",", " "),
                },
                {
                    "Показатель": "Время поиска",
                    "Базовый метод": f"{baseline_time:.5f} с",
                    "Предложенный метод": f"{proposed_time:.5f} с",
                },
                {
                    "Показатель": "Сокращение кандидатов",
                    "Базовый метод": "-",
                    "Предложенный метод": f"{candidate_reduction:.2f} %",
                },
                {
                    "Показатель": "Ускорение",
                    "Базовый метод": "-",
                    "Предложенный метод": f"{speedup:.2f} %",
                },
                {
                    "Показатель": "Совпадение top-k",
                    "Базовый метод": "-",
                    "Предложенный метод": f"{top_k_overlap:.2f} %",
                },
            ],
            columns=["Показатель", "Базовый метод", "Предложенный метод"],
        )

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
            tree.column(column, width=128, anchor="center")
        for item in tree.get_children():
            tree.delete(item)
        for row in frame.itertuples(index=False):
            tree.insert("", "end", values=list(row))

    def _clear_chart_host(self, host: ttk.Frame, chart_key: str) -> None:
        for child in host.winfo_children():
            child.destroy()
        self.chart_canvases[chart_key] = None

    def _render_overview_charts(self, analysis: DatasetAnalysis) -> None:
        self._clear_chart_host(self.overview_chart_host, "overview")

        figure = Figure(figsize=(11.5, 6.8), dpi=100, facecolor=SURFACE)
        axes = figure.subplots(2, 2)
        figure.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.12, hspace=0.55, wspace=0.30)
        figure.suptitle(f"{analysis.dataset_name}: обзор", fontsize=13, fontweight="bold", color=TEXT)

        for index, feature in enumerate(analysis.feature_columns):
            axis = axes[index // 2][index % 2]
            values = analysis.numeric_frame[feature]
            axis.hist(values, bins=24, color=PRIMARY, alpha=0.88, edgecolor="white")
            axis.axvline(analysis.query[feature], color=ACCENT, linewidth=2.0, linestyle="--")
            axis.set_title(f"Распределение: {feature}", color=TEXT, fontsize=10)
            axis.set_facecolor("#fbfcfe")
            axis.grid(color=GRID, alpha=0.8, linestyle="--", linewidth=0.7)

        last_axis = axes[1][1]
        if analysis.target_distribution is not None and not analysis.target_distribution.empty:
            categories = analysis.target_distribution.iloc[:, 0].astype(str)
            counts = analysis.target_distribution.iloc[:, 1]
            last_axis.bar(categories, counts, color=SECONDARY, width=0.55)
            last_axis.set_title("Целевой признак", color=TEXT, fontsize=10)
        else:
            means = analysis.normalized_frame.mean()
            last_axis.bar(means.index, means.values, color=SECONDARY, width=0.55)
            last_axis.set_title("Средние после нормализации", color=TEXT, fontsize=10)
        last_axis.set_facecolor("#fbfcfe")
        last_axis.grid(color=GRID, alpha=0.8, axis="y", linestyle="--", linewidth=0.7)

        self.chart_canvases["overview"] = FigureCanvasTkAgg(figure, master=self.overview_chart_host)
        self.chart_canvases["overview"].draw()
        self.chart_canvases["overview"].get_tk_widget().pack(fill="both", expand=True)

    def _render_quantization_charts(self, analysis: DatasetAnalysis) -> None:
        self._clear_chart_host(self.quantization_chart_host, "quantization")

        figure = Figure(figsize=(11.5, 6.8), dpi=100, facecolor=SURFACE)
        axes = figure.subplots(1, len(analysis.feature_columns))
        if len(analysis.feature_columns) == 1:
            axes = [axes]
        figure.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.16, wspace=0.30)
        figure.suptitle(f"{analysis.dataset_name}: квартильные группы", fontsize=13, fontweight="bold", color=TEXT)

        for axis, feature in zip(axes, analysis.feature_columns):
            feature_distribution = analysis.group_distribution[analysis.group_distribution["feature"] == feature]
            counts = (
                feature_distribution.set_index("group")["count"]
                .reindex([0, 1, 2, 3], fill_value=0)
                .astype(int)
            )
            axis.bar([str(group) for group in counts.index], counts.values, color=PRIMARY, width=0.55)
            axis.axvline(analysis.quantized_query[feature], color=ACCENT, linewidth=2.0, linestyle="--")
            axis.set_title(f"{feature}: группа {analysis.quantized_query[feature]}", color=TEXT, fontsize=10)
            axis.set_xlabel("Квартильная группа")
            axis.set_ylabel("Частота")
            axis.set_facecolor("#fbfcfe")
            axis.grid(color=GRID, alpha=0.8, axis="y", linestyle="--", linewidth=0.7)

        self.chart_canvases["quantization"] = FigureCanvasTkAgg(figure, master=self.quantization_chart_host)
        self.chart_canvases["quantization"].draw()
        self.chart_canvases["quantization"].get_tk_widget().pack(fill="both", expand=True)

    def _render_experiment_charts(self, analysis: DatasetAnalysis) -> None:
        self._clear_chart_host(self.experiment_chart_host, "experiment")

        figure = Figure(figsize=(11.5, 6.8), dpi=100, facecolor=SURFACE)
        axes = figure.subplots(1, 3)
        figure.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.18, wspace=0.32)
        figure.suptitle(f"{analysis.dataset_name}: сравнение методов", fontsize=13, fontweight="bold", color=TEXT)

        time_axis = axes[0]
        time_axis.bar(
            analysis.time_comparison["method"],
            analysis.time_comparison["mean_seconds"],
            color=[SECONDARY, ACCENT],
            width=0.55,
        )
        time_axis.set_title("Среднее время поиска", color=TEXT, fontsize=10)
        time_axis.set_facecolor("#fbfcfe")
        time_axis.grid(color=GRID, alpha=0.8, axis="y", linestyle="--", linewidth=0.7)

        rows_axis = axes[1]
        rows_axis.bar(
            analysis.time_comparison["method"],
            analysis.time_comparison["processed_rows"],
            color=[SECONDARY, ACCENT],
            width=0.55,
        )
        rows_axis.set_title("Обработано записей", color=TEXT, fontsize=10)
        rows_axis.set_facecolor("#fbfcfe")
        rows_axis.grid(color=GRID, alpha=0.8, axis="y", linestyle="--", linewidth=0.7)

        metrics = analysis.experiment_summary.set_index("metric")["value"]
        effect_axis = axes[2]
        effect_axis.bar(
            ["Сокращение\nкандидатов", "Ускорение\nпо времени", "Top-k\nсовпадение"],
            [
                float(metrics.get("Сокращение кандидатов, %", 0.0)),
                float(metrics.get("Ускорение, %", 0.0)),
                float(metrics.get("Совпадение top-k, %", 0.0)),
            ],
            color=[PRIMARY, ACCENT, SECONDARY],
            width=0.55,
        )
        effect_axis.set_title("Эффект предлагаемого метода, %", color=TEXT, fontsize=10)
        effect_axis.set_facecolor("#fbfcfe")
        effect_axis.grid(color=GRID, alpha=0.8, axis="y", linestyle="--", linewidth=0.7)

        self.chart_canvases["experiment"] = FigureCanvasTkAgg(figure, master=self.experiment_chart_host)
        self.chart_canvases["experiment"].draw()
        self.chart_canvases["experiment"].get_tk_widget().pack(fill="both", expand=True)
