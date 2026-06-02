from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
import unittest
from tkinter import ttk
from unittest.mock import Mock, patch

from pandas.core.frame import DataFrame

from src.datasets.base import run_analysis
from src.gui.app import CreditSearchApp, launch_app
from src.gui.backend import FigureCanvasTkAgg
from src.gui.dataset_tab import DatasetTab, format_dataframe
from src.gui.theme import configure_styles


def build_sample_analysis():
    frame = DataFrame(
        {
            "income": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "balance": [100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0],
            "duration": [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0],
            "target": ["A", "A", "B", "B", "A", "B", "A", "B"],
        }
    )
    return run_analysis(
        dataset_name="GUI Synthetic Dataset",
        frame=frame,
        source_name="gui-test",
        feature_columns=["income", "balance", "duration"],
        query_perturbation={"income": 1.0, "balance": 1.0, "duration": 1.0},
        weights={"income": 1.0, "balance": 1.0, "duration": 1.0},
        relative_window=0.2,
        neighbor_radius=1,
        alpha=3.0,
        top_k=3,
        repeats=1,
        target_column="target",
        minimum_span=1.0,
    )


class GuiTests(unittest.TestCase):
    def test_format_dataframe_formats_numeric_columns(self) -> None:
        frame = DataFrame({"x": [1.23456, None], "y": [1000, 2000], "label": ["a", None]})

        formatted = format_dataframe(frame)

        self.assertEqual(formatted.iloc[0]["x"], "1.2346")
        self.assertEqual(formatted.iloc[0]["y"], "1 000")
        self.assertEqual(formatted.iloc[1]["label"], "")

    def test_configure_styles_uses_clam_theme(self) -> None:
        style = Mock()

        configure_styles(style)

        style.theme_use.assert_called_once_with("clam")
        self.assertTrue(style.configure.called)
        self.assertTrue(style.map.called)

    def test_credit_search_app_builds_three_dataset_tabs(self) -> None:
        root = tk.Tk()
        try:
            app = CreditSearchApp(root)
            root.update_idletasks()

            wrapper = root.winfo_children()[0]
            notebook = next(child for child in wrapper.winfo_children() if isinstance(child, ttk.Notebook))
            self.assertEqual(len(notebook.tabs()), 3)
            self.assertEqual(root.title(), "Probabilistic Fuzzy Credit Search")
            self.assertIsInstance(app, CreditSearchApp)
        finally:
            root.destroy()

    def test_launch_app_creates_app_and_starts_mainloop(self) -> None:
        fake_root = Mock()

        with patch("src.gui.app.tk.Tk", return_value=fake_root) as tk_ctor:
            with patch("src.gui.app.CreditSearchApp") as app_ctor:
                launch_app()

        tk_ctor.assert_called_once_with()
        app_ctor.assert_called_once_with(fake_root)
        fake_root.mainloop.assert_called_once_with()

    def test_dataset_tab_apply_analysis_updates_ui_state(self) -> None:
        analysis = build_sample_analysis()
        root = tk.Tk()
        try:
            configure_styles(ttk.Style(root))
            notebook = ttk.Notebook(root)
            notebook.pack()
            tab = DatasetTab(notebook, "Title", "Subtitle", loader=lambda: analysis)
            notebook.add(tab, text="Synthetic")

            tab._apply_analysis(analysis)
            root.update_idletasks()

            self.assertEqual(tab.status_var.get(), "Анализ готов. Все разделы обновлены.")
            self.assertGreater(len(tab.stats_tree.get_children()), 0)
            self.assertGreater(len(tab.time_tree.get_children()), 0)
            self.assertGreater(len(tab.baseline_tree.get_children()), 0)
            self.assertIsInstance(tab.chart_canvases["overview"], FigureCanvasTkAgg)
            self.assertIsInstance(tab.chart_canvases["quantization"], FigureCanvasTkAgg)
            self.assertIsInstance(tab.chart_canvases["experiment"], FigureCanvasTkAgg)
            self.assertEqual(str(tab.export_button["state"]), "normal")
        finally:
            root.destroy()

    def test_dataset_tab_export_button_is_disabled_before_analysis(self) -> None:
        analysis = build_sample_analysis()
        root = tk.Tk()
        try:
            configure_styles(ttk.Style(root))
            notebook = ttk.Notebook(root)
            notebook.pack()
            tab = DatasetTab(notebook, "Title", "Subtitle", loader=lambda: analysis)

            self.assertEqual(str(tab.export_button["state"]), "disabled")
        finally:
            root.destroy()

    def test_dataset_tab_builds_comparison_export_table(self) -> None:
        analysis = build_sample_analysis()
        root = tk.Tk()
        try:
            configure_styles(ttk.Style(root))
            notebook = ttk.Notebook(root)
            notebook.pack()
            tab = DatasetTab(notebook, "Title", "Subtitle", loader=lambda: analysis)

            frame = tab._build_comparison_export_frame(analysis)

            self.assertEqual(list(frame.columns), ["Показатель", "Базовый метод", "Предложенный метод"])
            self.assertEqual(frame.iloc[0]["Показатель"], "Обработано записей")
            self.assertEqual(frame.iloc[0]["Базовый метод"], "8")
            self.assertEqual(frame.iloc[0]["Предложенный метод"], "1")
            self.assertIn("с", frame.iloc[1]["Базовый метод"])
            self.assertIn("%", frame.iloc[2]["Предложенный метод"])
        finally:
            root.destroy()

    def test_dataset_tab_exports_comparison_table_to_csv(self) -> None:
        analysis = build_sample_analysis()
        root = tk.Tk()
        try:
            configure_styles(ttk.Style(root))
            notebook = ttk.Notebook(root)
            notebook.pack()
            tab = DatasetTab(notebook, "Title", "Subtitle", loader=lambda: analysis)
            tab._apply_analysis(analysis)

            with TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "comparison.csv"
                with patch("src.gui.dataset_tab.filedialog.asksaveasfilename", return_value=str(output_path)):
                    tab.export_comparison_table()

                exported = output_path.read_text(encoding="utf-8-sig")

            self.assertIn("Показатель;Базовый метод;Предложенный метод", exported)
            self.assertIn("Обработано записей;8;1", exported)
            self.assertIn("Таблица сравнения экспортирована", tab.status_var.get())
        finally:
            root.destroy()

    def test_dataset_tab_export_without_analysis_shows_warning(self) -> None:
        analysis = build_sample_analysis()
        root = tk.Tk()
        try:
            configure_styles(ttk.Style(root))
            notebook = ttk.Notebook(root)
            notebook.pack()
            tab = DatasetTab(notebook, "Title", "Subtitle", loader=lambda: analysis)

            with patch("src.gui.dataset_tab.messagebox.showwarning") as showwarning:
                with patch("src.gui.dataset_tab.filedialog.asksaveasfilename") as asksaveasfilename:
                    tab.export_comparison_table()

            showwarning.assert_called_once_with("Экспорт", "Сначала постройте анализ.")
            asksaveasfilename.assert_not_called()
        finally:
            root.destroy()

    def test_dataset_tab_export_cancel_keeps_current_status(self) -> None:
        analysis = build_sample_analysis()
        root = tk.Tk()
        try:
            configure_styles(ttk.Style(root))
            notebook = ttk.Notebook(root)
            notebook.pack()
            tab = DatasetTab(notebook, "Title", "Subtitle", loader=lambda: analysis)
            tab._apply_analysis(analysis)
            status_before_export = tab.status_var.get()

            with patch("src.gui.dataset_tab.filedialog.asksaveasfilename", return_value=""):
                tab.export_comparison_table()

            self.assertEqual(tab.status_var.get(), status_before_export)
        finally:
            root.destroy()

    def test_dataset_tab_export_write_error_shows_message(self) -> None:
        analysis = build_sample_analysis()
        root = tk.Tk()
        try:
            configure_styles(ttk.Style(root))
            notebook = ttk.Notebook(root)
            notebook.pack()
            tab = DatasetTab(notebook, "Title", "Subtitle", loader=lambda: analysis)
            tab._apply_analysis(analysis)

            with TemporaryDirectory() as temp_dir:
                missing_path = Path(temp_dir) / "missing" / "comparison.csv"
                with patch("src.gui.dataset_tab.filedialog.asksaveasfilename", return_value=str(missing_path)):
                    with patch("src.gui.dataset_tab.messagebox.showerror") as showerror:
                        tab.export_comparison_table()

            showerror.assert_called_once()
            self.assertIn("Не удалось сохранить файл", showerror.call_args.args[1])
        finally:
            root.destroy()

    def test_dataset_tab_default_export_filename_uses_dataset_name(self) -> None:
        analysis = build_sample_analysis()
        root = tk.Tk()
        try:
            configure_styles(ttk.Style(root))
            notebook = ttk.Notebook(root)
            notebook.pack()
            tab = DatasetTab(notebook, "Title", "Subtitle", loader=lambda: analysis)
            tab._apply_analysis(analysis)

            self.assertEqual(tab._default_export_filename(), "gui_synthetic_dataset_comparison.csv")
        finally:
            root.destroy()

    def test_dataset_tab_handle_error_restores_button_and_shows_message(self) -> None:
        root = tk.Tk()
        try:
            configure_styles(ttk.Style(root))
            notebook = ttk.Notebook(root)
            notebook.pack()
            tab = DatasetTab(notebook, "Title", "Subtitle", loader=lambda: None)
            notebook.add(tab, text="Synthetic")
            tab.run_button.configure(state="disabled")

            with patch("src.gui.dataset_tab.messagebox.showerror") as showerror:
                tab._handle_error(RuntimeError("boom"))

            self.assertEqual(str(tab.run_button["state"]), "normal")
            self.assertEqual(tab.status_var.get(), "Ошибка при построении анализа.")
            showerror.assert_called_once()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
