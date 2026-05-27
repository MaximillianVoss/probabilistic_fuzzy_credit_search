from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..datasets import (
    analyze_credit_approval,
    analyze_credit_card_default,
    analyze_german_credit,
)
from .dataset_tab import DatasetTab
from .theme import BACKGROUND, configure_styles


class CreditSearchApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Probabilistic Fuzzy Credit Search")
        self._set_window_size()
        self.root.configure(bg=BACKGROUND)

        configure_styles(ttk.Style(self.root))
        self._build_root()

    def _set_window_size(self) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        min_width = min(980, max(860, screen_width - 80))
        min_height = min(640, max(600, screen_height - 120))
        width = min(1520, max(min_width, screen_width - 80))
        height = min(920, max(min_height, screen_height - 120))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min_width, min_height)

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
            text="Интерфейс под PyCharm: внутри каждого датасета есть разделы Обзор, Квантование, Поиск и Эксперимент.",
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
        notebook.add(
            DatasetTab(
                notebook,
                title="Credit Approval Dataset",
                subtitle="Датасет заявок на кредит с непрерывными признаками A2, A3 и A8 и бинарным решением по одобрению.",
                loader=analyze_credit_approval,
            ),
            text="Credit Approval",
        )


def launch_app() -> None:
    root = tk.Tk()
    CreditSearchApp(root)
    root.mainloop()
