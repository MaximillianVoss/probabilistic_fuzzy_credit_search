from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..datasets import analyze_credit_card_default, analyze_german_credit
from .dataset_tab import DatasetTab
from .theme import BACKGROUND, configure_styles


class CreditSearchApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Probabilistic Fuzzy Credit Search")
        self.root.geometry("1520x920")
        self.root.minsize(1280, 780)
        self.root.configure(bg=BACKGROUND)

        configure_styles(ttk.Style(self.root))
        self._build_root()

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
