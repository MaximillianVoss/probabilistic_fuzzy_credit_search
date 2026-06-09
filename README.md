# probabilistic_fuzzy_credit_search

<!-- codex-repo-note:start -->
## Справка о репозитории / Repository note

**RU:** исследовательский проект по вероятностному нечеткому поиску кредитных решений.

**EN:** a research project for probabilistic fuzzy credit-decision search.

**Статус / Status:** активный проект 2026 года; ожидает рефакторинга и переименования. / active 2026 project; refactoring and repository rename are pending.

**Текущее имя / Current name:** `probabilistic_fuzzy_credit_search`

**Плановое имя / Planned name:** `probabilistic-fuzzy-credit-search`

**Topics:** `cleanup-pending`, `credit-scoring`, `fuzzy-logic`, `needs-rename`, `needs-review`, `probabilistic-models`, `python`, `status-active`, `type-research`
<!-- codex-repo-note:end -->


Стартовый проект под PyCharm по теме: ускорение неточного поиска в БД на основе вероятностных соотношений.

В проекте есть GUI-приложение и пакет с аналитическими датасетами:

- `main.py` запускает оконное приложение с вкладками, таблицами и графиками
- `src/datasets/base.py` содержит базовый класс `AnalyticalDataset` и общую аналитику
- `src/datasets/german_credit.py` для `German Credit Dataset`
- `src/datasets/credit_card_default.py` для `Credit Card Default Dataset`
- `src/datasets/credit_approval.py` для `Credit Approval Dataset`
- `src/gui/app.py`, `src/gui/dataset_tab.py`, `src/gui/theme.py` отвечают за интерфейс

Что уже подготовлено:

- отдельные папки `data/german_credit` и `data/credit_card_default`
- автоматическая загрузка официальных версий датасетов с UCI, если локальных файлов ещё нет
- единый каркас для:
  - загрузки данных
  - выбора числовых признаков
  - построения неточного запроса
  - квартильного квантования по группам `0..3`
  - сравнения базового полного перебора и предлагаемого поиска через квартильную и числовую фильтрацию
- визуальный интерфейс на `tkinter`:
  - вкладки по датасетам
  - разделы `Обзор`, `Квантование`, `Поиск`, `Эксперимент`
  - подписи и карточки с ключевыми метриками
  - таблицы `top-k`, шагов фильтра и итоговых метрик
  - встроенные графики распределений, квартильных групп и сравнения методов

Если позже будут скачаны именно Kaggle-версии датасетов, их можно положить в папки:

- `data/german_credit/german_credit_data.csv`
- `data/credit_card_default/UCI_Credit_Card.csv`
- `data/credit_approval/crx.data`

Тогда скрипты автоматически возьмут локальные файлы вместо UCI-источников.

Как открыть и запустить в PyCharm:

1. Открыть папку `probabilistic_fuzzy_credit_search` как проект.
2. Создать виртуальное окружение.
3. Установить зависимости: `pip install -r requirements.txt`
4. Для визуального режима запускать `main.py`.
5. При необходимости можно импортировать конкретные классы датасетов из `src/datasets`.

Запуск тестов:

- `python -m unittest discover -s tests -v`
