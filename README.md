# FEVER Evidence Search

Векторный поиск доказательств по датасету [BEIR/FEVER](https://huggingface.co/datasets/BeIR/fever): по claim (утверждению) находим релевантные пассажи Wikipedia.

Проект для курса **Deep Learning for Search** (Innopolis, 2026).

## Задача

**Стадия 1 (этот репозиторий):** retrieval — по query найти top-k документов из корпуса.  
Оценка: nDCG@10, Recall@k, MRR на `qrels_test` (6 666 test claims).

**Стадия 2 (опционально):** NLI-классификатор SUPPORTS / REFUTES / NOT ENOUGH INFO поверх найденных пассажей.

## Структура репозитория

```
dls_project/
├── data/
│   ├── corpus/           # fever_500k.jsonl (не в git, ~300 MB)
│   ├── queries/          # queries.jsonl — все claim'ы
│   ├── qrels/            # qrels_{train,validation,test}.tsv
│   ├── index/            # эмбеддинги + FAISS (не в git)
│   ├── index/            # эмбеддинги + FAISS (не в git)
│   ├── analysis/         # отчёты, графики, hash.md (не в git)
│   └── quality/          # метрики retrieval (не в git)
├── scripts/
│   ├── create_test.py    # выгрузка queries + qrels с HuggingFace
│   ├── create_corpus.py  # сборка корпуса 500k
│   ├── analysis.py       # статистика датасета и графики
│   ├── corpus_vector_bge_small_en_v1.5.py  # индексация корпуса
│   ├── query_search_bge_small_en_v1.5.py   # API векторного поиска
│   ├── terminal_test_bge_small_en_v1.5.py  # демо в терминале
│   └── test_bge_small_en_v1.5.py           # eval на test split
├── requirements.txt
└── README.md
```

## Данные

| Компонент | Источник | Формат |
|-----------|----------|--------|
| Corpus | `BeIR/fever` | JSONL: `{_id, title, text}` |
| Queries | `BeIR/fever` | JSONL: `{_id, title, text}` — claim в `text` |
| Qrels | `BeIR/fever-qrels` | TSV: `query-id`, `corpus-id`, `score` |

Сплит train/validation/test задаётся **только в qrels** (в `queries` одного файла на все 123k claim'ов).

### Корпус 500k

Срез из 5.42M пассажей:

1. **Все** документы-доказательства из `qrels_test` (~1 499 шт.) — обязательно в индексе.
2. Остальное — случайная добивка до 500 000 (reservoir sampling, seed=42).

Без gold-документов метрики на test были бы невалидны.

## Быстрый старт

```bash
cd dls_project
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 1. Выгрузить queries и qrels

```bash
python scripts/create_test.py
```

Создаёт `data/queries/queries.jsonl` и `data/qrels/qrels_*.tsv`.

### 2. Собрать корпус 500k

```bash
python scripts/create_corpus.py
```

Создаёт `data/corpus/fever_500k.jsonl` (streaming по полному корпусу, ~5–10 мин).

### 3. Аналитика датасета

```bash
python scripts/analysis.py
```

Создаёт `data/analysis/report.json`, `README.md`, `hash.md`, графики в `figures/`.

## Векторный поиск

```bash
python scripts/corpus_vector_bge_small_en_v1.5.py   # индекс (один раз)
python scripts/terminal_test_bge_small_en_v1.5.py   # демо в терминале
python scripts/test_bge_small_en_v1.5.py            # eval → data/quality/
```

- **Модель:** `BAAI/bge-small-en-v1.5`
- **Индекс:** FAISS `IndexFlatIP`
- **Метрики:** Precision@k, Recall@k, MRR, nDCG@10

## Что коммитить в git

| Путь | В git? |
|------|--------|
| `scripts/`, `requirements.txt`, `README.md`, `.gitignore` | да |
| `data/**` | **нет** — только пустые папки (`.gitkeep`) |

После клонирования репозитория:

```bash
pip install -r requirements.txt
python scripts/create_test.py      # queries + qrels
python scripts/create_corpus.py    # corpus 500k
python scripts/analysis.py         # аналитика
python scripts/corpus_vector_bge_small_en_v1.5.py
```

## Команда / защита

- Презентация: до 15 мин + 5 мин вопросы (неделя 7, 15 июля)
- Нужно: value proposition, архитектура, 2+ итерации (модель / индекс), сравнительная таблица метрик и железа

## Ссылки

- [BeIR/fever](https://huggingface.co/datasets/BeIR/fever)
- [BeIR/fever-qrels](https://huggingface.co/datasets/BeIR/fever-qrels)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
