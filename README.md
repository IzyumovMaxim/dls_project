# FEVER Evidence Search

Векторный поиск доказательств: по claim (утверждению) находим релевантные пассажи Wikipedia. Проект курса **Deep Learning for Search** (Innopolis, 2026).

**Основная задача — [BeIR/fever](https://huggingface.co/datasets/BeIR/fever)** (123 142 claim'а) поверх единого корпуса 500k. [BeIR/climate-fever](https://huggingface.co/datasets/BeIR/climate-fever) (1 535 claim'ов) — дополнительный out-of-domain бенчмарк на **том же** индексе, оценивается отдельно.

Метрики: nDCG@10, Recall@k, MRR. Итерации задаются **конфигами** (модель / тип индекса / дообучение) — по одному YAML на эксперимент.

## Структура

```
dls_project/
├── configs/                  # 1 YAML = 1 эксперимент
│   ├── bge_small_flat.yaml   ├── bge_small_ivf.yaml
│   ├── bge_small_hnsw.yaml   └── bge_small_ft.yaml   # дообученная модель
├── src/fever_search/         # библиотека
│   ├── config.py             # dataclass + load_config(yaml)
│   ├── paths.py              # пути к data/, артефакты по имени конфига
│   ├── data_io.py            # чтение corpus/queries/qrels
│   ├── encoder.py            # обёртка SentenceTransformer
│   ├── index.py              # FAISS build/load: flat | ivf | hnsw
│   ├── search.py             # SearchEngine
│   ├── eval.py               # метрики + run_eval
│   └── train/
│       ├── mine.py           # hard-negative mining
│       └── train.py          # MNRL fine-tune
├── scripts/                  # тонкие CLI (вся логика в пакете)
│   ├── data/                 # подготовка данных (не config-driven)
│   │   ├── export_fever.py   ├── export_climate.py
│   │   ├── build_corpus.py   └── analyze.py
│   ├── build_index.py    ├── evaluate.py    ├── demo.py
│   ├── mine_negatives.py └── train.py
├── data/                     # артефакты, не в git
├── models/                   # чекпоинты дообучения, не в git
└── pyproject.toml            # зависимости + пакет (uv)
```

## Установка (uv)

```bash
uv sync                       # создаёт .venv + uv.lock из pyproject.toml
```

Дальше либо `uv run <cmd>`, либо активировать `.venv`. Скрипты сами добавляют `src/` в путь, установка пакета необязательна.

## Данные

Формат BeIR: `corpus` / `queries` (`{_id, title, text}`) / `qrels` (TSV `query-id`, `corpus-id`, `score`). Сплит train/validation/test задаётся **только в qrels** (FEVER — все три, climate-fever — только `test`).

| Бенчмарк | Корпус | Индекс | Файлы |
|----------|--------|--------|-------|
| **FEVER** (основной) | 5.4M Wikipedia → срез 500k | per-config | `data/queries/`, `data/qrels/` |
| **Climate-FEVER** | тот же Wikipedia | тот же индекс | `data/climate-fever/` |

**Корпус 500k:** все gold-документы из qrels FEVER (train/val/test) и climate-fever (test) обязательно в индексе (потолок Recall/nDCG = 1.0), остальное — reservoir sampling (seed=42). Собрано 500 000 (15 613 gold + 484 387 filler), FEVER test-gold покрыт 100%; 57 climate-evidence отсутствуют в дампе fever → `data/corpus/missing_gold_ids.txt`.

## Пайплайн

```bash
# 1. данные (один раз)
python scripts/data/export_fever.py
python scripts/data/export_climate.py
python scripts/data/build_corpus.py        # ~5-10 мин
python scripts/data/analyze.py             # статистика -> data/analysis

# 2. индекс + eval для конфига
python scripts/build_index.py --config configs/bge_small_flat.yaml
python scripts/evaluate.py   --config configs/bge_small_flat.yaml --benchmark fever   --split test
python scripts/evaluate.py   --config configs/bge_small_flat.yaml --benchmark climate --split test
python scripts/demo.py       --config configs/bge_small_flat.yaml

# 3. дообучение (итерация: hard negatives + MNRL)
python scripts/mine_negatives.py --config configs/bge_small_ft.yaml   # использует индекс bge_small_flat
python scripts/train.py          --config configs/bge_small_ft.yaml   # -> models/bge_small_ft
python scripts/build_index.py    --config configs/bge_small_ft.yaml --model-path models/bge_small_ft
python scripts/evaluate.py       --config configs/bge_small_ft.yaml --model-path models/bge_small_ft --benchmark fever --split test
```

Артефакты ключуются именем конфига: индекс → `data/index/<name>/`, метрики → `data/quality/<name>/<benchmark>/report.json`. Сравнительная таблица итераций собирается из этих `report.json`.

## Конфиг

```yaml
name: bge_small_flat          # имя = ключ артефактов
model:
  name: BAAI/bge-small-en-v1.5
  batch_size: 64
index:
  type: flat                  # flat | ivf (nlist/nprobe) | hnsw (hnsw_m/ef_*)
eval:
  top_k: 100
  k_values: [1, 5, 10, 100]
train:                        # только для train.py
  base_config: bge_small_flat # индекс для майнинга hard negatives
  epochs: 1
  hard_negatives: 4
```

## Git

В git — `src/`, `scripts/`, `configs/`, `pyproject.toml`, `README.md`, `.gitignore`. Всё под `data/**` и `models/` игнорируется и восстанавливается запуском скриптов.

## Ссылки

- [BeIR/fever](https://huggingface.co/datasets/BeIR/fever) · [BeIR/climate-fever](https://huggingface.co/datasets/BeIR/climate-fever)
- [FAISS](https://github.com/facebookresearch/faiss) · [Sentence Transformers](https://www.sbert.net/)
