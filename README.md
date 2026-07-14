# FEVER Evidence Search

Векторный поиск доказательств: по claim'у (утверждению) находим пассажи Wikipedia, которые его подтверждают или опровергают, и подсвечиваем конкретное предложение-доказательство. Проект курса **Deep Learning for Search** (Innopolis, 2026).

**Основной бенчмарк — [BeIR/fever](https://huggingface.co/datasets/BeIR/fever)** (123 142 claim'а) поверх корпуса в 500 000 пассажей. [BeIR/climate-fever](https://huggingface.co/datasets/BeIR/climate-fever) — дополнительный out-of-domain бенчмарк на **том же** индексе.

Метрики: P@k, Recall@k, MRR, nDCG@10 плюс латентность и память. Каждый эксперимент задаётся одним YAML-конфигом.

## Как устроен поиск

**Оффлайн:**

1. Корпус 500k строится из FEVER: все gold-документы из qrels плюс случайный филлер (reservoir sampling, seed 42).
2. `intfloat/e5-base-v2` кодирует пассажи в 768-мерные нормализованные векторы. e5 обучен с префиксами, поэтому запрос идёт как `query: …`, документ как `passage: …`.
3. Векторы уходят в FAISS `IndexFlatIP` — раз векторы нормализованы, скалярное произведение равно косинусу.
4. Отдельно кодируются **все 2.26 млн предложений** корпуса (fp16, 3.5 ГБ) плюс массив оффсетов: документ *i* владеет строками `[offsets[i], offsets[i+1])`.

**На запрос:**

1. Запрос кодируется один раз (~45 мс на CPU — самая дорогая операция всего пути).
2. FAISS отдаёт top-10.
3. Пассаж режется на предложения тем же модулем `fever_search.text`, которым пользовался оффлайн-билдер — иначе предпосчитанные векторы указывали бы не на те предложения.
4. Из memmap поднимаются только векторы предложений найденных документов (~44 строки), умножаются на вектор запроса, argmax даёт предложение-доказательство. **0.1 мс.**

Вторая итерация — дообучение: hard negatives майнятся по индексу, модель дообучается на `MultipleNegativesRankingLoss`.

## Структура

```
dls_project/
├── app.py                       # Streamlit UI
├── configs/                     # 1 YAML = 1 эксперимент
│   ├── e5_base_flat.yaml        # обслуживает приложение
│   ├── e5_base_binary_rerank.yaml
│   ├── e5_base_{ivf,ivfpq,hnsw}.yaml
│   ├── e5_base_ft.yaml          # дообученная модель
│   └── bge_{small,large}_flat.yaml
├── src/fever_search/
│   ├── config.py                # dataclass + load_config(yaml)
│   ├── paths.py                 # артефакты адресуются именем конфига
│   ├── data_io.py               # corpus / queries / qrels
│   ├── encoder.py               # обёртка SentenceTransformer
│   ├── index.py                 # FAISS: flat | ivf | ivfpq | hnsw | binary_rerank
│   ├── search.py                # SearchEngine + поиск предложения-доказательства
│   ├── text.py                  # разбиение на предложения (общее для билдера и рантайма)
│   ├── bench.py                 # латентность, память, метрики
│   ├── eval.py                  # метрики + run_eval
│   └── train/                   # mining + MNRL fine-tune
├── scripts/
│   ├── data/                    # export_fever, export_climate, build_corpus, analyze
│   ├── index/                   # build_index, build_sentence_index, tune_ann
│   ├── train/                   # mine_negatives, train
│   ├── bench/                   # benchmark_all + отдельные оси
│   ├── evaluate.py, demo.py
├── data/                        # артефакты, не в git
└── models/                      # чекпоинты, не в git
```

## Установка

```bash
uv sync
```

Скрипты сами добавляют `src/` в путь. На Linux torch ставится из CUDA-индекса (`pytorch-cu124`), на macOS — с PyPI; пин в `pyproject.toml` привязан к платформе.

Запуск приложения:

```bash
uv run streamlit run app.py     # -> http://localhost:8501
```

## Данные

Формат BeIR: `corpus` / `queries` (`{_id, title, text}`) / `qrels` (TSV). Сплит train/validation/test задаётся **только в qrels**.

**Про срез 500k.** В корпус обязательно попадают все gold-документы из qrels, остальное добирается случайной выборкой. Срез **намеренно смещён**: без gold-документов в индексе метрики были бы бессмысленны. Как следствие, наши числа выше, чем были бы на полных 5.4M FEVER — это цена того, чтобы вообще иметь измеримый потолок. Распределения длин gold и филлера сравниваются в `data/analysis/figures/`.

## Пайплайн

```bash
# 1. данные (один раз)
python scripts/data/export_fever.py
python scripts/data/export_climate.py
python scripts/data/build_corpus.py
python scripts/data/analyze.py

# 2. индекс
python scripts/index/build_index.py --config configs/e5_base_flat.yaml
python scripts/index/build_sentence_index.py --config configs/e5_base_flat.yaml --device cuda

# другие типы индексов строятся из тех же эмбеддингов, без перекодирования корпуса
python scripts/index/build_index.py --config configs/e5_base_binary_rerank.yaml \
    --from-embeddings data/index/e5_base_flat

# 3. оценка
python scripts/evaluate.py --config configs/e5_base_flat.yaml --benchmark fever --split test

# 4. дообучение (итерация 2)
python scripts/train/mine_negatives.py --config configs/e5_base_flat.yaml
python scripts/train/train.py --config configs/e5_base_ft.yaml
python scripts/index/build_index.py --config configs/e5_base_ft.yaml --model-path models/e5_base_ft

# 5. все замеры одной командой
python scripts/bench/benchmark_all.py --with-bm25
```

`benchmark_all.py` пишет `data/analysis/RESULTS.md` (таблицы) и `benchmark_all.json` (сырые числа), чтобы любую цифру из презентации можно было проследить до прогона, который её произвёл.

## Результаты

Таблицы — в `data/analysis/RESULTS.md`, сырые числа — в `benchmark_all.json`. Оба файла генерируются `scripts/bench/benchmark_all.py`.

Приложение обслуживает `configs/e5_base_flat.yaml` — точный поиск.

## Требования к железу

**Инференс**: CPU, GPU не нужен. Разбивка по стадиям и RSS — в разделе *Serving* в `RESULTS.md`.

**Сборка**: кодирование 500k пассажей и 2.26M предложений практично только на GPU. Дообучение — там же.

## Ссылки

- [BeIR/fever](https://huggingface.co/datasets/BeIR/fever) · [BeIR/climate-fever](https://huggingface.co/datasets/BeIR/climate-fever)
- [FAISS](https://github.com/facebookresearch/faiss) · [Sentence Transformers](https://www.sbert.net/) · [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
