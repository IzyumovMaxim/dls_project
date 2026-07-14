# FEVER Evidence Search

Поисковик доказательств: находим пассажи из Wikipedia по утверждению и подсвечиваем предложение-доказательство. Также по ссылке можно перейти на страницу Wikipedia, чтобы прочитать статью полностью.
Проект курса **Deep Learning for Search** (Innopolis University, 2026).

**Основной бенчмарк — [BeIR/fever](https://huggingface.co/datasets/BeIR/fever)** (123 142 утверждения) поверх корпуса в 500 000 пассажей. [BeIR/climate-fever](https://huggingface.co/datasets/BeIR/climate-fever) — дополнительный out-of-domain бенчмарк на **том же** индексе.

Метрики: P@k, Recall@k, MRR, nDCG@10, latency, time. Каждый эксперимент задаётся одним YAML-конфигом.

## Как устроен поиск

**Оффлайн:**

1. Корпус 500k строится из FEVER: все документы из qrels + случайно выбранные пассажи (reservoir sampling, seed 42).
2. `intfloat/e5-base-v2` кодирует пассажи в 768-мерные нормализованные векторы. Т.к. e5 обучен с префиксами, запрос идет как `query: …`, документ как `passage: …`.
3. Векторы уходят в FAISS `IndexFlatIP`, скалярное произведение равно косинусу из-за нормализации. Так получается точный поиск и источник эмбеддингов, из которых без перекодирования собираются все остальные индексы.
4. Приложение работает на **OPQ-сжатой копии тех же векторов** (`e5_base_opq192`): каждый вектор — 192 байта кодов вместо 3 КБ fp32. Индекс занимает 95 МБ вместо 1.4 ГБ, поиск при этом даже быстрее, а качество — nDCG@10 0.9154 против 0.9206 у точного.
5. Отдельно кодируются **все 2.26 млн предложений** корпуса (fp16, 3.5 ГБ) и массив оффсетов: документ *i* владеет строками `[offsets[i], offsets[i+1])`. Сжатый индекс хранит коды, а не векторы, поэтому предложения он читает из каталога эталонного индекса (`index.vectors_from`).

**На каждый запрос:**

1. Запрос кодируется один раз (~45 мс на CPU).
2. FAISS отдаёт top-10.
3. Пассаж режется на предложения тем же модулем `fever_search.text`, которым пользовался оффлайн-билдер.
4. Из memmap поднимаются только векторы предложений найденных документов (~44 строки), умножаются на вектор запроса, argmax даёт предложение-доказательство. **0.1 мс.**

Вторая итерация — дообучение: hard negatives майнятся по индексу, модель дообучается на `MultipleNegativesRankingLoss`.

## Структура проекта

```
dls_project/
├── app.py                       # Streamlit UI
├── configs/                     # 1 YAML = 1 эксперимент
│   ├── e5_base_opq192.yaml      # обслуживает приложение
│   ├── e5_base_flat.yaml        # эталон: точный поиск + источник эмбеддингов
│   ├── e5_base_binary_rerank.yaml
│   ├── e5_base_{ivf,ivfpq,ivfpq192,hnsw}.yaml
│   ├── e5_base_pq192.yaml       # тот же размер кода, но без OPQ-вращения
│   ├── e5_base_ft.yaml          # дообученная модель
│   └── bge_{small,large}_flat.yaml
├── src/fever_search/
│   ├── config.py                # dataclass + load_config(yaml)
│   ├── paths.py                 # артефакты адресуются именем конфига
│   ├── data_io.py               # corpus / queries / qrels
│   ├── encoder.py               # обёртка SentenceTransformer
│   ├── index.py                 # FAISS: flat | pq | ivf | ivfpq | hnsw | binary_rerank
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
uv run streamlit run app.py     # -> запускается на http://localhost:8501
```

## Данные

Формат BeIR: `corpus` / `queries` (`{_id, title, text}`) / `qrels` (TSV). Сплит train/validation/test задаётся **только в qrels**.

В корпус попадают все gold-документы из qrels, остальное добирается случайной выборкой. Распределения длин gold docs и случайно отобранных сравниваются в `data/analysis/figures/`.

## Пайплайн

```bash
# 1. загрузка данных (выполняется один раз)
python scripts/data/export_fever.py
python scripts/data/export_climate.py
python scripts/data/build_corpus.py
python scripts/data/analyze.py

# 2. сборка индекса
python scripts/index/build_index.py --config configs/e5_base_flat.yaml
python scripts/index/build_sentence_index.py --config configs/e5_base_flat.yaml --device cuda

# другие типы индексов строятся из тех же эмбеддингов, без перекодирования корпуса
python scripts/index/build_index.py --config configs/e5_base_binary_rerank.yaml \
    --from-embeddings data/index/e5_base_flat

# индекс, который обслуживает приложение (обучение OPQ-вращения ~8 минут, разово)
python scripts/index/build_index.py --config configs/e5_base_opq192.yaml \
    --from-embeddings data/index/e5_base_flat

# 3. оценка на тестовых данных
python scripts/evaluate.py --config configs/e5_base_flat.yaml --benchmark fever --split test

# 4. дообучение (итерация 2)
python scripts/train/mine_negatives.py --config configs/e5_base_flat.yaml
python scripts/train/train.py --config configs/e5_base_ft.yaml
python scripts/index/build_index.py --config configs/e5_base_ft.yaml --model-path models/e5_base_ft

# 5. все замеры одной командой
python scripts/bench/benchmark_all.py --with-bm25
```

`benchmark_all.py` сохраняет `data/analysis/RESULTS.md` (таблицы) и `benchmark_all.json` (сырые числа).

## Результаты

Приложение работает на конфигурации `configs/e5_base_opq192.yaml`. Выбрали именно ее после сравнения основных вариантов индексов:

| | RAM индекса | RSS процесса | поиск | nDCG@10 | P@1 |
|---|---|---|---|---|---|
| flat (точный) | 1.4 ГБ | 2.1 ГБ | 43 мс | 0.9206 | 0.8858 |
| **opq192** | **95 МБ** | **1.1 ГБ** | **37 мс** | 0.9154 | 0.8767 |

Почему выбрали OPQ192:

**Приоритет — сжатие, а не ускорение поиска.** IVF и HNSW ускоряют поиск до долей миллисекунды, но снижают качество retrieval. В нашем случае скорость поиска не является узким местом: кодирование запроса занимает около 50 мс на CPU, и для пользователя разница между 37 мс и 1 мс поиска по индексу незаметна. Зато есть разница в памяти: 1.4 ГБ против 95 МБ.

**Размер кода 192 байта вместо 96.** При стандартных 96 суб-квантизаторах потеря в nDCG@10 составляет −0.048 — слишком большая цена за экономию. Увеличение кода до 192 байт снижает потерю до −0.0052, менее 0.6% от baseline. При этом индекс занимает менее 100 МБ.

**OPQ вместо обычного PQ.** OPQ-вращение дает прирост P@1 на +0.0038 по сравнению с PQ. Это превосходит разброс качества от случайной инициализации k-means при обучении кодбуков PQ, который оценили в 0.0013. Единственный недостаток OPQ - около 8 минут на разовое обучение вращения при сборке индекса.

Полные таблицы со всеми конфигурациями и сырые числа генерируются через `scripts/bench/benchmark_all.py` и доступны в `data/analysis/RESULTS.md` и `benchmark_all.json`.

## Технические требования

**Инференс**: достаточно CPU, GPU не требуется. Пиковое потребление памяти — около 1.1 ГБ RSS (включая модель кодирования и OPQ-индекс). Время обработки одного запроса: ~50 мс на кодирование + 37 мс на поиск + 0.1 мс на поиск предложения-доказательства.

**Сборка индекса**: кодирование 500 000 пассажей и 2.26 млн предложений, а также дообучение модели, практичны только на GPU. Итоговые артефакты занимают около 5 ГБ на диске (индексы + модели + sentence-векторы).

## Ссылки на датасеты и использованные модели

- [BeIR/fever](https://huggingface.co/datasets/BeIR/fever) · [BeIR/climate-fever](https://huggingface.co/datasets/BeIR/climate-fever)
- [FAISS](https://github.com/facebookresearch/faiss) · [Sentence Transformers](https://www.sbert.net/) · [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
