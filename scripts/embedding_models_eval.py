
import json
import random
import torch.nn.functional as F
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_JSONL_PATH = PROJECT_ROOT / "data" / "corpus" / "fever_500k.jsonl"
QRELS_TEST_PATH = PROJECT_ROOT / "data" / "qrels" / "qrels_test.tsv"
MODEL_1 = "BAAI/bge-large-en-v1.5"
MODEL_2 = "intfloat/e5-base-v2"

QUERIES_PATH = PROJECT_ROOT / "data" / "queries" / "queries.jsonl"
QRELS_TEST_PATH = PROJECT_ROOT / "data" / "qrels" / "qrels_test.tsv"
QUALITY_DIR = PROJECT_ROOT / "data" / "quality"


EVAL_TOP_K = 100
K_VALUES = [1, 5, 10, 100]

qrels: dict[str, set[str]] = defaultdict(set)
test_ids = set(qrels)
queries: dict[str, str] = {}

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "std": round(float(arr.std()), 4),
    }

with CORPUS_JSONL_PATH.open(encoding="utf-8") as c:
    corpus = json.load(c)

with QRELS_TEST_PATH.open(encoding="utf-8") as q:
    qrels = json.load(q)
    next(q)
    for line in q:
        query_id, corpus_id, _ = line.rstrip("\n").split("\t")
        qrels[query_id].add(corpus_id)

with QUERIES_PATH.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            query_id = str(row["_id"])
            if query_id in test_ids:
                queries[query_id] = str(row.get("text") or "").strip()

sentences = [i for i in CORPUS_JSONL_PATH]
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
embeddings_1 = model_1.encode(corpus, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# Load model from HuggingFace Hub
tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_1)
model_1 = AutoModel.from_pretrained(MODEL_1)
model_1.eval()

# Tokenize sentences
encoded_input = tokenizer_1(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model_1(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = ['query: how much protein should a female eat',
               'query: summit define',
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
               "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."]

tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_2)
model_2 = AutoModel.from_pretrained(MODEL_2)

# Tokenize the input texts
batch_dict = tokenizer_2(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model_2(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
