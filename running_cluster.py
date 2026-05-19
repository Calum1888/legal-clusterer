import json

from legal_clustering import DocumentClusterer, LLMEvaluation, EmbeddingClusterer

# data
IN_FILE = "data/CUADv1.json"

# clustering parameters
NGRAM_RANGE = (1,3)
N_COMPONENTS = 50
N_ITERATIONS = 7
DISTANCE_THRESHOLD = 2.5
LINKAGE = 'ward'
METRIC = 'euclidean'
INPUT_TYPE = 'content'
RANDOM_STATE = 42

# LLM evaluation parameters 
LLM_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
MAX_TOKENS = 100
TOKEN_PRICE = 0.0001
N_LLM_SAMPLES = 5
PROMPT_TYPE_OF_DOC = 'legal contracts'
BATCH_SIZE = 4
MIN_CLUSTER_SIZE = 2
EXCERPT_CHARAS = 2000

# LLM clustering parameters
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_CHARS = 2000

# read in data 
with open(IN_FILE) as fs:
    cuad = json.load(fs)

cuad_data = {
    doc["title"]: doc["paragraphs"][0]["context"]
    for doc in cuad["data"]
}

# define clusterer
clusterer = DocumentClusterer(
    ngram=NGRAM_RANGE,
    n_components=N_COMPONENTS,
    n_iter=N_ITERATIONS,
    dist_threshold=DISTANCE_THRESHOLD,
    linkage=LINKAGE,
    metric=METRIC,
    input_type=INPUT_TYPE,
    random_state=RANDOM_STATE
)

# define llm evaluator
llm_eval = LLMEvaluation(
    llm_model=LLM_MODEL,
    max_tokens=MAX_TOKENS,
    token_price=TOKEN_PRICE,
    n_llm_samples=N_LLM_SAMPLES,
    prompt_type_of_doc=PROMPT_TYPE_OF_DOC,
    seed=RANDOM_STATE,
    batch_size=BATCH_SIZE,
    min_cluster_size=MIN_CLUSTER_SIZE,
    excerpt_chars=EXCERPT_CHARAS   
)

llm_clusterer = LLMClusterer(
    embedding_model=EMBEDDING_MODEL,
    dist_threshold=DISTANCE_THRESHOLD,
    linkage=LINKAGE,
    metric=METRIC,
    max_chars=MAX_CHARS,
    batch_size=BATCH_SIZE,
    random_state=RANDOM_STATE
)

# cluster CUAD data
results = clusterer.fit(cuad_data)

# cluster CUAD data with LLMClusterer
emb_results = llm_clusterer.fit(cuad_data)

# llm labels
labels = llm_eval.llm_label(results, cuad_data)

print("\n=== Generated cluster labels ===")
for cid, lbl in sorted(labels.items()):
    print(f"Cluster {cid}: {lbl}")

# Verify each cluster: does the LLM agree the docs belong together?
verdicts = llm_eval.evaluate_all(labels, results, cuad_data)

passed = sum(v["passed"] for v in verdicts)
print(f"\n{passed}/{len(verdicts)} clusters passed verification\n")

# Show the ones that failed
print("=== Failed clusters ===")
for v in verdicts:
    if not v["passed"]:
        print(f"\nCluster {v['cluster_id']} — label: {v['label']!r}")
        print(f"  verdict: {v['verdict']}")