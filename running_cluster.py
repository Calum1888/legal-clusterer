import json
from legal_clustering import DocumentClusterer, LLMEvaluation

# data
IN_FILE = "data/CUADv1.json"

# clustering parameters
NGRAM_RANGE = (1,3)
N_COMPONENTS = 100
N_ITERATIONS = 7
DISTANCE_THRESHOLD = 1.5
LINKAGE = 'ward'
INPUT_TYPE = 'content'
RANDOM_STATE = 42
LLM_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
MAX_TOKENS = 50
TOKEN_PRICE = 0.0001
N_LLM_SAMPLES = 5
PROMPT_TYPE_OF_DOC = 'legal contracts'

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
    input_type=INPUT_TYPE,
    random_state=RANDOM_STATE
)

# cluster CUAD data
results = clusterer.fit(cuad_data)

# define evaluator and share state from clusterer
evaluator = LLMEvaluation(
    llm_model=LLM_MODEL,  # HuggingFace model name
    max_tokens=MAX_TOKENS,
    token_price=TOKEN_PRICE,
    n_llm_samples=N_LLM_SAMPLES,
    prompt_type_of_doc=PROMPT_TYPE_OF_DOC,
)

# pass cluster state from clusterer to evaluator
evaluator.doc_ids_ = clusterer.doc_ids_
evaluator.labels_ = clusterer.labels_

# generate labels for each cluster
cluster_labels = evaluator.llm_label()
print(cluster_labels)

# check a specific cluster for coherence
verdict = evaluator.error_detection(
    cluster_id=0,
    generated_labels=cluster_labels
)
print(verdict)
print(evaluator.count_price_tokens())
