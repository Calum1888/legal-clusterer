import json

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
LLM_MODEL = 'llama3.2:3b'
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

# cluster and give labels to clusters
results = clusterer.fit(cuad_data)
