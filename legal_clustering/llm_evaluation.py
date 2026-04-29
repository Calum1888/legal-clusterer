from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import random


class LLMEvaluation:
    def __init__(
        self,
        llm_model: str,
        max_tokens: int,
        token_price: float,
        n_llm_samples: int,
        prompt_type_of_doc: str,
        seed: int
    ):
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.token_price = token_price
        self.n_llm_samples = n_llm_samples
        self.prompt_type_of_doc = prompt_type_of_doc
        self.seed = seed

        self._build_pipeline()

    def _build_pipeline(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        self._hf_llm = pipeline(
            task="text-generation",
            model=self.llm_model,
            device_map="auto",
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=0.3
        )

    def count_price_tokens(self, prompt: str) -> dict:
        num_tokens = len(self._tokenizer.encode(prompt))
        return {
            "number_of_tokens": num_tokens,
            "price": num_tokens * self.token_price,
        }

    def llm_label(self, id_and_label: dict) -> dict:
        clusters = {}
        for doc_id, label in zip(id_and_label):
            clusters.setdefault(int(label), []).append(doc_id)

        generated_cluster_labels = {}
        for cluster_id, doc_ids in tqdm(clusters.items(), desc="Labelling clusters"):
            sample = random.sample(doc_ids, min(self.n_llm_samples, len(doc_ids)))
            prompt = (
                f"These {self.prompt_type_of_doc} were grouped together by a "
                "clustering algorithm:\n\n"
                + "\n".join(sample)
                + "\n\nRespond with only a short 3-5 word label describing "
                f"what {self.prompt_type_of_doc} these are. No explanation."
            )

            generated_cluster_labels[cluster_id] = self._hf_llm(prompt)[0]["generated_text"].replace(prompt, "").strip()

        return generated_cluster_labels

    def error_detection(self, cluster_id: int, generated_labels: dict, id_and_label: dict) -> dict:
        cluster_label = generated_labels[cluster_id]

        doc_titles = [
            doc_id for doc_id, label in zip(id_and_label)
            if int(label) == cluster_id
        ]

        random.seed(self.seed)
        sample = random.sample(doc_titles, min(self.n_llm_samples, len(doc_titles)))

        checking_prompt = (
            f"A clustering algorithm grouped these {self.prompt_type_of_doc} together "
            f'and labelled the cluster: "{cluster_label}".\n\n'
            + "\n".join(sample)
            + "\n\nDo these titles all belong to the same type? "
            "Reply with YES or NO, then a one sentence explanation."
        )
        verdict = self._hf_llm(checking_prompt)[0]["generated_text"].replace(checking_prompt, "").strip()

        return {
            "cluster_id": cluster_id,
            "label": cluster_label,
            "verdict": verdict,
        }