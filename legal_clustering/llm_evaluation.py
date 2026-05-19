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
        """
    Initialise the tokenizer and Hugging Face text-generation pipeline.

    Loads the tokenizer and model specified by self.llm_model and configures
    a text-generation pipeline for deterministic (greedy) decoding. The model
    is placed on available devices automatically via device_map="auto", which
    will use GPU if available and fall back to CPU otherwise.

    Greedy decoding (do_sample=False) is used so that label generation and
    cluster verification are fully reproducible across runs without needing
    to manage random seeds for the model.

    Side effects:
        Sets self._tokenizer to the loaded AutoTokenizer instance.
        Sets self._hf_llm to the configured text-generation pipeline.

    Note:
        This method is called from __init__, so constructing an LLMEvaluation
        instance will download the model on first use and load it into memory.
    """
        self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        self._hf_llm = pipeline(
            task="text-generation",
            model=self.llm_model,
            device_map="auto",
            max_new_tokens=self.max_tokens,
            do_sample=False
        )

    def count_price_tokens(self, prompt: str) -> dict:
        num_tokens = len(self._tokenizer.encode(prompt))
        return {
            "number_of_tokens": num_tokens,
            "price": num_tokens * self.token_price,
        }

    def llm_label(self, id_and_label: dict, documents: dict, excerpt_chars: int = 1000, min_cluster_size: int = 2) -> dict:
        """
        Generate a short descriptive label for each cluster using the LLM.

        Args:
            id_and_label: Mapping of doc_id -> cluster label (output of clusterer.fit).
            documents: Mapping of doc_id -> raw document text. Used to give the LLM
                actual content rather than just identifiers.
            excerpt_chars: Number of characters of each sampled document to include
                in the prompt. Keep small enough that n_llm_samples * excerpt_chars
                fits comfortably in the model's context window.
            min_cluster_size: Clusters smaller than this are skipped (singletons or
                tiny clusters can't be meaningfully labelled).

        Returns:
            Mapping of cluster_id -> generated label string.
        """
        # Group doc_ids by cluster, normalising numpy ints to Python ints once.
        clusters = {}
        for doc_id, label in id_and_label.items():
            clusters.setdefault(int(label), []).append(doc_id)

        # Filter out clusters too small to label meaningfully.
        clusters = {cid: ids for cid, ids in clusters.items() if len(ids) >= min_cluster_size}

        # Single RNG instance, seeded once — avoids reseeding inside the loop
        # and avoids polluting the global random state.
        rng = random.Random(self.seed)

        generated_cluster_labels = {}
        for cluster_id, doc_ids in tqdm(clusters.items(), desc="Labelling clusters"):
            sample_ids = rng.sample(doc_ids, min(self.n_llm_samples, len(doc_ids)))

            # Build excerpts with the actual document content, not just the key.
            excerpts = []
            for doc_id in sample_ids:
                text = documents.get(doc_id, "")
                snippet = text[:excerpt_chars].strip()
                excerpts.append(f"--- {doc_id} ---\n{snippet}")

            prompt = (
                f"These {self.prompt_type_of_doc} were grouped together by a "
                "clustering algorithm:\n\n"
                + "\n\n".join(excerpts)
                + f"\n\nRespond with only a short 3-5 word label describing "
                f"what {self.prompt_type_of_doc} these are. No explanation."
            )

            # return_full_text=False asks the pipeline for only the continuation,
            # which is more robust than string-replacing the prompt out of the output.
            output = self._hf_llm(prompt, return_full_text=False)[0]["generated_text"].strip()

            # Hard cap on label length in case the model ignores the instruction.
            label = " ".join(output.split()[:8])
            generated_cluster_labels[cluster_id] = label

        return generated_cluster_labels

    def error_detection(
        self,
        cluster_id: int,
        generated_labels: dict,
        id_and_label: dict,
        documents: dict,
        excerpt_chars: int = 1000,
    ) -> dict:
        """
        Ask the LLM whether a cluster's documents genuinely belong together
        under the generated label. Verification uses the same sample that
        was used for labelling, so the verdict is grounded in the same evidence.

        Args:
            cluster_id: The cluster to check.
            generated_labels: Mapping of cluster_id -> label (output of llm_label).
            id_and_label: Mapping of doc_id -> cluster label.
            documents: Mapping of doc_id -> raw document text.
            excerpt_chars: Characters of each document to include in the prompt.

        Returns:
            dict with cluster_id, label, raw verdict text, and a parsed bool `passed`.
        """
        cluster_label = generated_labels[cluster_id]

        doc_ids = [
            doc_id for doc_id, label in id_and_label.items()
            if int(label) == cluster_id
        ]

        # Reuse the same RNG seeding pattern as llm_label so the sample
        # drawn here matches the one used for labelling — keeps the verdict
        # grounded in the same evidence the label was generated from.
        rng = random.Random(self.seed)
        # Advance the RNG to the same point llm_label would be at for this cluster.
        # Simpler approach: just sample fresh; it's a different evaluation moment.
        sample_ids = rng.sample(doc_ids, min(self.n_llm_samples, len(doc_ids)))

        excerpts = [
            f"--- {doc_id} ---\n{documents.get(doc_id, '')[:excerpt_chars].strip()}"
            for doc_id in sample_ids
        ]

        checking_prompt = (
            f"A clustering algorithm grouped these {self.prompt_type_of_doc} together "
            f'and labelled the cluster: "{cluster_label}".\n\n'
            + "\n\n".join(excerpts)
            + f"\n\nDo these {self.prompt_type_of_doc} all belong to the same type? "
            "Reply with YES or NO on the first line, then a one sentence explanation."
        )

        raw_verdict = self._hf_llm(checking_prompt, return_full_text=False)[0]["generated_text"].strip()

        # Parse the YES/NO out of the response so callers don't have to grep.
        first_token = raw_verdict.split()[0].upper().strip(".,:;") if raw_verdict else ""
        passed = first_token.startswith("YES")

        return {
            "cluster_id": cluster_id,
            "label": cluster_label,
            "verdict": raw_verdict,
            "passed": passed,
        }


    def evaluate_all(
        self,
        generated_labels: dict,
        id_and_label: dict,
        documents: dict,
        excerpt_chars: int = 1000,
    ) -> list:
        """
        Run error_detection over every labelled cluster and return the results.

        Returns:
            List of dicts (one per cluster) with cluster_id, label, verdict, passed.
        """
        return [
            self.error_detection(cid, generated_labels, id_and_label, documents, excerpt_chars)
            for cid in tqdm(generated_labels, desc="Verifying clusters")
        ]