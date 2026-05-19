from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import random

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
        seed: int,
        batch_size: int,
        min_cluster_size: int,
        excerpt_chars: int
    ):
        """
        Args:
            llm_model: Hugging Face model identifier (e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0").
            max_tokens: Maximum new tokens to generate per prompt.
            token_price: Cost per token (used by count_price_tokens for budgeting).
            n_llm_samples: Number of documents to sample from each cluster for prompts.
            prompt_type_of_doc: Human-readable document type used in prompts
                (e.g. "legal contracts", "news articles").
            seed: Random seed used for sampling documents from clusters.
            batch_size: Number of prompts to send to the model at once. Larger is
                faster on GPU but uses more memory. Set to 1 on CPU-only machines.
        """
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.token_price = token_price
        self.n_llm_samples = n_llm_samples
        self.prompt_type_of_doc = prompt_type_of_doc
        self.seed = seed
        self.batch_size = batch_size
        self.min_cluster_size = min_cluster_size
        self.excerpt_chars = excerpt_chars

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

        Configures the tokenizer for batched generation:
            - Sets pad_token to eos_token if missing (required by many decoder-only
              models like TinyLlama and Llama, which don't ship with a pad token).
            - Sets padding_side="left" because decoder-only models generate from
              the right end of the input; right-padding would cause them to
              continue from pad tokens and produce garbage.

        Side effects:
            Sets self._tokenizer to the loaded AutoTokenizer instance.
            Sets self._hf_llm to the configured text-generation pipeline.

        Note:
            This method is called from __init__, so constructing an LLMEvaluation
            instance will download the model on first use and load it into memory.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model)

        # Decoder-only models often lack a pad token; reuse EOS for padding.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Left-padding is required for batched generation with decoder-only models.
        self._tokenizer.padding_side = "left"

        self._hf_llm = pipeline(
            task="text-generation",
            model=self.llm_model,
            tokenizer=self._tokenizer,
            device_map="auto",
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

    def count_price_tokens(self, prompt: str) -> dict:
        """Return the token count and cost of a single prompt."""
        num_tokens = len(self._tokenizer.encode(prompt))
        return {
            "number_of_tokens": num_tokens,
            "price": num_tokens * self.token_price,
        }

    def _build_label_prompt(self, sample_ids: list, documents: dict) -> str:
        """Construct a prompt asking the LLM to label a cluster."""
        excerpts = [
            f"--- {doc_id} ---\n{documents.get(doc_id, '')[:self.excerpt_chars].strip()}"
            for doc_id in sample_ids
        ]
        return (
            f"These {self.prompt_type_of_doc} were grouped together by a "
            "clustering algorithm:\n\n"
            + "\n\n".join(excerpts)
            + f"\n\nRespond with only a short 3-5 word label describing "
            f"what {self.prompt_type_of_doc} these are. No explanation."
        )

    def _build_verification_prompt(
        self, cluster_label: str, sample_ids: list, documents: dict) -> str:
        """Construct a prompt asking the LLM to verify a cluster's label."""
        excerpts = [
            f"--- {doc_id} ---\n{documents.get(doc_id, '')[:self.excerpt_chars].strip()}"
            for doc_id in sample_ids
        ]
        return (
            f"A clustering algorithm grouped these {self.prompt_type_of_doc} together "
            f'and labelled the cluster: "{cluster_label}".\n\n'
            + "\n\n".join(excerpts)
            + f"\n\nDo these {self.prompt_type_of_doc} all belong to the same type? "
            "Reply with YES or NO on the first line, then a one sentence explanation."
        )

    def llm_label(
        self,
        id_and_label: dict,
        documents: dict) -> dict:
        """
        Generate a short descriptive label for each cluster using the LLM.

        Prompts for all eligible clusters are built up front, then sent to the
        pipeline in batches (size controlled by self.batch_size) for efficient
        GPU utilisation.

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
        # Group doc_ids by cluster, normalising numpy ints to Python ints.
        clusters = {}
        for doc_id, label in id_and_label.items():
            clusters.setdefault(int(label), []).append(doc_id)

        # Skip clusters too small to label meaningfully.
        clusters = {cid: ids for cid, ids in clusters.items() if len(ids) >= self.min_cluster_size}

        # Single RNG, seeded once — no global state pollution, samples vary per cluster.
        rng = random.Random(self.seed)

        # Build all prompts up front so they can be batched.
        cluster_ids = []
        prompts = []
        for cluster_id, doc_ids in clusters.items():
            sample_ids = rng.sample(doc_ids, min(self.n_llm_samples, len(doc_ids)))
            cluster_ids.append(cluster_id)
            prompts.append(self._build_label_prompt(sample_ids, documents, self.excerpt_chars))

        # Batched generation. return_full_text=False yields only the continuation,
        # avoiding fragile prompt-stripping logic.
        outputs = list(tqdm(
            self._hf_llm(prompts, batch_size=self.batch_size, return_full_text=False),
            total=len(prompts),
            desc="Labelling clusters",
        ))

        # Zip outputs back to cluster IDs and clean up the labels.
        generated_cluster_labels = {}
        for cluster_id, out in zip(cluster_ids, outputs):
            text = out[0]["generated_text"].strip()
            # Hard cap on label length in case the model ignores the instruction.
            label = " ".join(text.split()[:8])
            generated_cluster_labels[cluster_id] = label

        return generated_cluster_labels

    def error_detection(
        self,
        cluster_id: int,
        generated_labels: dict,
        id_and_label: dict,
        documents: dict) -> dict:
        """
        Verify a single cluster: ask the LLM whether its documents genuinely
        belong together under the generated label.

        For evaluating many clusters efficiently, use evaluate_all instead —
        it batches all verification prompts together.

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

        rng = random.Random(self.seed)
        sample_ids = rng.sample(doc_ids, min(self.n_llm_samples, len(doc_ids)))

        prompt = self._build_verification_prompt(cluster_label, sample_ids, documents, self.excerpt_chars)
        raw_verdict = self._hf_llm(prompt, return_full_text=False)[0]["generated_text"].strip()

        first_token = raw_verdict.split()[0].upper().strip(".,:;") if raw_verdict else ""
        return {
            "cluster_id": cluster_id,
            "label": cluster_label,
            "verdict": raw_verdict,
            "passed": first_token.startswith("YES"),
        }

    def evaluate_all(
        self,
        generated_labels: dict,
        id_and_label: dict,
        documents: dict) -> list:
        """
        Run cluster verification over every labelled cluster, batching prompts
        through the LLM for speed.

        Args:
            generated_labels: Mapping of cluster_id -> label (output of llm_label).
            id_and_label: Mapping of doc_id -> cluster label.
            documents: Mapping of doc_id -> raw document text.
            excerpt_chars: Characters of each document to include in the prompt.

        Returns:
            List of dicts (one per cluster), each with cluster_id, label,
            raw verdict text, and a parsed bool `passed`.
        """
        rng = random.Random(self.seed)

        # Build all verification prompts up front so they can be batched.
        cluster_ids = []
        cluster_labels = []
        prompts = []
        for cluster_id, cluster_label in generated_labels.items():
            doc_ids = [
                doc_id for doc_id, label in id_and_label.items()
                if int(label) == cluster_id
            ]
            sample_ids = rng.sample(doc_ids, min(self.n_llm_samples, len(doc_ids)))

            cluster_ids.append(cluster_id)
            cluster_labels.append(cluster_label)
            prompts.append(self._build_verification_prompt(
                cluster_label, sample_ids, documents, self.excerpt_chars
            ))

        # Batched generation across all clusters.
        outputs = list(tqdm(
            self._hf_llm(prompts, batch_size=self.batch_size, return_full_text=False),
            total=len(prompts),
            desc="Verifying clusters",
        ))

        results = []
        for cluster_id, cluster_label, out in zip(cluster_ids, cluster_labels, outputs):
            raw_verdict = out[0]["generated_text"].strip()
            first_token = raw_verdict.split()[0].upper().strip(".,:;") if raw_verdict else ""
            results.append({
                "cluster_id": cluster_id,
                "label": cluster_label,
                "verdict": raw_verdict,
                "passed": first_token.startswith("YES"),
            })

        return results