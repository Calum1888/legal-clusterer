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
        excerpt_chars: int,
        hf_llm=None,
        tokenizer=None,
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

        if hf_llm is not None and tokenizer is not None:
            self._tokenizer = tokenizer
            self._hf_llm = hf_llm
        else:
            self._build_pipeline()

    def _build_pipeline(self):
        """
        Initialise the tokenizer and Hugging Face text-generation pipeline.

        Greedy decoding (do_sample=False) is used so label generation and
        verification are reproducible across runs without seed management.

        Tokenizer is configured for batched generation: pad_token falls back to
        eos_token if missing, and padding_side="left" because decoder-only models
        generate from the right end of the input (right-padding would make them
        continue from pad tokens and produce garbage).
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model)

        # decoder-only models often lack a pad token; reuse EOS for padding.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # left-padding is required for batched generation with decoder-only models.
        self._tokenizer.padding_side = "left"

        self._hf_llm = pipeline(
            task="text-generation",
            model=self.llm_model,
            tokenizer=self._tokenizer,
            device_map="auto",
            max_new_tokens=self.max_tokens,
            do_sample=False,
        )

    def _format_chat(self, user_content: str) -> str:
        """
        Wrap an instruction in the model's chat template so a *chat* model runs
        in instruct mode and actually follows the instruction. Falls back to the
        raw text if the tokenizer has no chat template (e.g. a base model).
        """
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_content

    def count_price_tokens(self, prompt: str) -> dict:
        """
        Return the token count and cost of a single prompt.

        Will probably be removed in future updates.
        """
        num_tokens = len(self._tokenizer.encode(prompt))
        return {
            "number_of_tokens": num_tokens,
            "price": num_tokens * self.token_price,
        }

    def _excerpts(self, sample_ids: list, documents: dict) -> str:
        """
        Build the content block for a prompt.

        Uses neutral "Document N" headers and the document *content only* — the
        filename / doc_id is deliberately NOT included, because a small model
        will otherwise just echo the filename back as the label.
        """
        blocks = [
            f"Document {i}:\n{documents.get(doc_id, '')[:self.excerpt_chars].strip()}"
            for i, doc_id in enumerate(sample_ids, 1)
        ]
        return "\n\n".join(blocks)

    def _build_label_prompt(self, sample_ids: list, documents: dict) -> str:
        """
        Construct a prompt asking the LLM to name the common theme of a cluster.

        Content only (no filenames), with explicit rules and a one-shot example
        so a small model returns a short category rather than a copied title.
        """
        instruction = (
            f"Below are excerpts from several {self.prompt_type_of_doc} that a "
            "clustering algorithm placed in the same group. Identify the common "
            "theme and give a short category name for the group.\n\n"
            "Rules:\n"
            "- 2 to 4 words.\n"
            "- Describe the shared topic, not any single document.\n"
            "- Do NOT use file names, document titles, or document numbers.\n"
            "- Reply with the category name only, nothing else.\n\n"
            "Example reply: Employment Contracts\n\n"
            f"{self._excerpts(sample_ids, documents)}\n\n"
            "Category name:"
        )
        return self._format_chat(instruction)

    def _build_verification_prompt(
        self, cluster_label: str, sample_ids: list, documents: dict) -> str:
        """
        Construct a prompt asking whether the documents fit the candidate label.

        Phrased to reduce the YES-priming bias (it does not assert that an
        algorithm already decided they belong together) and to force a clean
        one-word verdict that can be parsed reliably.
        """
        instruction = (
            f"Here are excerpts from several {self.prompt_type_of_doc}:\n\n"
            f"{self._excerpts(sample_ids, documents)}\n\n"
            f'Do all of these documents fit the category "{cluster_label}"?\n'
            "Answer with exactly one word on the first line: YES or NO.\n"
            "Then, on a new line, give a one-sentence reason."
        )
        return self._format_chat(instruction)

    @staticmethod
    def _parse_verdict(raw_verdict: str) -> bool:
        """
        Parse a verification reply into a pass/fail bool. Looks at the first
        word of the first non-empty line and checks whether it is YES.
        """
        for line in (raw_verdict or "").splitlines():
            line = line.strip()
            if not line:
                continue
            first = line.upper().replace(".", " ").replace(",", " ").split()
            return bool(first) and first[0].startswith("YES")
        return False

    @staticmethod
    def _clean_label(text: str) -> str:
        """
        Tidy a generated label: first line only, strip quotes and any echoed
        "Category name:" prefix, and cap length as a safety net.
        """
        text = (text or "").strip()
        if text:
            text = text.splitlines()[0].strip().strip('"').strip("'")
        for prefix in ("Category name:", "Category:", "Label:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip().strip('"').strip("'")
        return " ".join(text.split()[:8])

    def _batched_generate(self, prompts: list, desc: str, progress=None) -> list:
        '''
        Batches prompts together for the LLM to run over in parallel.

        Shows a progress bar: gr.Progress in the browser when `progress` is
        supplied (Gradio), otherwise a tqdm bar in the terminal.

        Args:
            prompts (list): The prompts generated for cluster labels and verification.
            desc (str): Label shown on the progress bar.
            progress: Optional gradio.Progress; if given, the bar renders in the UI.

        Return:
            list: The outputs from the LLM.
        '''
        outputs = []
        # gr.Progress exposes .tqdm with the same interface as tqdm, so we just
        # pick which iterator wrapper to use. Both render a live, moving bar.
        starts = range(0, len(prompts), self.batch_size)
        bar = progress.tqdm(starts, desc=desc) if progress is not None else tqdm(starts, desc=desc)
        for i in bar:
            batch = prompts[i : i + self.batch_size]
            batch_outputs = self._hf_llm(
                batch, batch_size=self.batch_size, return_full_text=False,
            )
            outputs.extend(batch_outputs)
        return outputs

    def llm_label(
        self,
        id_and_label: dict,
        documents: dict,
        progress=None) -> dict:
        """
        Generate a short descriptive label for each cluster using the LLM.

        Prompts for all eligible clusters are built up front, then sent to the
        pipeline in batches (size controlled by self.batch_size).

        Args:
            id_and_label: Mapping of doc_id -> cluster label (output of clusterer.fit).
            documents: Mapping of doc_id -> raw document text.

        Returns:
            Mapping of cluster_id -> generated label string.
        """
        # group doc_ids by cluster, normalising numpy ints to Python ints.
        clusters = {}
        for doc_id, label in id_and_label.items():
            clusters.setdefault(int(label), []).append(doc_id)

        # skip clusters too small to label meaningfully.
        clusters = {cid: ids for cid, ids in clusters.items() if len(ids) >= self.min_cluster_size}

        # single RNG, seeded once — no global state pollution, samples vary per cluster.
        rng = random.Random(self.seed)

        # build all prompts up front so they can be batched.
        cluster_ids = []
        prompts = []
        for cluster_id, doc_ids in clusters.items():
            sample_ids = rng.sample(doc_ids, min(self.n_llm_samples, len(doc_ids)))
            cluster_ids.append(cluster_id)
            prompts.append(self._build_label_prompt(sample_ids, documents))

        # batched generation. return_full_text=False yields only the continuation.
        outputs = self._batched_generate(prompts, desc="Labelling clusters", progress=progress)

        # zip outputs back to cluster IDs and clean up the labels.
        generated_cluster_labels = {}
        for cluster_id, out in zip(cluster_ids, outputs):
            generated_cluster_labels[cluster_id] = self._clean_label(out[0]["generated_text"])

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

        For evaluating many clusters efficiently, use evaluate_all instead.

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

        prompt = self._build_verification_prompt(cluster_label, sample_ids, documents)
        raw_verdict = self._hf_llm(prompt, return_full_text=False)[0]["generated_text"].strip()

        return {
            "cluster_id": cluster_id,
            "label": cluster_label,
            "verdict": raw_verdict,
            "passed": self._parse_verdict(raw_verdict),
        }

    def evaluate_all(
        self,
        generated_labels: dict,
        id_and_label: dict,
        documents: dict,
        progress=None) -> list:
        """
        Run cluster verification over every labelled cluster, batching prompts
        through the LLM for speed.

        Returns:
            List of dicts (one per cluster), each with cluster_id, label,
            raw verdict text, and a parsed bool `passed`.
        """
        rng = random.Random(self.seed)

        # build all verification prompts up front so they can be batched.
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
                cluster_label, sample_ids, documents))

        # batched generation across all clusters.
        outputs = self._batched_generate(prompts, desc="Verifying clusters", progress=progress)

        results = []
        for cluster_id, cluster_label, out in zip(cluster_ids, cluster_labels, outputs):
            raw_verdict = out[0]["generated_text"].strip()
            results.append({
                "cluster_id": cluster_id,
                "label": cluster_label,
                "verdict": raw_verdict,
                "passed": self._parse_verdict(raw_verdict),
            })

        return results