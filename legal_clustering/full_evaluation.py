# legal_clustering/evaluation.py
import re
from collections import Counter
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
)


# Known CUAD contract types, ordered longest-first so multi-word types
# match before their substrings (e.g. "ServiceAgreement" before "Agreement").
# legal_clustering/evaluation.py

CUAD_CONTRACT_TYPES = [
    # Multi-word types first so they win over shorter substrings.
    # Format: canonical name -> match patterns (all lowercase, no spaces/punct)
    ("StrategicAlliance",        ["strategicalliance"]),
    ("JointVenture",             ["jointventure"]),
    ("CoBranding",               ["cobranding", "co-branding"]),
    ("IntellectualProperty",     ["intellectualproperty"]),
    ("NonCompete",               ["noncompete", "non-compete"]),
    ("JointFiling",              ["jointfiling"]),
    ("Remarketing",              ["remarketing"]),
    ("Distributor",              ["distributor", "distribution"]),
    ("License",                  ["license", "licensing", "licence"]),
    ("Hosting",                  ["hosting"]),
    ("Endorsement",              ["endorsement"]),
    ("Franchise",                ["franchise"]),
    ("Sponsorship",              ["sponsorship"]),
    ("Supply",                   ["supply"]),
    ("Consulting",               ["consulting"]),
    ("Promotion",                ["promotion"]),
    ("Affiliate",                ["affiliate"]),
    ("Marketing",                ["marketing"]),
    ("Maintenance",              ["maintenance"]),
    ("Manufacturing",            ["manufacturing"]),
    ("Outsourcing",              ["outsourcing"]),
    ("Reseller",                 ["reseller"]),
    ("Service",                  ["service"]),
    ("Transportation",           ["transportation"]),
    ("Agency",                   ["agency"]),
    ("Operating",                ["operating"]),
    ("Development",              ["development"]),
]


def extract_contract_type(title: str) -> str:
    """
    Extract the contract type from a CUAD filename.

    CUAD titles vary wildly in format — some use underscores, some spaces,
    some have dates and exhibit numbers embedded. The contract type usually
    appears at the end. We normalise to lowercase alphanumeric only and
    match against known type substrings in priority order.
    """
    cleaned = re.sub(r"[^a-z0-9]", "", title.lower())

    for canonical, patterns in CUAD_CONTRACT_TYPES:
        for pattern in patterns:
            normalised_pattern = re.sub(r"[^a-z0-9]", "", pattern)
            if normalised_pattern in cleaned:
                return canonical
    return "Unknown"


def evaluate_clustering(
    name: str,
    embeddings,
    pred_labels: list,
    true_labels: list,
    metric: str = "cosine",
) -> dict:
    """
    Compute internal and external clustering metrics and print a summary.

    Internal metrics (no ground truth needed) measure cluster shape:
        - Silhouette: how tight and well-separated clusters are.
        - Davies-Bouldin: lower is better; ratio of within- to between-
          cluster scatter.

    External metrics compare predictions against known categories:
        - ARI (Adjusted Rand Index): agreement adjusted for chance.
          0 = random, 1 = perfect.
        - AMI (Adjusted Mutual Information): chance-adjusted NMI.
        - Homogeneity: do clusters contain only one true class?
        - Completeness: are all docs of a class in one cluster?
        - V-measure: harmonic mean of homogeneity and completeness.

    Args:
        name: Display name for this pipeline (e.g. "TF-IDF").
        embeddings: Vector space the clustering was performed in.
            Required for internal metrics.
        pred_labels: Cluster assignments output by the pipeline.
        true_labels: Ground-truth labels (e.g. contract types).
        metric: Distance metric used by silhouette_score.

    Returns:
        Dict of all computed metrics, suitable for tabulation.
    """
    sizes = Counter(pred_labels)

    sil = silhouette_score(embeddings, pred_labels, metric=metric)
    db = davies_bouldin_score(embeddings, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    hom, comp, vm = homogeneity_completeness_v_measure(true_labels, pred_labels)

    results = {
        "name": name,
        "n_clusters": len(sizes),
        "n_singletons": sum(1 for c in sizes.values() if c == 1),
        "largest": max(sizes.values()),
        "smallest": min(sizes.values()),
        "silhouette": sil,
        "davies_bouldin": db,
        "ari": ari,
        "ami": ami,
        "homogeneity": hom,
        "completeness": comp,
        "v_measure": vm,
    }

    print(f"\n=== {name} ===")
    print(f"  Clusters:           {results['n_clusters']}")
    print(f"  Singletons:         {results['n_singletons']}")
    print(f"  Largest / Smallest: {results['largest']} / {results['smallest']}")
    print(f"  Silhouette:         {sil:.4f}   (higher better)")
    print(f"  Davies-Bouldin:     {db:.4f}   (lower better)")
    print(f"  ARI:                {ari:.4f}   (chance-adjusted, 0=random, 1=perfect)")
    print(f"  AMI:                {ami:.4f}   (chance-adjusted MI)")
    print(f"  Homogeneity:        {hom:.4f}   (pure clusters?)")
    print(f"  Completeness:       {comp:.4f}   (types intact?)")
    print(f"  V-measure:          {vm:.4f}   (harmonic mean)")

    return results


def print_comparison(results_list: list) -> None:
    """Print a side-by-side comparison table of multiple pipeline results."""
    if not results_list:
        return

    metrics = ["n_clusters", "silhouette", "davies_bouldin", "ari", "ami",
               "homogeneity", "completeness", "v_measure"]

    print("\n" + "=" * 70)
    print(f"{'Metric':<18}" + "".join(f"{r['name']:>16}" for r in results_list))
    print("=" * 70)
    for m in metrics:
        row = f"{m:<18}"
        for r in results_list:
            val = r[m]
            row += f"{val:>16.4f}" if isinstance(val, float) else f"{val:>16}"
        print(row)
    print("=" * 70)