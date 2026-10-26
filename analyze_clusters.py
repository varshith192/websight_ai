
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE = Path("embeddings.npy")
DOMAINS_FILE = Path("embedding_domains.csv")


def main():
    print("Loading CLIP embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE)
    domains = pd.read_csv(DOMAINS_FILE)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Domains loaded: {len(domains)}")

    # Try several cluster counts and compare their silhouette scores.
    results = []

    for k in range(2, min(9, len(embeddings))):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(embeddings)

        score = silhouette_score(
            embeddings,
            labels,
            metric="cosine",
        )

        results.append({"clusters": k, "silhouette_score": score})
        print(f"k={k}: silhouette score={score:.4f}")

    scores = pd.DataFrame(results)
    scores.to_csv("cluster_scores.csv", index=False)

    best_k = int(
        scores.loc[scores["silhouette_score"].idxmax(), "clusters"]
    )
    print(f"\nBest tested cluster count: {best_k}")

    # Fit the selected model.
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    domains["cluster"] = model.fit_predict(embeddings)
    domains.to_csv("cluster_assignments.csv", index=False)

    # Reduce embeddings to two dimensions for visualization.
    points = PCA(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=domains["cluster"],
        cmap="tab10",
        s=35,
        alpha=0.8,
    )
    plt.title("Website Visual Similarity Clusters (CLIP + PCA)")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig("cluster_visualization.png", dpi=200)
    plt.close()

    # Find the most visually similar pairs.
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, -1)

    pairs = []
    for i in range(len(domains)):
        j = int(np.argmax(similarities[i]))
        if i < j:
            pairs.append({
                "domain_1": domains.iloc[i]["domain"],
                "domain_2": domains.iloc[j]["domain"],
                "similarity": float(similarities[i, j]),
            })

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values(
        "similarity", ascending=False
    )
    pairs_df.to_csv("similar_website_pairs.csv", index=False)

    print("\nCluster sizes:")
    print(domains["cluster"].value_counts().sort_index())

    print("\nTop visually similar website pairs:")
    print(pairs_df.head(10).to_string(index=False))

    print("\nSaved files:")
    print("- cluster_scores.csv")
    print("- cluster_assignments.csv")
    print("- cluster_visualization.png")
    print("- similar_website_pairs.csv")


if __name__ == "__main__":
    main()
