import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from itertools import combinations

# === CONFIG ===
result_path = "./auto-labeled/output/llama7b"
N_SAMPLES = 500  # For speed - increase for better viz
DIMS = 100       # First 100 dims for t-SNE speed

print("🔍 Loading LLaMA embeddings...")

# === LOAD EMBEDDINGS ===
right_last = json.load(open(f"{result_path}/last_token_mean_train.json"))
right_mean = json.load(open(f"{result_path}/last_mean_train.json"))
harm_last = json.load(open(f"{result_path}/last_token_mean_harmfull_train.json"))
harm_mean = json.load(open(f"{result_path}/last_mean_harmfull_train.json"))

print(f"Loaded: {len(right_last)} benign, {len(harm_last)} harmful")

# === EXTRACT + LIMIT SAMPLES ===
right_last_hd = np.array([item["right"][:DIMS] for item in right_last[:N_SAMPLES]])
right_mean_hd = np.array([item["right"][:DIMS] for item in right_mean[:N_SAMPLES]])
harm_last_hd = np.array([item["harmfull"][:DIMS] for item in harm_last[:N_SAMPLES]])
harm_mean_hd = np.array([item["harmfull"][:DIMS] for item in harm_mean[:N_SAMPLES]])

print(f"Shapes: right_last={right_last_hd.shape}, harm_last={harm_last_hd.shape}")

# === t-SNE VISUALIZATION ===
fig, axes = plt.subplots(1, 2, figsize=(16, 12))

# 1. Last Token Mean (your main training feature!)
axes[0].scatter(TSNE(n_components=2, random_state=42).fit_transform(right_last_hd)[:,0], 
                  TSNE(n_components=2, random_state=42).fit_transform(right_last_hd)[:,1], 
                  c='blue', alpha=0.6, s=20, label='Benign')
axes[0].scatter(TSNE(n_components=2, random_state=42).fit_transform(harm_last_hd)[:,0], 
                  TSNE(n_components=2, random_state=42).fit_transform(harm_last_hd)[:,1], 
                  c='red', alpha=0.6, s=20, label='Harmful')
axes[0].set_title('🧠 Last Token Mean (Main Training Feature)', fontsize=14)
axes[0].legend()

# 2. Last Mean Pooling
axes[1].scatter(TSNE(n_components=2, random_state=42).fit_transform(right_mean_hd)[:,0], 
                  TSNE(n_components=2, random_state=42).fit_transform(right_mean_hd)[:,1], 
                  c='blue', alpha=0.6, s=20, label='Benign')
axes[1].scatter(TSNE(n_components=2, random_state=42).fit_transform(harm_mean_hd)[:,0], 
                  TSNE(n_components=2, random_state=42).fit_transform(harm_mean_hd)[:,1], 
                  c='red', alpha=0.6, s=20, label='Harmful')
axes[1].set_title('📊 Last Mean Pooling', fontsize=14)
axes[1].legend()

plt.suptitle('LLaMA Hidden States: Benign (Blue) vs Harmful (Red)\nPerfect Separation = Great Classifier!', fontsize=16)
plt.tight_layout()
plt.savefig(f"{result_path}/clusters_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# === FIXED DISTANCE ANALYSIS ===
print("\n🔢 DISTANCE ANALYSIS (Higher cross-class = BETTER!)")

for name, right_vecs, harm_vecs in [
    ("Last Token Mean", right_last_hd, harm_last_hd),
    ("Mean Pooling", right_mean_hd, harm_mean_hd)
]:
    n = right_vecs.shape[0]  # Number of samples
    
    # Compute distance matrices
    bb_dists = euclidean_distances(right_vecs, right_vecs)
    hh_dists = euclidean_distances(harm_vecs, harm_vecs)
    bh_dists = euclidean_distances(right_vecs, harm_vecs)
    
    # FIXED: np.triu_indices(n) not np.triu_indices_from()
    bb_mean = bb_dists[np.triu_indices(n, k=1)].mean()
    hh_mean = hh_dists[np.triu_indices(n, k=1)].mean()
    bh_mean = bh_dists.mean()
    
    within_avg = (bb_mean + hh_mean) / 2
    separation_ratio = bh_mean / within_avg
    
    status = "✅ EXCELLENT" if separation_ratio > 1.05 else "⚠️ OK" if separation_ratio > 1.0 else "❌ POOR"
    
    print(f"\n{name}:")
    print(f"  Benign-Benign:     {bb_mean:.3f}")
    print(f"  Harmful-Harmful:   {hh_mean:.3f}")
    print(f"  Benign-Harmful:    {bh_mean:.3f}")
    print(f"  Within-class avg:  {within_avg:.3f}")
    print(f"  🎯 Separation:     {separation_ratio:.3f}x {status}")


print(f"\n✅ Saved: {result_path}/clusters_comparison.png")
