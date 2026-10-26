
import hashlib
from pathlib import Path

import pandas as pd


pairs_file = Path("similar_website_pairs.csv")
screenshots_dir = Path("screenshots")

pairs = pd.read_csv(pairs_file).head(20)
results = []

for _, pair in pairs.iterrows():
    domain_1 = str(pair["domain_1"])
    domain_2 = str(pair["domain_2"])

    image_1 = screenshots_dir / f"{domain_1}.png"
    image_2 = screenshots_dir / f"{domain_2}.png"

    if image_1.exists() and image_2.exists():
        hash_1 = hashlib.sha256(image_1.read_bytes()).hexdigest()
        hash_2 = hashlib.sha256(image_2.read_bytes()).hexdigest()
        identical = hash_1 == hash_2
    else:
        identical = None

    results.append({
        "domain_1": domain_1,
        "domain_2": domain_2,
        "similarity": pair["similarity"],
        "identical_images": identical,
    })

result_df = pd.DataFrame(results)
result_df.to_csv("top_pair_hash_check.csv", index=False)

print(result_df.to_string(index=False))
print("\nSaved results to top_pair_hash_check.csv")
