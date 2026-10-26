
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


MANIFEST_FILE = Path("clean_captures.csv")
OUTPUT_FILE = Path("embeddings.npy")
DOMAINS_FILE = Path("embedding_domains.csv")

MODEL_NAME = "openai/clip-vit-base-patch32"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("Loading pretrained CLIP model...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    if not MANIFEST_FILE.exists():
        print(f"Error: {MANIFEST_FILE} was not found.")
        return

    manifest = pd.read_csv(MANIFEST_FILE)
    records = []

    print(f"Records to process: {len(manifest)}")

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Generating embeddings",
    ):
        domain = row.get("domain", "")
        path = str(row.get("screenshot_path", ""))

        if not path or path == "nan":
            print(f"Skipping {domain}: no screenshot path.")
            continue

        # Support Windows paths stored in the CSV.
        image_path = Path(path)

        if not image_path.exists():
            image_path = Path(path.replace("\\", "/"))

        if not image_path.exists():
            print(f"Skipping {domain}: screenshot file not found.")
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                inputs = processor(
                    images=image,
                    return_tensors="pt",
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():
                    # Extract visual features from the CLIP vision encoder.
                    vision_outputs = model.vision_model(**inputs)

                    pooled_features = vision_outputs.pooler_output

                    features = model.visual_projection(
                        pooled_features
                    )

                    # Normalize vectors for cosine similarity.
                    features = torch.nn.functional.normalize(
                        features,
                        p=2,
                        dim=-1,
                    )

                records.append(
                    {
                        "domain": domain,
                        "embedding": features.cpu().numpy()[0],
                    }
                )

        except Exception as exc:
            print(f"Skipping {domain}: {exc}")

    if not records:
        print("No embeddings were created.")
        print("Check the screenshot paths and input CSV.")
        return

    # Preserve the exact same order for vectors and domain names.
    embeddings = np.stack(
        [record["embedding"] for record in records]
    ).astype(np.float32)

    domains = pd.DataFrame(
        {
            "domain": [
                record["domain"] for record in records
            ]
        }
    )

    np.save(OUTPUT_FILE, embeddings)
    domains.to_csv(DOMAINS_FILE, index=False)

    print("\nEmbedding generation complete!")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Successfully processed: {len(records)} screenshots")
    print(f"Saved embeddings to: {OUTPUT_FILE}")
    print(f"Saved domain order to: {DOMAINS_FILE}")


if __name__ == "__main__":
    main()
