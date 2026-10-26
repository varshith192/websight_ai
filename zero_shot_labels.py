
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


INPUT_FILE = Path("clean_captures.csv")
OUTPUT_FILE = Path("zero_shot_predictions.csv")
SUMMARY_FILE = Path("zero_shot_summary.csv")

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16

LABEL_PROMPTS = {
    "anti_bot": [
        "A website asking the visitor to verify they are human.",
        "A webpage showing a CAPTCHA or automated traffic security challenge.",
    ],
    "cookie_banner": [
        "A website with a cookie consent banner and accept or reject buttons.",
        "A webpage displaying cookie preferences at the bottom of the screen.",
    ],
    "login_page": [
        "A website login page with username and password fields.",
        "A sign-in form for an online account.",
    ],
    "domain_for_sale": [
        "A parked domain webpage saying this domain is for sale.",
        "A website advertising a domain name for purchase.",
    ],
    "blank_or_failed": [
        "A completely blank white or black webpage with no visible content.",
        "A failed website screenshot with an empty page or loading failure.",
    ],
    "coming_soon": [
        "A website displaying a coming soon or under construction message.",
        "A placeholder homepage announcing that the website is launching soon.",
    ],
    "error_page": [
        "A browser webpage showing an error message or access denied page.",
        "A website displaying a server error or unavailable page.",
    ],
    "ecommerce": [
        "An online shopping homepage with product images and shopping navigation.",
        "An ecommerce store showing products, prices, or shopping categories.",
    ],
    "news_media": [
        "A news website homepage with headlines, article lists, and news images.",
        "An online newspaper or media homepage.",
    ],
    "sports": [
        "A sports website showing live scores, teams, fixtures, or match results.",
        "A sports results homepage with scoreboards and sports navigation.",
    ],
    "forum_community": [
        "An online discussion forum with posts and user conversations.",
        "A community website with discussion threads and comments.",
    ],
    "government_service": [
        "An official government or public service website homepage.",
        "A public administration website with official service links and information.",
    ],
    "other_homepage": [
        "A normal website homepage with its own content and navigation.",
        "A general website landing page that does not fit the other categories.",
    ],
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} was not found. Run this script "
            "from your websight_ai project folder."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading CLIP model...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    labels = list(LABEL_PROMPTS.keys())
    prompts_per_label = len(LABEL_PROMPTS[labels[0]])

    if any(
        len(LABEL_PROMPTS[label]) != prompts_per_label
        for label in labels
    ):
        raise ValueError("Every label must have the same number of prompts.")

    prompts = [
        prompt
        for label in labels
        for prompt in LABEL_PROMPTS[label]
    ]

    # Encode text descriptions once.
    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    text_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
    }

    with torch.no_grad():
        # Extract pooled text features and project into CLIP's
        # shared image-text embedding space.
        text_outputs = model.text_model(**text_inputs)
        pooled_text = text_outputs.pooler_output
        text_features = model.text_projection(pooled_text)
        text_features = F.normalize(
            text_features, p=2, dim=-1
        )

        # Average the text embeddings for each category.
        text_features = text_features.reshape(
            len(labels), prompts_per_label, -1
        ).mean(dim=1)

        text_features = F.normalize(
            text_features, p=2, dim=-1
        )

    data = pd.read_csv(INPUT_FILE)

    required_columns = {"domain", "screenshot_path"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{INPUT_FILE} is missing columns: {sorted(missing_columns)}"
        )

    results = []
    missing_count = 0

    print(f"Screenshots listed: {len(data)}")

    for start in tqdm(
        range(0, len(data), BATCH_SIZE),
        desc="Classifying screenshots",
    ):
        batch = data.iloc[start:start + BATCH_SIZE]
        images = []
        valid_rows = []

        for _, row in batch.iterrows():
            raw_path = row.get("screenshot_path", "")

            if pd.isna(raw_path) or not str(raw_path).strip():
                missing_count += 1
                continue

            image_path = Path(str(raw_path))

            if not image_path.exists():
                image_path = Path(str(raw_path).replace("\\", "/"))

            if not image_path.exists():
                missing_count += 1
                continue

            try:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
                valid_rows.append(row)
            except Exception:
                missing_count += 1

        if not images:
            continue

        image_inputs = processor(
            images=images,
            return_tensors="pt",
        )
        image_inputs = {
            key: value.to(device)
            for key, value in image_inputs.items()
        }

        with torch.no_grad():
            image_features = model.vision_model(**image_inputs)
            pooled_images = image_features.pooler_output
            image_features = model.visual_projection(pooled_images)

            image_features = F.normalize(
                image_features, p=2, dim=-1
            )

            # Cosine similarities because both feature sets are normalized.
            similarities = image_features @ text_features.T

        similarity_rows = similarities.cpu().tolist()

        for row, scores in zip(valid_rows, similarity_rows):
            ranked_indices = sorted(
                range(len(labels)),
                key=lambda index: scores[index],
                reverse=True,
            )

            best_index = ranked_indices[0]
            second_index = ranked_indices[1]

            result = {
                "domain": row["domain"],
                "screenshot_path": row["screenshot_path"],
                "predicted_label": labels[best_index],
                "top_similarity": round(scores[best_index], 4),
                "second_label": labels[second_index],
                "second_similarity": round(
                    scores[second_index], 4
                ),
                "score_margin": round(
                    scores[best_index] - scores[second_index], 4
                ),
            }

            for index, label in enumerate(labels):
                result[f"score_{label}"] = round(scores[index], 4)

            results.append(result)

    if not results:
        raise RuntimeError(
            "No screenshots were classified. Check screenshot paths "
            "in clean_captures.csv."
        )

    predictions = pd.DataFrame(results)
    predictions.to_csv(OUTPUT_FILE, index=False)

    summary = (
        predictions["predicted_label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )

    summary["percentage"] = (
        summary["count"] / len(predictions) * 100
    ).round(2)

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nZero-shot classification complete!")
    print(f"Classified screenshots: {len(predictions)}")
    print(f"Missing or unreadable screenshots: {missing_count}")
    print(f"Saved predictions to: {OUTPUT_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")
    print("\nPredicted label counts:")
    print(summary.to_string(index=False))

    print(
        "\nIMPORTANT: These are model predictions, not verified labels. "
        "Similarity scores are not calibrated probabilities. "
        "Manually inspect examples before reporting category counts."
    )


if __name__ == "__main__":
    main()
