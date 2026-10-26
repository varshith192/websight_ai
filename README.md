# WebSight AI: Visual Website Analysis at Scale

## 1. Overview

WebSight AI analyzes website screenshots using computer vision and pretrained image embeddings. The project processes a list of 5,000 domains, identifies visually similar websites, explores visual clusters and outliers, and applies zero-shot classification to characterize website screenshots.

The objective is to discover meaningful visual patterns rather than simply collect screenshots.

### Key capabilities

- Capture website homepage screenshots using Playwright and Chromium.
- Filter blank, repeated, and questionable screenshots.
- Generate visual embeddings using a pretrained CLIP model.
- Explore visual clusters using silhouette scores.
- Identify visually similar website pairs using cosine similarity.
- Classify screenshots into exploratory categories using zero-shot classification.
- Detect visual outlier candidates using nearest-neighbor similarity.
- Compare predicted website categories across popularity rank buckets.

## 2. Dataset and Capture Results

The input dataset, `domains.csv`, contains 5,000 domain records with metadata including:

- `tier`
- `rank_bucket`
- `domain`
- `crux_origin`

### Capture statistics

| Metric | Result |
|---|---:|
| Domains processed | 5,000 |
| Successful-status captures (`ok`) | 3,323 |
| Screenshots retained for embeddings | 2,791 |
| PNG files present | 4,315 |

### Capture status breakdown

| Status | Count |
|---|---:|
| `ok` | 3,323 |
| `http_error` | 992 |
| `dns_error` | 284 |
| `timeout` | 250 |
| `other` | 151 |
| **Total** | **5,000** |

A successful HTTP status does not guarantee that the screenshot contains the intended website content. Some pages may show anti-bot challenges, errors, blank content, or incomplete rendering.

Filtering excluded 532 records from the 3,323 successful-status captures, leaving 2,791 screenshots for embedding analysis. This filtering is provisional and may exclude legitimate pages with sparse or repeated designs.

## 3. Methodology

### 3.1 Screenshot collection

Playwright and Chromium were used to visit domains and capture homepage screenshots. Capture outcomes and associated metadata were recorded in `manifest.csv`.

The collection follows a homepage-only approach without crawling additional pages. Access restrictions and failed requests are treated as capture outcomes rather than bypassed.

### 3.2 Screenshot quality checks

Screenshot quality checks included basic visual checks for blank or uniform images and exact-image hashing to identify repeated captures.

The filtering process produced `clean_captures.csv`, containing 2,791 retained records for downstream analysis.

### 3.3 Visual embeddings

The pretrained model `openai/clip-vit-base-patch32` was used to generate 512-dimensional image embeddings.

The vectors were L2-normalized and saved in `embeddings.npy`. The corresponding domain order was saved in `embedding_domains.csv`.

### 3.4 Clustering

Candidate cluster counts from 2 through 8 were evaluated using silhouette scores. Four clusters achieved the highest score among the tested values.

### 3.5 Similarity analysis

Cosine similarity was used to rank screenshot pairs. High-similarity pairs were inspected to distinguish potentially related website designs from shared capture artifacts.

### 3.6 Zero-shot classification

CLIP image and text representations were used to assign each retained screenshot its highest-scoring label from 13 exploratory categories:

1. Ecommerce
2. Coming soon
3. Government service
4. News and media
5. Other homepage
6. Cookie banner
7. Login page
8. Sports
9. Anti-bot challenge
10. Error page
11. Domain for sale
12. Forum and community
13. Blank or failed capture

These predictions are not verified ground-truth labels. CLIP similarity scores are not calibrated probabilities.

### 3.7 Outlier analysis

Each embedding was compared with its nearest neighbors. Screenshots with low average cosine similarity to their five nearest neighbors were selected as visual outlier candidates.

A contact sheet of 15 candidates was generated for manual inspection.

## 4. Preliminary Findings

### Finding 1: Cluster separation is weak

The best tested silhouette score was approximately **0.0960 at k = 4**. This indicates substantial overlap between the clusters. They should be treated as exploratory visual groupings rather than definitive website categories.

### Finding 2: High similarity can reflect capture artifacts

Examples of high-similarity pairs include:

- `thomann.at` and `thomannmusic.ch`: approximately 0.9975.
- `baritoday.it` and `modenatoday.it`: approximately 0.9947.
- `bootbarn.com` and `anntaylor.com`: approximately 0.9985.

Manual inspection confirmed that the Thomann and local-news pairs showed real website content. The Boot Barn and Ann Taylor screenshots both showed the same anti-bot verification challenge.

This demonstrates that the model can match the captured challenge rather than the underlying websites.

### Finding 3: Predicted categories vary by rank bucket

Among retained screenshots, predicted ecommerce share increased from approximately 14.3% in the 1,000 rank bucket to 27.4% in the 1,000,000 rank bucket.

Predicted news/media share decreased from approximately 25.0% to 6.7%.

These are preliminary associations within the retained sample. They do not establish causation or actual website-category prevalence.

### Finding 4: Zero-shot predictions require validation

All 2,791 retained screenshots received a top prediction across the 13 categories. The largest predicted categories were ecommerce, coming soon, and government service.

The labels require manual validation before the percentages can be interpreted as reliable category prevalence estimates.

### Finding 5: Visual outliers show diverse designs

The reviewed outlier candidates included educational, government, sports-news, immigration, event, ecommerce, and specialized websites.

Low nearest-neighbor similarity alone does not explain why a website is an outlier. Unusual content, minimal layouts, and capture artifacts may all contribute.

## 5. Runtime

Observed CPU runtimes for the following stages were:

| Stage | Runtime |
|---|---:|
| CLIP embedding generation for 2,791 screenshots | Approximately 3 minutes 34 seconds |
| Zero-shot classification for 2,791 screenshots | Approximately 2 minutes 28 seconds |

These figures cover only the listed stages. Screenshot collection and other analysis stages are not included in these measurements.

Runtime can vary depending on hardware, network conditions, and the number of screenshots processed.

## 6. Setup and Installation

### Prerequisites

- Python 3.12 or a compatible Python version.
- Git, if cloning the repository.
- An internet connection for package installation and website capture.
- Sufficient disk space for screenshots, embeddings, and intermediate outputs.

### Step 1: Obtain the project

Clone the repository if it is hosted in Git:

```powershell
git clone <YOUR_REPOSITORY_URL>
cd websight_ai
```

Alternatively, extract the submitted ZIP file and open the project directory in VS Code.

Replace `<YOUR_REPOSITORY_URL>` with the actual repository URL if using Git.

### Step 2: Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks environment activation, use the appropriate execution policy permitted on your computer, or activate the environment through the VS Code Python interpreter selector.

### Step 3: Install dependencies

If the repository contains `requirements.txt`, install the listed dependencies:

```powershell
pip install -r requirements.txt
```

Install the Playwright Chromium browser:

```powershell
playwright install chromium
```

The project uses Python packages including Playwright, pandas, NumPy, Pillow, tqdm, scikit-learn, Matplotlib, Transformers, and PyTorch.

The dependency file should contain the tested package versions used for the submitted run.

## 7. Reproducing the Analysis

Run commands from the project root with the virtual environment activated.

### Step 1: Capture website screenshots

```powershell
python capture_screenshots.py
```

This produces the screenshot collection and capture manifest.

### Step 2: Prepare the retained screenshot dataset

```powershell
python create_clean_captures.py
```

This generates `clean_captures.csv` using the project's screenshot quality and duplicate-filtering rules.

### Step 3: Generate CLIP embeddings

```powershell
python generate_embeddings.py
```

This produces `embeddings.npy` and `embedding_domains.csv`.

### Step 4: Analyze clusters and visual similarity

```powershell
python analyze_clusters.py
```

This generates cluster scores, assignments, a visualization, and ranked similar website pairs.

### Step 5: Run zero-shot classification

```powershell
python zero_shot_labels.py
```

This generates per-screenshot predictions and a category summary.

### Step 6: Generate the outlier review contact sheet

```powershell
python make_outlier_review.py
```

This generates `outlier_review.jpg` for visual inspection of candidate outliers.

**Reproduction note:** The commands above reflect the project's script names and intended workflow. A fresh run should be verified against the submitted scripts, including their input and output dependencies. The analysis stages can also be run using the supplied intermediate files where appropriate, without repeating the full screenshot collection.

## 8. Output Files

| File | Purpose |
|---|---|
| `domains.csv` | Input domain dataset |
| `manifest.csv` | Capture outcomes and metadata |
| `clean_captures.csv` | Filtered records used for embeddings |
| `embeddings.npy` | 512-dimensional CLIP image embeddings |
| `embedding_domains.csv` | Domain order corresponding to embeddings |
| `cluster_scores.csv` | Silhouette scores for candidate cluster counts |
| `cluster_assignments.csv` | Cluster assignments |
| `cluster_visualization.png` | Two-dimensional cluster visualization |
| `similar_website_pairs.csv` | Ranked similar screenshot pairs |
| `zero_shot_predictions.csv` | Per-screenshot classification predictions |
| `zero_shot_summary.csv` | Counts and shares by predicted category |
| `visual_outliers.csv` | Visual outlier candidates |
| `outlier_review.jpg` | Contact sheet of 15 outlier screenshots |
| `top_pair_hash_check.csv` | Exact-image hash comparison for top pairs |

Additional intermediate files may be generated by the quality-checking scripts.

## 9. Limitations

- Capture coverage is incomplete: not every domain produced a usable screenshot.
- Filtering may exclude legitimate pages with sparse or repeated designs.
- Anti-bot challenges can create misleading near-duplicate matches.
- The best tested silhouette score indicates weak cluster separation.
- CLIP similarity is not proof that two websites are duplicates.
- Zero-shot predictions have not been comprehensively validated against human labels.
- Rank-bucket results may be biased by differences in capture success and filtering.
- Screenshots do not represent the full interactive website experience.
- The reported category shares are exploratory and do not establish causal relationships.

## 10. Recommended Next Steps

1. Manually validate a stratified sample of predictions across rank buckets and categories.
2. Review high-similarity pairs and separate genuine design similarities from shared anti-bot pages.
3. Measure capture success and filtering rates by rank bucket.
4. Re-run comparisons after excluding suspicious captures and check whether the findings persist.
5. Improve retry handling and incomplete-render detection without bypassing access restrictions.
6. Reassess clustering after improving capture quality and validating the labels.

## 11. Conclusion

WebSight AI demonstrates an end-to-end workflow for visual website analysis using pretrained CLIP embeddings. The strongest current value is surfacing candidate visual relationships, unusual designs, and potential category trends for further review.

The results also demonstrate why capture quality, careful model interpretation, and human validation are essential before making firm claims about website similarity or category prevalence.

The analysis outputs, manifest, embeddings, scripts, and documentation should be retained together to support reproducibility.
