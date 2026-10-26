
from pathlib import Path
import pandas as pd

manifest = pd.read_csv("manifest.csv")
blank = pd.read_csv("blank_screenshots.csv")
duplicates = pd.read_csv("duplicate_screenshots.csv")

successful = manifest[manifest["status"] == "ok"].copy()

# Exclude potential blank/uniform screenshots.
blank_domains = set(blank["domain"].astype(str))

# Exclude every member of an exact-duplicate group.
duplicate_domains = set(duplicates["domain"].astype(str))

excluded_domains = blank_domains | duplicate_domains

clean = successful[
    ~successful["domain"].astype(str).isin(excluded_domains)
].copy()

clean.to_csv("clean_captures.csv", index=False)

print("Successful captures:", len(successful))
print("Potential blank/uniform domains:", len(blank_domains))
print("Domains in exact-duplicate groups:", len(duplicate_domains))
print("Domains excluded (combined):", len(excluded_domains))
print("Captures retained for analysis:", len(clean))
print("Saved: clean_captures.csv")
