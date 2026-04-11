import pandas as pd
from pathlib import Path

data_dir = Path("data")

records = []

for file in data_dir.glob("phish-*.txt"):

    brand = file.stem.split("-")[1]  # amazon, google etc

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

        # skip metadata lines
        urls = lines[3:]

        for url in urls:
            url = url.strip()

            if url:
                records.append({
                    "url": url,
                    "brand": brand
                })

df = pd.DataFrame(records)

print(df.head())
print("\nTotal rows:", len(df))

df.to_csv("data/phishing_dataset.csv", index=False)