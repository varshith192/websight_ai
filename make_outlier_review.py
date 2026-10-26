
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

INPUT_FILE = Path("visual_outliers.csv")
OUTPUT_FILE = Path("outlier_review.jpg")
SCREENSHOT_DIR = Path("screenshots")

data = pd.read_csv(INPUT_FILE).head(15)

cell_width = 320
cell_height = 220
columns = 3
rows = (len(data) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (cell_width * columns, cell_height * rows),
    "white",
)
draw = ImageDraw.Draw(sheet)

for i, row in enumerate(data.itertuples(index=False)):
    domain = row.domain
    screenshot_path = SCREENSHOT_DIR / f"{domain}.png"

    x = (i % columns) * cell_width
    y = (i // columns) * cell_height

    draw.text(
        (x + 5, y + 5),
        str(domain)[:42],
        fill="black",
    )

    try:
        with Image.open(screenshot_path) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_width - 10, cell_height - 35))
            sheet.paste(image, (x + 5, y + 28))
    except Exception as error:
        draw.text(
            (x + 5, y + 40),
            f"Could not load image: {error}",
            fill="red",
        )

sheet.save(OUTPUT_FILE, quality=90)

print(f"Created: {OUTPUT_FILE.resolve()}")
print(f"Screenshots included: {len(data)}")
