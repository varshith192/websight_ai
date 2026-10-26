
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from playwright.async_api import async_playwright

INPUT_FILE = "domains.csv"
OUTPUT_DIR = Path("screenshots")
OUTPUT_DIR.mkdir(exist_ok=True)

MANIFEST_FILE = Path("manifest.csv")
MAX_SITES = 5000
TIMEOUT_MS = 20000


async def main():
    df = pd.read_csv(INPUT_FILE)
    results = []

    if MANIFEST_FILE.exists():
        results = pd.read_csv(MANIFEST_FILE).to_dict("records")

    completed = {row["domain"] for row in results}
    df = df[~df["domain"].astype(str).isin(completed)].head(MAX_SITES)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for _, row in df.iterrows():
            domain = str(row["domain"]).strip()
            url = "https://" + domain
            page = None

            result = {
                "domain": domain,
                "tier": row.get("tier", ""),
                "status": "other",
                "http_status": None,
                "final_url": "",
                "page_title": "",
                "load_ms": None,
                "screenshot_path": "",
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }

            try:
                page = await browser.new_page(
                    viewport={"width": 1280, "height": 800}
                )

                start = asyncio.get_running_loop().time()
                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=TIMEOUT_MS,
                )

                result["load_ms"] = round(
                    (asyncio.get_running_loop().time() - start) * 1000, 2
                )
                result["http_status"] = response.status if response else None
                result["final_url"] = page.url
                result["page_title"] = await page.title()

                screenshot = OUTPUT_DIR / f"{domain.replace('/', '_')}.png"
                await page.screenshot(
                    path=str(screenshot),
                    full_page=False,
                    animations="disabled",
                )

                result["screenshot_path"] = str(screenshot)
                result["status"] = (
                    "ok" if response is None or response.status < 400
                    else "http_error"
                )

            except Exception as e:
                message = str(e).lower()
                if "timeout" in message:
                    result["status"] = "timeout"
                elif "name_not_resolved" in message or "nxdomain" in message:
                    result["status"] = "dns_error"
                else:
                    result["status"] = "other"
                print(f"Failed: {domain} — {result['status']}")

            finally:
                if page:
                    await page.close()

                results.append(result)
                pd.DataFrame(results).to_csv(MANIFEST_FILE, index=False)
                print(f"{domain}: {result['status']}")

        await browser.close()

    print(f"\nFinished. Results saved to {MANIFEST_FILE}")
    print(f"Screenshots saved in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
