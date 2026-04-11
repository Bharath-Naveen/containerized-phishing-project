import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT_DIR = Path("captures")
OUT_DIR.mkdir(exist_ok=True)


def capture(url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1365, "height": 900})
        page.goto(url, wait_until="networkidle", timeout=30000)

        final_url = page.url
        title = page.title()
        html = page.content()

        screenshot_path = OUT_DIR / "page.png"
        html_path = OUT_DIR / "page.html"
        meta_path = OUT_DIR / "meta.json"

        page.screenshot(path=str(screenshot_path), full_page=True)
        html_path.write_text(html, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "original_url": url,
                    "final_url": final_url,
                    "title": title,
                    "screenshot_path": str(screenshot_path),
                    "html_path": str(html_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        browser.close()
        return {
            "original_url": url,
            "final_url": final_url,
            "title": title,
            "screenshot_path": str(screenshot_path),
            "html_path": str(html_path),
        }


if __name__ == "__main__":
    test_url = "https://example.com"
    print(capture(test_url))