

#go to https://www.cricbuzz.com/
#waite for page to load

#click on live score
#wait for the live score page to load
#print the score in terminal
#take a screenshot of the live score page and save it as "cricbuzz_live_score.png"

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

SCORE_SELECTOR = "body > div > main > div > div.flex.w-full > div.w-full.wb\:w-\[67\%\].min-h-page.relative.wb\:bg-white > div.flex.flex-col.gap-3.wb\:mt-1.mt-3.wb\:w-full.wb\:px-5 > div > div > div:nth-child(1) > div > div > a"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # First run: watch it work
    page = browser.new_page()

    try:
        page.goto("https://www.cricbuzz.com/", wait_until="domcontentloaded")

        page.get_by_text("Live Scores", exact=True).click(timeout=15000)

        score_element = page.locator(SCORE_SELECTOR).first
        score_element.wait_for(state="visible", timeout=15000)

        score = score_element.inner_text().strip()

        if score:
            print("Live score:")
            print(score)
            page.screenshot(path="score.png")
            print("Screenshot saved as score.png")
        else:
            print("The score element was found, but it contains no text.")

    except PlaywrightTimeoutError:
        print("Timed out while opening Live Scores or waiting for the score.")
        print("Current page:", page.url)
        print("Check that Live Scores opened and that SCORE_SELECTOR is correct.")

    finally:
        browser.close()