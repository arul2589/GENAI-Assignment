
import requests
from bs4 import BeautifulSoup

# BBC News website
URL = "https://www.bbc.com/news"

# Send a normal browser User-Agent
headers = {"User-Agent": "Mozilla/5.0"}

try:
    # Fetch the webpage
    response = requests.get(URL, headers=headers, timeout=10)

    # Check the HTTP status before parsing
    response.raise_for_status()

    # Parse the downloaded HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Use the headline selector found using Inspect
    items = soup.select('h2[data-testid="card-headline"]')

    # Remove empty and duplicate headlines
    headlines = []
    seen = set()

    for item in items:
        headline = item.get_text(" ", strip=True)

        if headline and headline not in seen:
            headlines.append(headline)
            seen.add(headline)

        # Stop once we have five headlines
        if len(headlines) == 5:
            break

    # Print the results
    print("Top 5 BBC News Headlines:\n")

    for number, headline in enumerate(headlines, start=1):
        print(f"{number}. {headline}")

    # Handle fewer than five results
    if len(headlines) < 5:
        print(f"\nOnly {len(headlines)} headlines found.")

except requests.RequestException as error:
    # Handle network and HTTP errors
    print("Failed to fetch BBC News:", error)
