from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from zoneinfo import ZoneInfo
import asyncio
from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

today = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")

# Tool: Search weather using DDGS
def search_weather_news(city: str) -> str:
    """
    Search DuckDuckGo for the latest weather news
    for the requested city.

    Return recent weather-related news results.
    """
    try:
        today = datetime.now(
            ZoneInfo("Asia/Kolkata")
        ).strftime("%d %B %Y")

        results = DDGS().news(
            query=f"{city} weather news forecast {today}",
            max_results=5
        )

        if not results or not any(
            item.get("title") or item.get("body")
            for item in results
        ):
            return "No recent weather news found."

        return str(results)

    except Exception as e:
        return f"Weather search error: {e}"

def search_weather_news(city: str) -> str:
    """Search for current weather information."""
    today = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")

    results = DDGS().text(
        query=f"{city} weather temperature forecast {today}",
        max_results=5
    )
    return str(results)


city = input("Which city? ").strip()

# Configure OpenAI model
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# System prompt
weather_system_prompt = """
You are a professional AI Weather News Reporter.

Your task is to generate a short weather news report
for the city requested by the user.

Rules:
1. Always use the search_weather_news tool.
2. Use only information supported by the search results.
3. Do not invent temperatures, rainfall, or forecasts.
4. If the current temperature is unavailable, say so.
5. Generate EXACTLY 3 short lines.
6. Every line must contain meaningful information.
7. Never return only 1 or 2 lines.
8. Do not add headings, bullet points, or extra explanations.

Required output format:
Current Weather: [City and verified weather information].
Forecast: [Verified forecast or information unavailable].
Advice: [One practical recommendation].

Before responding, verify that your final answer
contains exactly 3 non-empty lines.

Return only the final report.
"""

agent = AssistantAgent(
   name="weather_news", 
   model_client=model_client,
 tools=[search_weather_news], 
 system_message=weather_system_prompt,
 reflect_on_tool_use=True
 )

async def main():
    today = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y")

    try:
        await Console(
            agent.run_stream(
                task=(
                    f"Today's date is {today}. "
                    f"Search the latest weather for {city} on {today}. "
                    "Generate exactly 3 short lines. "
                    "Do not use outdated weather reports."
                )
            )
        )
    finally:
        await model_client.close()

asyncio.run(main())