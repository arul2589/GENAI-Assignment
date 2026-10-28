
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

assistant = AssistantAgent(
    name="HR_Assistant",
    model_client=model_client,
    system_message=(
        "You are an experienced HR recruiter. "
        "Your responsibility is to generate "
        "professional job descriptions."
        )

)

async def main():
    try:
        await Console(
            assistant.run_stream(
                task=(
                    "Write a job description for a Junior "
                    "Python Developer in exactly 5 numbered "
                    "lines. Do not add a heading or extra text."
                )
            )
        )
    finally:
        await model_client.close()

asyncio.run(main())

