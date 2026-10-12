
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

prompts = [
    "Explain Python in one simple sentence.",
    "What is LangChain? Answer in two short sentences.",
    "Explain RAG to a beginner, using a simple example."
]

for prompt in prompts:
    response = llm.invoke(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Answer: {response.content}")