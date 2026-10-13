from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # Load the key from .env in this script too

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

response = model.invoke("Reply with exactly: OpenAI connection works.")
print(response.content)