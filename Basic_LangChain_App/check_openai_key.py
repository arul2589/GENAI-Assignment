import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPENAI_API_KEY"):
    print("OpenAI API key loaded. Value hidden.")
else:
    print("OpenAI API key not found.")