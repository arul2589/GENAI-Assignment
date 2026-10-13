from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load OPENAI_API_KEY from the .env file
load_dotenv()

# Create a LangChain connection to the OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200)

# Tell the model exactly what answer format we want
prompt = """
Suggest exactly 5 healthy breakfast ideas.
For this request, define healthy as including protein and fiber, with little added sugar.
Format the answer as a numbered list from 1 to 5, one idea per line.
Do not add an introduction or explanations.
"""

# Send the prompt to the model
reply = llm.invoke(prompt)

# For this first run, print the complete reply object
print(reply.content)