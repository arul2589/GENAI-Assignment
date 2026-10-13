from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Explain things in simple words."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

model = ChatOllama(model="qwen2.5:3b")
output_parser = StrOutputParser()

chain = prompt_template | model | output_parser
chat_history = []

while True:
    question = input("\nYou (type 'exit' to stop): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    answer = chain.invoke({
        "chat_history": chat_history,
        "question": question
    })

    print("\nAI:", answer)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))