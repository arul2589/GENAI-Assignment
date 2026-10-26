from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
load_dotenv()



model = LLM(model="gpt-4o-mini")

writer = Agent(
    role="Travel Planner",
    goal="Help users plan trips and recommend places to visit",
    backstory=(
        "You are an experienced travel planner with extensive "
        "knowledge of destinations across India, local culture, "
        "tourist attractions, and travel tips."
    ),
    llm=model,
    verbose=True
)



travel_adviser = Task(
    description=(
        "The user wants to visit {city}. "
        "Recommend the 3 best tourist places in {city}. "
        "Explain why each place is worth visiting."
    ),
    expected_output=(
        "A list of 3 tourist places with descriptions."
    ),
    agent=writer
)

crew = Crew(agents=[writer], tasks=[travel_adviser])

city = input("Enter the city you want to visit: ")

result = crew.kickoff(inputs = {"city": city})

print("\nTravel Recommendations:")
print(result)



