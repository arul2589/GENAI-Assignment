from dotenv import load_dotenv
load_dotenv()
from crewai import Agent, LLM, Crew, Task, Process
from crewai_tools import SerperDevTool
from datetime import datetime
from zoneinfo import ZoneInfo


today = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%B %d, %Y")

print("Today:", today)

llm = LLM(model="gpt-4o-mini")

serper_tool = SerperDevTool()

researcher = Agent(
    role="Technology News Researcher",
    goal="Find today's top 3 important technology news stories using web search.",
    backstory=(
    "You are an experienced technology journalist who researches "
    "the latest developments in AI, cloud computing, cybersecurity, "
    "and emerging technologies. You verify information from "
    "reliable sources and provide accurate news summaries."
    ),
    tools=[serper_tool],
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Technology Newsletter Writer",

    goal="Convert the researcher's top 3 technology stories into a clear, engaging newsletter of exactly 5 lines.",

    backstory=(
        "You are a professional newsletter writer who simplifies "
        "complex technology news for beginners. You write short, "
        "interesting, and easy-to-understand newsletters using "
        "only the information provided by the researcher."
    ),
    llm=llm,
    verbose=True
)



task1 = Task(
    description=(
    f"Today is {today}. "
    "Use SerperDevTool to search for today's top 3 technology news stories. "
    "Prioritize important developments in AI, cloud computing, "
    "cybersecurity, and emerging technologies. "
    "For each story, provide its headline, a short summary, "
    "the publication date, and a reliable source URL. "
    "Do not invent news or use outdated stories."
    ),

    expected_output=(
    "Exactly 3 verified technology news stories, each containing "
    "a headline, short summary, publication date, and source URL."
    ),
    agent =researcher
)


task2 = Task(
    description=(
        f"Today is {today}. "
        "Using ONLY the Researcher's output, create a tech newsletter. "
        "You MUST format the newsletter as EXACTLY 5 separate lines. "
        "Each line must end with a newline character except the last. "
        "Do not combine multiple stories into one paragraph. "
        "Use this exact structure:\n"
        "Tech Today Newsletter - {today}\n"
        "1. [First technology news summary]\n"
        "2. [Second technology news summary]\n"
        "3. [Third technology news summary]\n"
        "Why it matters: [One short concluding sentence]\n"
        "Do not add introductions, extra headings, blank lines, "
        "Markdown formatting, or explanations. "
        "Before submitting, count the lines and ensure there are exactly 5."
    ),
    expected_output=(
        "Exactly 5 non-empty lines separated by newline characters:\n"
        "Line 1: Tech Today Newsletter - {today}\n"
        "Line 2: 1. First story summary\n"
        "Line 3: 2. Second story summary\n"
        "Line 4: 3. Third story summary\n"
        "Line 5: Why it matters: One sentence\n"
        "No blank lines, extra text, or additional lines."
    ),
    agent=writer,
    context=[task1]
)


crew = Crew(agents=[researcher, writer], tasks=[task1, task2],process=Process.sequential,verbose=True)

result = crew.kickoff()

newsletter = result.raw.strip()

lines = [
    line.strip()
    for line in newsletter.splitlines()
    if line.strip()
]

print("\nFinal Newsletter:\n")
print(newsletter)

print("\nTotal lines:", len(lines))

if len(lines) == 5:
    print("PASS: Newsletter contains exactly 5 lines.")
else:
    print("FAIL: Expected 5 lines, received", len(lines))