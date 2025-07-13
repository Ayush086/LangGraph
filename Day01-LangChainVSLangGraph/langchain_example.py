# build a JD creator tool                                                                                 08/07/2025
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

hiring_role = """ We need to hire a Machine Learning Engineer for our backend team."""

# create JD using LLM
llm = ChatCohere(model='command-r')

job_desc_prompt = ChatPromptTemplate.from_template(
    "Create a well structured job description based on the hiring request: \n\n {request}"
)

JD_chain = job_desc_prompt | llm | StrOutputParser()

# approval logic
def approve_job_desc(jd: str) -> bool: 
    res = input("jd approval ? (y/n): ")
    return res.lower() == 'y'

def post_job(jd: str):
    print("Job Description approved and posted:\n")
    print(jd)
    
    

# loop untill the jd is approved
approved = False
jd_output = None

while not approved:
    jd_output = JD_chain.invoke({'request': hiring_role})
    print(jd_output)
    
    approved = approve_job_desc(jd_output)
    
    if not approved:
        print("JD not approved, Regenerating....\n")
    else:
        post_job(jd_output)
