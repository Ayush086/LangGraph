# from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT'] = 'Sequential Chain App'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# model = ChatOpenAI()
model1 = ChatCohere(model = 'command-r', temperature=0.7)
model2 = ChatCohere(model = 'command-r-plus', temperature=0.4)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name': 'sequential-chain',
    'tags': ['summarization', 'report generation'],
    'metadata': {'model1': 'command-r', 'model2': 'command-r-plus'}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
