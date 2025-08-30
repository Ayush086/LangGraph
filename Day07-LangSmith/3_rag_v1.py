import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = 'RAG ChatBot' # tracing name

load_dotenv()  

PDF_PATH = "islr.pdf"  

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
splits = splitter.split_documents(docs[:10])

# 3) Embed + index
# emb = OpenAIEmbeddings(model="text-embedding-3-small")
emb = CohereEmbeddings(model = 'embed-english-v3.0')
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatCohere(model='command-r-plus', temperature=0)
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)


"""
By default, langsmith traces the langchain runnables. That's why entire workflow is not being traced
Therefore, it's partially traced

Worflow flaws:
for every run we are creating embeddings, chunking and all. What I must do is to store this one time process because then every time I want to ask a query it will take quite much time.

Problems:
1. whole workflow is not getting traced
2. repeated execution of steps which can be executed for one time and it's enough.
"""