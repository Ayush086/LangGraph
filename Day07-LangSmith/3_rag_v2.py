import os
from dotenv import load_dotenv

from langsmith import traceable  # used as decorators to trace steps which we want to trace

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = 'RAG ChatBot'


load_dotenv()

PDF_PATH = "islr.pdf" 

# making steps other than runnables traceble
@traceable(name="load_document")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs  

@traceable(name="split_documents")
def split_documents(docs, chunk_size=2000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs[:5])

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    # emb = OpenAIEmbeddings(model="text-embedding-3-small")
    emb = CohereEmbeddings(model = 'embed-english-v3.0')
    # FAISS.from_documents internally calls the embedding model:
    vs = FAISS.from_documents(splits, emb)
    return vs

# tracing workflows separately which are not default
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

# ---------- pipeline ----------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatCohere(model="command-r", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Build the index under traced setup
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# 
config = {
    "run_name": "workflow-traceble-RAG" # setting the trace run name
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)


"""
Resolved Issues:
- not tracing workflow other than runnable

Issues:
- both traces are listed separately we must group them under one tree
"""








# import os
# from dotenv import load_dotenv

# from langsmith import traceable
# from langsmith.run_helpers import tracing_context

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_cohere import ChatCohere, CohereEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()
# PDF_PATH = "islr.pdf"

# # ----------------- Traceable Functions -----------------
# @traceable(name="load_document")
# def load_pdf(path: str):
#     loader = PyPDFLoader(path)
#     docs = loader.load()
#     return docs

# @traceable(name="split_documents")
# def split_documents(docs, chunk_size=2000, chunk_overlap=150):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return splitter.split_documents(docs[:5])

# @traceable(name="build_vectorstore")
# def build_vectorstore(splits):
#     emb = CohereEmbeddings(model="embed-english-v3.0")
#     vs = FAISS.from_documents(splits, emb)
#     return vs

# @traceable(name="setup_pipeline")
# def setup_pipeline(pdf_path: str):
#     docs = load_pdf(pdf_path)
#     splits = split_documents(docs)
#     vs = build_vectorstore(splits)
#     return vs

# # ----------------- RAG Chain Setup -----------------
# llm = ChatCohere(model="command-r", temperature=0)
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
#     ("human", "Question: {question}\n\nContext:\n{context}")
# ])

# def format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)

# with tracing_context(project_name="traceble-RAG", run_name="pipeline-setup-trace", enabled=True):
#     vectorstore = setup_pipeline(PDF_PATH)
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# # Build the RAG chain outside of the first trace
# parallel = RunnableParallel({
#     "context": retriever | RunnableLambda(format_docs),
#     "question": RunnablePassthrough(),
# })
# chain = parallel | prompt | llm | StrOutputParser()

# print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
# q = input("\nQ: ").strip()

# with tracing_context(project_name="traceble-RAG", run_name="workflow-traceable-RAG", enabled=True):
#     ans = chain.invoke(q)

# print("\nA:", ans)
