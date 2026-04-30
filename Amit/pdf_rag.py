from langchain_community.document_loaders import (
    PyMuPDFLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

pdf_loader = PyMuPDFLoader("ecs-dg.pdf")
#creating docs
pdf_docs = pdf_loader.load()
print(len(pdf_docs))

#create recursive text splitter
splitted_text = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n"],
)

chunks = splitted_text.split_documents(pdf_docs)
print(len(chunks))

# Create Open AI embedding 
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
#embedded_doc = embeddings.embed_documents(pdf_docs)
#print(embedded_doc)[0]

# Create FAISS vector store
vectorstore=FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
vectorstore.save_local("faiss_index")

## Similarity Search 
#query="How to create container image for ECS"

#results=vectorstore.similarity_search(query,k=3)
# results_with_score = vectorstore.similarity_search_with_score(query,k=3)
# print(results_with_score)

#create Retriver
retriever=vectorstore.as_retriever(
    search_kwarg={"k":3} ## Retrieve top 3 relevant chunks
)

#create LLM 
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    model_name="gpt-3.5-turbo"
)


## Create a prompt template
from langchain_core.prompts import ChatPromptTemplate
system_prompt="""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

### Create a document chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

document_chain=create_stuff_documents_chain(llm,prompt)

### Create The Final RAG Chain
from langchain_classic.chains import create_retrieval_chain
rag_chain=create_retrieval_chain(retriever,document_chain)

#ask question
response=rag_chain.invoke({"input":"How to create container image for ECS"})
print(response['answer'])