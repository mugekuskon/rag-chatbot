import os
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint

from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pprint, getpass

dotenv.load_dotenv() 

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",          
    temperature=0.3,
    max_new_tokens=512,              
    do_sample=True,
    trust_remote_code=True,
)
model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

pdf_name = input("PDF to be submitted:")
file_path = f"./rag_files/{pdf_name}.pdf"

loader = PDFMinerLoader(file_path)
docs = loader.load()
print(f"{pdf_name} submitted.")
pprint.pp(docs[0].metadata)

print(f"Loaded {len(docs)} document section(s).")
print(f"Total characters: {len(docs[0].page_content)}")
print(docs[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split pdf file into {len(all_splits)} sub-documents.")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])


