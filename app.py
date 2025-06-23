from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Load and split documents
loader = TextLoader("data/my_docs.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 2. Convert text to vector using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# 3. Load a local LLM (e.g., tiny model from HuggingFace)
# model_name = "microsoft/DialoGPT-medium"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=300, do_sample=True)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 4. Create a RetrievalQA Chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

# 5. Chat Loop
print("🤖 AI Chatbot Ready! Ask anything based on your documents.\n(Type 'exit' to quit)")


while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

   
    retrieved_docs = db.as_retriever().get_relevant_documents(query)

    if not retrieved_docs:
        print("Bot: Sorry, I couldn't find anything related to that in the documents.\n")
        continue

    # Proceed with normal QA if documents were found
    answer = qa.run(query)
    print(f"Bot: {answer}\n")
