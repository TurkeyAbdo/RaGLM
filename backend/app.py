import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. SETUP: Define constants and models ---
# This is the model we'll use for embeddings. It's small and runs locally.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# This is the LLM we'll use for generation.
# Using a smaller, instruction-tuned model for the PoC.
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" # A great small model as an alternative to DeepSeek

# --- 2. DATA PREPARATION: Create sample documents and store them in ChromaDB ---
print("Step 2: Preparing data and creating vector store...")

# Sample documents with role-based metadata.
# This simulates data that would come from your MySQL database.
documents = [
    ("The annual financial report is due on August 30th. It must be submitted by the finance head.", {"role": "Reporter"}),
    ("Our latest research indicates a 20% increase in market share for product X.", {"role": "Researcher"}),
    ("Company policy requires all employees to complete security training by October 1st.", {"role": "Reporter"}),
    ("The quantum computing research paper outlines a new algorithm for data encryption.", {"role": "Researcher"}),
]

# Extract just the text content for processing
doc_texts = [doc[0] for doc in documents]
doc_metadatas = [doc[1] for doc in documents]

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunked_texts = text_splitter.create_documents(doc_texts, metadatas=doc_metadatas)

# Create embeddings
# This will download the model from Hugging Face the first time you run it.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Create a Chroma vector store in memory
# This stores the vectorized chunks.
vector_store = Chroma.from_documents(
    documents=chunked_texts,
    embedding=embeddings,
    collection_name="local_rag"
)
print("Vector store created successfully.")

# --- 3. LOAD THE LOCAL LLM ---
print("\nStep 3: Loading the local LLM from Hugging Face...")
# This will download the model files (can be several GBs) the first time.
# Ensure you have enough disk space.
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.bfloat16, # Use bfloat16 for less memory usage
    device_map="auto", # Automatically use GPU if available
    trust_remote_code=True, # Required for some models like Phi-3
)

# Create a text generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256, # Limit the length of the generated answer
)
print("LLM loaded successfully.")


# --- 4. THE RAG FUNCTION (with Role-Based Access Control) ---
def answer_query(query: str, user_role: str):
    print(f"\n--- Answering Query for Role: {user_role} ---")
    print(f"Query: {query}")

    # 4a. Retrieve relevant documents with RBAC filter
    print("Retrieving relevant documents from vector store...")
    retriever = vector_store.as_retriever(
        search_kwargs={'k': 2, 'filter': {'role': user_role}}
    )
    relevant_docs = retriever.get_relevant_documents(query)

    if not relevant_docs:
        return "I do not have access to information for that query with your role."

    context = "\n".join([doc.page_content for doc in relevant_docs])
    print(f"Context retrieved:\n{context}")

    # 4b. Create the prompt for the LLM
    prompt_template = f"""
    <|system|>
    You are a helpful AI assistant. Answer the user's question based only on the context provided below.
    If the context does not contain the answer, say "I do not have enough information to answer."

    Context:
    {context}
    <|end|>
    <|user|>
    {query}
    <|end|>
    <|assistant|>
    """

    # 4c. Generate the answer
    print("Generating answer with LLM...")
    result = llm_pipeline(prompt_template)
    return result[0]['generated_text'].split("<|assistant|>")[1].strip()


# --- 5. EXECUTION: Simulate user queries ---
# Simulate a query from a "Reporter"
reporter_answer = answer_query("What is the deadline for the financial report?", "Reporter")
print(f"\nFinal Answer for Reporter: {reporter_answer}")

# Simulate a query from a "Researcher"
researcher_answer = answer_query("What did the latest research find?", "Researcher")
print(f"\nFinal Answer for Researcher: {researcher_answer}")

# Simulate a query where the role does not have access
failed_answer = answer_query("Tell me about the quantum computing paper.", "Reporter")
print(f"\nFinal Answer for failed query: {failed_answer}")

