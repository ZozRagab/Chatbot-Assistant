import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ============================================
# Set up the embedding model (same one used during indexing)
# ============================================
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ============================================
# Connect to the ALREADY-INDEXED ChromaDB collection
# (no re-embedding of documents happens here, just loading what indexing.py already built)
# ============================================
vectorstore = Chroma(
    collection_name="ecommerce_docs",
    embedding_function=embedder,
    persist_directory="./chroma_data"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # top 3 chunks per query

# ============================================
# Set up the LLM (Groq)
# ============================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ============================================
# RAG-FUSION: Step 1 - generate multiple rephrased versions of the question
# ============================================
from langchain_core.load import dumps, loads

fusion_template = """You are a helpful assistant that generates multiple search
queries based on a single input query.
Generate 4 search queries related to: {question}
Output (one per line):"""

fusion_prompt = ChatPromptTemplate.from_template(fusion_template)

generate_queries = (
    fusion_prompt
    | llm
    | StrOutputParser()
    | (lambda x: [line.strip() for line in x.split("\n") if line.strip()])
)


# ============================================
# RAG-FUSION: Step 2 - Reciprocal Rank Fusion merge
# ============================================
def reciprocal_rank_fusion(results: list[list], k: int = 60):
    """Merges multiple ranked chunk lists into one ranked list,
    scoring chunks higher when they appear across multiple variants."""
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [loads(doc) for doc, score in reranked]


# ============================================
# RAG-FUSION: full retrieval chain (generate variants -> retrieve each -> RRF merge)
# ============================================
fusion_retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

# ============================================
# Generation prompt template
# ============================================
template = """Answer the question based only on the context below.
If the context doesn't contain enough information to answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

generation_chain = prompt | llm | StrOutputParser()


# ============================================
# The main function: question in, answer out
# ============================================
def answer_question(question: str) -> str:
    # Step 1: retrieve relevant chunks using RAG-Fusion
    # (generates 4 query variants, retrieves for each, merges via RRF)
    retrieved_chunks = fusion_retrieval_chain.invoke({"question": question})

    # Step 2: format retrieved chunks into a single context string
    # (limit to top 5 after fusion, to keep context focused)
    context_text = "\n\n".join(chunk.page_content for chunk in retrieved_chunks[:5])

    # Step 3: generate the answer using context + question
    answer = generation_chain.invoke({"context": context_text, "question": question})

    return answer


# ============================================
# Sanity check - run this file directly to test
# ============================================
if __name__ == "__main__":
    test_questions = [
        "What is your return policy?",
        "How long does shipping take?",
        "Do you sell umbrellas?"  # unrelated - should say it doesn't know
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {answer_question(q)}")