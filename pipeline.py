# pipeline.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads

load_dotenv()

# ============================================
# Embedding model (same one used during indexing)
# ============================================
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ============================================
# ChromaDB collection (already indexed by indexing.py)
# ============================================
vectorstore = Chroma(
    collection_name="ecommerce_docs",
    embedding_function=embedder,
    persist_directory="./chroma_data"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ============================================
# LLM (Groq - used for query generation and final answer)
# ============================================
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# ============================================
# RAG-FUSION: query variant generation
# ============================================
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
# RAG-FUSION: Reciprocal Rank Fusion merge
# ============================================
def reciprocal_rank_fusion(results: list[list], k: int = 60):
    """Merges multiple ranked chunk lists into one ranked list, scoring
    chunks higher when they appear across multiple variants."""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [loads(doc) for doc, score in reranked]


# Full fusion retrieval chain (variants -> retrieve each -> RRF merge)
fusion_retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

# ============================================
# Generation prompt
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
# Full RAG-Fusion answer (used by the "careful" branch)
# ============================================
def answer_question(question: str) -> str:
    """RAG-fusion: 4 query variants + retrieval + RRF merge + generation.
    Costs 2 LLM calls total."""
    retrieved_chunks = fusion_retrieval_chain.invoke({"question": question})
    context_text = "\n\n".join(chunk.page_content for chunk in retrieved_chunks[:5])
    return generation_chain.invoke({"context": context_text, "question": question})


# ============================================
# Direct single-query answer (used by the "simple" branch)
# ============================================
def simple_answer_question(question: str) -> str:
    """Direct retrieval + generation, no fusion. Costs 1 LLM call total.
    Faster; use when the question is a straightforward factual lookup where
    vocabulary mismatch is unlikely."""
    retrieved_chunks = retriever.invoke(question)
    context_text = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)
    return generation_chain.invoke({"context": context_text, "question": question})

