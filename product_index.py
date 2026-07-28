from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
PRODUCT_CATALOG = [

    {"name": "AeroBook Pro 14", "category": "Electronics",
     "synonyms": "laptop, notebook, computer"},
    {"name": "Nova X12 Smartphone", "category": "Electronics",
     "synonyms": "phone, mobile, cellphone, smartphone"},
    {"name": "SoundWave Pro Wireless Headphones", "category": "Electronics",
     "synonyms": "headphones, earphones, audio, headset"},
    {"name": "Summit Insulated Winter Jacket", "category": "Clothing",
     "synonyms": "jacket, coat, winter wear, outerwear"},
    {"name": "Nike Air Insulated Jacket", "category": "Clothing",
     "synonyms": "jacket, coat, winter wear, outerwear"},   # ← NEW, deliberately ambiguous
    {"name": "StrideFlex Running Sneakers", "category": "Footwear",
     "synonyms": "sneakers, running shoes, footwear, trainers"},
    {"name": "PulseTrack Smartwatch", "category": "Electronics",
     "synonyms": "watch, smartwatch, wearable"},
    {"name": "TrailPro 28L Backpack", "category": "Accessories",
     "synonyms": "backpack, bag, rucksack, daypack"},
    {"name": "BrewCraft Programmable Coffee Maker", "category": "Home Appliances",
     "synonyms": "coffee maker, coffee machine, brewer"},
]

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain_core.documents import Document

documents = []
for product in PRODUCT_CATALOG:
    enriched_text = f"{product['name']} - {product['synonyms']}. Category: {product['category']}"
    documents.append(Document(
        page_content=enriched_text,                    # ← this gets EMBEDDED
        metadata={"product_name": product["name"]}       # ← this gets STORED but NOT embedded, 
                                                           #    retrieved later as-is
    ))
def build_product_index():
    documents = []
    for product in PRODUCT_CATALOG:
        enriched_text = f"{product['name']} - {product['synonyms']}. Category: {product['category']}"
        documents.append(Document(
            page_content=enriched_text,
            metadata={"product_name": product["name"]}
        ))

    vectorstore = Chroma.from_documents(
        documents,
        embedding=embedder,
        collection_name="indexed_products",
        persist_directory="./chroma_data",
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

def get_product_index():
    return Chroma(
        collection_name="indexed_products",
        embedding_function=embedder,
        persist_directory="./chroma_data"
    )
def find_likely_products(question: str, k: int = 3, score_threshold: float = 0.15) -> list[str]:
    index = get_product_index()   # ← calls function 1 to get the connection
    results = index.similarity_search_with_relevance_scores(question, k=k)
    
    matches = [
        doc.metadata["product_name"]
        for doc, score in results
        if score >= score_threshold
    ]
    return matches
if __name__ == "__main__":
    build_product_index()
    print("Product index built.")