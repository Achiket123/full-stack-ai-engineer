from pinecone import Pinecone, ServerlessSpec
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
model= SentenceTransformer("all-MiniLM-L6-v2")
import time
docs = [
    {"id":"d1","text":"Agentic AI agents use tools, memory, and goals to act."},
    {"id":"d2","text":"LangChain and CrewAI help orchestrate multi-agent workflows."},
    {"id":"d3","text":"RAG retrieves external knowledge to improve answer accuracy."},
    {"id":"d4","text":"Vector databases enable fast similarity search over embeddings."},
    {"id":"d5","text":"Planning loops and ReAct improve reasoning in complex tasks."},
]
queries = [
    "How do agents use memory?",
    "Name a framework for multi-agent orchestration.",
    "Why is RAG useful?"
]

def gen_embeddings(text:str, model:SentenceTransformer):
    return model.encode(text)

X = [gen_embeddings(d["text"],model) for d in docs]     # document vectors
Q = [gen_embeddings(q,model) for q in queries]          # query vectors

dim= 384

xb = np.array(X,dtype="float32")
index = faiss.IndexFlatIP(dim)

faiss.normalize_L2(xb)
index.add(xb)
print("Indexed vectors:", index.ntotal)


def faiss_search(query_vec,k =3):
    q = np.array([query_vec],dtype="float32")
    faiss.normalize_L2(q)
    D,I = index.search(q, k )
    return D[0], I[0]

for qi, qv in enumerate(Q):
    D, I = faiss_search(qv, k=2)
    print("\nQuery:", queries[qi])
    for rank, (score, idx) in enumerate(zip(D, I), 1):
        print(f"  {rank}. id={docs[idx]['id']} score={round(float(score),4)}  text={docs[idx]['text']}")

faiss.write_index(index, "faiss_agentic.index")
index2 = faiss.read_index("faiss_agentic.index")
print("Reloaded vectors:", index2.ntotal)

pc = Pinecone(api_key= "pcsk_48VHeH_FZtfhx8VYjVUUW7t2tWV9KjyPkgY2UesoBhqPXCdGyW9ZKMyML13Wo9zGq724Yx")
index_name = "developer-quickstart-py"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
    while True:
        d = pc.describe_index(index_name)
        if d.status["ready"]: break
        time.sleep(2)
 
index = pc.Index(index_name)

def normalize(v):
    v = np.array(v, dtype="float32")
    n = np.linalg.norm(v)
    return (v/n).tolist() if n>0 else v.tolist()

vectors = [{
    "id": d["id"],
    "values": normalize(vec),
    "metadata": {"text": d["text"]}
} for d, vec in zip(docs, X)]


index.upsert(vectors=vectors)

def pinecone_search(query_vec, top_k=3):
    res = index.query(
        vector=normalize(query_vec),
        top_k=top_k,
        include_metadata=True
    )
    return res

for qi, qv in enumerate(Q):
    res = pinecone_search(qv, top_k=2)
    print("\n[Query]", queries[qi])
    for match in res["matches"]:
        print(f"  id={match['id']}  score={round(match['score'],4)}  text={match['metadata']['text']}")