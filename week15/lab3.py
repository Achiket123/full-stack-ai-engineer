# Episodic Memory
from sentence_transformers import SentenceTransformer
import chromadb
import time
import uuid
import datetime 
from datetime import timezone

import math
def embed (model,text):
    return model.encode(text)

chroma = chromadb.Client()
episodes = chroma.get_or_create_collection(name="episodic_memory")  # id, embedding, metadata

def now_ts():
    return int(time.time())


def add_episode(summary:str, who:str="user", tags=None):
    eid = str (uuid.uuid4())
    meta = {
        "who":who,
        "summary":summary,
        "ts":now_ts(),
        "tags":",".join(tags) if tags else ""
    }
    episodes.add(ids=[eid], embeddings=[embed(summary)],metadatas=[meta], documents=[summary])
    return eid

def ts_to_str(ts:int):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def search_episodes(query:str, k:int=5, alpha:float=0.7, tau_hours:float=72):
    qemb = embed(query)
    # raw similarity search
    res = episodes.query(query_embeddings=[qemb], n_results=20, include=["metadatas","distances","documents","embeddings"])
    if not res["ids"]: return []
 
    now = now_ts()
    tau = tau_hours * 3600.0
    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        doc  = res["documents"][0][i]
        # Chroma returns distance; convert to similarity (cosine-ish). Guard if distance missing.
        distance = res["distances"][0][i] if res["distances"] else 0.0
        similarity = 1 - distance  # crude invert; OK for our demo
        dt = max(0, now - int(meta["ts"]))
        rec = math.exp(-dt / tau)
        blended = alpha*similarity + (1-alpha)*rec
        out.append({
            "id": res["ids"][0][i],
            "summary": doc,
            "who": meta["who"],
            "ts": meta["ts"],
            "when": ts_to_str(int(meta["ts"])),
            "similarity": round(similarity,4),
            "recency": round(rec,4),
            "score": round(blended,4),
            "tags": meta.get("tags","")
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:k]


"""
def llm_chat(system:str, user:str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ]
    )
    return resp.choices[0].message.content
 
def format_memories(mem_list):
    lines = []
    for m in mem_list:
        lines.append(f"- ({m['when']}) [{m['who']}] {m['summary']}")
    return "\n".join(lines)
 
def agent_respond(user_text:str):
    # 1) retrieve related episodes
    mems = search_episodes(user_text, k=3, alpha=0.7, tau_hours=72)
    mem_context = format_memories(mems) if mems else "None"
    system = (
        "You are an assistant with episodic memory. "
        "If 'Relevant episodes' contain useful context, use it to inform your answer. "
        "Be concise and cite facts to the extent they appear in the episodes."
        f"\n\nRelevant episodes:\n{mem_context}"
    )
    # 2) generate answer
    answer = llm_chat(system, user_text)
    # 3) log this interaction as an episode
    add_episode(summary=f"Q: {user_text} | A: {answer[:250]}...", who="agent", tags=["dialog"])
    return answer, mems

"""