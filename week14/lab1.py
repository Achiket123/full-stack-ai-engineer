from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity 


from sklearn.manifold import TSNE
from parametric_umap import ParametricUMAP
def tut():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [    "Agentic AI",
        "Autonomous agents",
        "AI agents",
        "Artificial intelligence",
        "Machine learning",
        "Deep learning",
        "Dogs are animals",
        "Cats drink milk",
        "Bananas are yellow",
        "Apples are fruits",
        "Cars use petrol",
        "Trains run on tracks"]
    embeddings = model.encode(texts)

    # pca = PCA(n_components=2)
    # points = pca.fit_transform(embeddings)
    
    tsne = PCA(
    n_components=2, 
    )
    sim = cosine_similarity(embeddings)
    
    plt.imshow(sim)
    plt.xticks(range(len(texts)), texts, rotation=45)
    plt.yticks(range(len(texts)), texts)
    plt.colorbar()
    plt.show()

def token_embed_sim(texts):
    assert texts is not list, "Must be a list"
        
    model:SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(texts)
    similarity  = cosine_similarity([ embedding[0]], embedding)

    tokens = model.preprocess(texts)

    return tokens, similarity, embedding

if __name__ == "__main__":
    print(*token_embed_sim(["Hello My Friend, How are you ?","I am fine how about you ?"]), sep="\n")