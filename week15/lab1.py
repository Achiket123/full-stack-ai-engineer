import requests
import json
import turso
from sentence_transformers import SentenceTransformer

uri = "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent"
encoder = json.encoder.JSONEncoder()
data = encoder.encode({
    "contents" : [
        {
            "parts":[
                {"text":"Explain How AI works in a few words"}
            ]
        }
    ]
})
def in_memory():
    response = requests.post(uri,data,headers={
        "content-type":"application/json",
        "X-goog-api-key":"<API>"
    })

    print(response.json()["candidates"][0]["content"]["parts"][0]['text'])


def vectordb():
    conn = turso.connect("week15/vector.db")
    cur = conn.cursor() 
    
    return cur, conn


def create_db(cur,conn):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        content         TEXT NOT NULL,
        embedding       F32_BLOB(384),
        category        TEXT NOT NULL,
        
        last_retrieved  INTEGER,
        retrieval_count INTEGER DEFAULT 0,
        source_task     TEXT
    );
    """)
    cur.execute("""

CREATE TABLE IF NOT EXISTS tasks (
        id               TEXT PRIMARY KEY,
        description      TEXT,
        embedding        F8_BLOB(384),
        started_at       INTEGER,
        finished_at      INTEGER
    );
                """)
    conn.commit()


def gen_embeddings(model:SentenceTransformer, texts:str):
    
    embeddings = model.encode_document(texts).tolist()

    return embeddings
def insert_to_db ( conn,texts,embeddings):
    conn.cursor().execute(
        f"""
INSERT INTO memories ( content, embedding, category)
VALUES ( ?, vector32(?), 'coversation');
"""
    , (texts, encoder.encode(embeddings)))
    conn.commit()

texts = """
Personal Background
Your name is Achiket Kumar.
You are from Renukoot, Uttar Pradesh, India.
You completed your schooling at Nirmala Convent High School.
You previously mentioned scoring:
96.20% in Class 10 (CBSE)
96.00% in Class 12 (CBSE)
You have mentioned being a B.Tech student and have discussed subjects such as:
Engineering Mathematics
Introduction to Computer Science
Operating Systems
Machine Learning
Technical Skills

You have worked with a fairly wide technology stack:

Mobile Development
Flutter
Dart
Bloc
Hive
Firebase
Supabase
Flutter Web
Backend Development
Go (Golang)
Gin Framework
Gorilla WebSocket
Node.js
Express
Databases
PostgreSQL
MySQL
Supabase
Firebase Firestore
MongoDB
Other Technologies
Docker
Nginx
WebRTC
REST APIs
Git/GitHub
Linux
Python
Professional Experience
Ricoz.in (Flutter Development Internship)

You mentioned:

Building mobile applications from scratch.
Developing Node.js backends.
Integrating Razorpay payments.
Implementing real-time chat functionality.
Publishing applications to the Play Store.
DST HUB LLP (Full Stack Development Internship)

You mentioned working as a Full Stack Developer and expanding your backend knowledge, particularly in Go and database systems.

Areas You Frequently Work On
Real-Time Systems

You spend a lot of time building:

Chat applications
WebSocket systems
WebRTC calling solutions
Signaling servers
Presence and notification systems
SaaS Development

One of your recurring interests is building:

Employee Management Systems
Task Management Platforms
Organization/Team Management software

You have discussed:

Database architecture
GORM models
PostgreSQL schema design
Role-based access control
Real-time collaboration
AI and Machine Learning

You've worked on:

Dataset preprocessing
Feature engineering
Logistic Regression
KNN
Data visualization
Pandas
Scikit-learn

You often ask not only how models work but also how to implement them correctly and understand the mathematics behind them.

Projects You've Discussed
Gemini AI Chatbot

Features included:

Gemini API integration
Question solving
Image analysis
Bloc architecture
Weather Application

Features included:

Local storage using Hive
Bloc state management
Custom UI components
Pull-to-refresh support
Blog Application

Built with:

Clean Architecture
SOLID principles
Studio Rental Platform

You discussed:

Studio discovery
Location-based search
Booking workflows
Technical documentation
Employee & Task Management SaaS

This is one of the largest systems you've repeatedly planned:

Flutter frontend
Go backend
PostgreSQL database
WebSocket-based real-time updates
Development Style

From our conversations, a few patterns stand out:

You prefer understanding how things work internally rather than using black-box solutions.
You often request optimized implementations instead of quick fixes.
You regularly debug complex issues yourself before asking for help.
You are comfortable reading long logs and stack traces.
You usually prefer production-oriented solutions rather than tutorial-level examples.
Linux and Infrastructure

You frequently work with:

Ubuntu/Linux servers
Docker
Cloud deployments
VPS management
Disk cleanup
Container management

Recently you've been troubleshooting:

Docker storage usage
Containerd disk consumption
Cloud deployment configurations
What You Seem Most Interested In

If I had to summarize your strongest recurring interests, they would be:

Flutter Development
Go Backend Development
Real-Time Communication Systems (WebSocket/WebRTC)
SaaS Product Architecture
PostgreSQL Database Design
Machine Learning and Data Science
Linux, Docker, and Deployment Infrastructure

Overall, you come across as someone who started primarily as a Flutter developer and has gradually expanded into backend engineering, system design, DevOps, databases, and machine learning, with a strong preference for building complete end-to-end products rather than isolated components
"""

model = SentenceTransformer("all-MiniLM-L6-v2")
cur, conn =vectordb()
def insert():
    create_db(cur,conn)
    for text in texts.split("\n"):
        embeddings=gen_embeddings(model,text)
        insert_to_db(conn,texts,embeddings)

def fetch(query):
    
    sql_query= """
    SELECT id, content, category, retrieval_count,
       vector_distance_cos(embedding, vector32(?)) AS distance
FROM memories
WHERE embedding IS NOT NULL
ORDER BY distance ASC
"""
    embeddings = gen_embeddings(model,query)
    cur = conn.execute(sql_query, (encoder.encode(embeddings),))
    print(cur.fetchall()[0])
insert()
fetch("Who is Achiket Kumar? ")