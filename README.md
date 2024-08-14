# Product Chatbot

### Description
1. The data is stored in Neo4j GraphDB, the data onsistes of cloth products, in Neo4j-DB/dataUpload.py relations between nodes are created and uploaded.

2. We can perform a prompt and image search, we are using OpenAI's embedding tool to perform vector similarity search.
We are using OpenAI's CLIP model and Microsoft's ResNET model to perform a vector image search.

3. A chatbot is created to facilitate conversation between user and the data stored, we are using OpenAI's GPT and Langchain to handle the conversations. For contextual understanding, we are using MilvusDB( Vector Database) to store the conversattions between the user and the bot to maintain continuous conversations.

### Tools
1. OpenAI: CLIP, GPT-4o, ADA-002
2. Microdoft: ResNET
3. Database: MongoDB, Neo4j-GraphDB, Milvus-Vector-DB
4. Languages: Python, Cypher
5. UI: Streamlit
