import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage
import openai
from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
openai.api_key = os.getenv('OPENAI_API_KEY')
uri = "bolt://localhost:7687"
user = ""
password = ""

# Connect to Milvus server
connections.connect(alias='default', host='localhost', port='19530')

# Define schema for the collection
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='user_id', dtype=DataType.INT64),
    FieldSchema(name='session_id', dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1536),  # Adjust dimension based on embedding model
    FieldSchema(name='message', dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name='role', dtype=DataType.VARCHAR, max_length=50)
]

schema = CollectionSchema(fields, description="Conversation data")
collection_name = 'conversations'
collection = Collection(name=collection_name, schema=schema)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def get_embedding(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response['data'][0]['embedding']


def find_most_similar_product(prompt):
    prompt_embedding = get_embedding(prompt)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        products = session.execute_read(fetch_product_and_connected_nodes)

    driver.close()

    max_similarity = -1
    most_similar_product = None

    for record in products:
        product = record['product']
        age_groups = record['age_groups']
        colors = record['colors']
        tags = record['tags']

        title_embedding = np.array(product.get('embedded_title', []))
        description_embedding = np.array(product.get('embedded_description', []))
        tags_embedding = np.array(product.get('embedded_tags', []))

        age_group_embeddings = np.mean([np.array(a.get('embedded', [])) for a in age_groups],
                                       axis=0) if age_groups else np.zeros_like(title_embedding)
        color_embeddings = np.mean([np.array(c.get('embedded_color', [])) for c in colors],
                                   axis=0) if colors else np.zeros_like(title_embedding)
        tag_embeddings = np.mean([np.array(t.get('embedded_tag', [])) for t in tags],
                                 axis=0) if tags else np.zeros_like(title_embedding)

        combined_embedding = np.mean([
            title_embedding,
            description_embedding,
            tags_embedding,
            age_group_embeddings,
            color_embeddings,
            tag_embeddings
        ], axis=0)

        similarity = cosine_similarity([prompt_embedding], [combined_embedding])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_product = {
                'title': product.get('title'),
                'description': product.get('description'),
                'product_type': product.get('product_type'),
                'handle': product.get('handle'),
                'status': product.get('status'),
                'tags': product.get('tags'),
                'image': product.get('image'),
                'age_groups': [a.get('name') for a in age_groups],
                'colors': [c.get('name') for c in colors],
                'tags': [t.get('name') for t in tags]
            }
    return most_similar_product


def fetch_product_and_connected_nodes(tx):
    query = """
    MATCH (p:Product)
    OPTIONAL MATCH (p)-[:HAS_AGE_GROUP]->(a:AgeGroup)
    OPTIONAL MATCH (p)-[:HAS_COLOR]->(c:Color)
    OPTIONAL MATCH (p)-[:HAS_TAG]->(t:Tag)
    RETURN p AS product,
           COLLECT(a) AS age_groups,
           COLLECT(c) AS colors,
           COLLECT(t) AS tags
    """
    result = tx.run(query)
    products = []
    for record in result:
        products.append({
            'product': record['product'],
            'age_groups': record['age_groups'],
            'colors': record['colors'],
            'tags': record['tags']
        })
    return products


def save_conversation(user_id, session_id, embedding, message, role):
    entities = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "embedding": embedding,
            "message": message,
            "role": role
        }
    ]
    collection.insert(entities)
    collection.flush()


def retrieve_conversations(session_id):
    query = f"session_id == '{session_id}'"
    results = collection.query(query, output_fields=['user_id', 'session_id', 'message', 'role'])
    return results


@tool
def product_search(prompt) -> str:
    """Tool to search for a product based on a text prompt."""
    most_similar_product = find_most_similar_product(prompt)
    if most_similar_product:
        return json.dumps(most_similar_product)
    else:
        return "No similar product found."


tools = [product_search]

model_with_tools = llm.bind_tools(tools)
tools_map = {tool.name: tool for tool in tools}


def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()

    prev_output = None

    for tool_call in tool_calls:
        if prev_output is not None:
            tool_call["args"] = prev_output
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        prev_output = tool_call["output"]
    return tool_calls


chain = model_with_tools | call_tools


def main():
    st.title('Product Search Chat')

    # Sidebar for session selection
    session_id = st.sidebar.text_input("Session ID", "")
    user_id = 1  # Placeholder for user ID, you might want to fetch it dynamically

    if st.sidebar.button("Load Conversation"):
        messages = retrieve_conversations(session_id)
        for message in messages:
            st.write(f"{message['role']}: {message['message']}")

    user_question = st.text_input('Ask a question about a product:')

    if st.button('Submit'):
        response = chain.invoke([HumanMessage(content=user_question)])
        for i, step in enumerate(response, 1):
            st.write(f"function #{i} : {step['name']}({step['args']})")
            st.write(f"Output #{i} : {step['output']}\n")

            # Save conversation to Milvus
            embedding = get_embedding(step['output'])
            save_conversation(user_id, session_id, embedding, step['output'], "system")

        # Save user question to Milvus
        user_embedding = get_embedding(user_question)
        save_conversation(user_id, session_id, user_embedding, user_question, "user")

        st.write("Conversation saved!")


if __name__ == "__main__":
    main()
