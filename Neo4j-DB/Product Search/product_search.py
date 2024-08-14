import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import openai
import json

openai.api_key = ""

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"


def get_embedding(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response['data'][0]['embedding']


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


prompt = "Adorable Baby Girl Floral Dress with Peach Shrug"
most_similar_product = find_most_similar_product(prompt)
if most_similar_product:
    print("Most similar product details:")
    print(f"Title: {most_similar_product['title']}")
    print(f"Description: {most_similar_product['description']}")
    print(f"Product Type: {most_similar_product['product_type']}")
    print(f"Handle: {most_similar_product['handle']}")
    print(f"Images: {most_similar_product['image']}")
    print(f"Status: {most_similar_product['status']}")
    print(f"Tags: {most_similar_product['tags']}")
    print(f"Age Groups: {most_similar_product['age_groups']}")
    print(f"Colors: {most_similar_product['colors']}")
    print(f"Shopify URL: https://2a78a1-f6.myshopify.com/products/{most_similar_product['handle']}\n")
else:
    print("No similar product found.")
