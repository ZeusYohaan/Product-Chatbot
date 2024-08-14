from fastapi import FastAPI, HTTPException, Query
from neo4j import GraphDatabase
import openai
import numpy as np
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

app = FastAPI()
openai.api_key = ""

uri = "bolt://localhost:7687"
user = ""
password = "password"
driver = GraphDatabase.driver(uri, auth=(user, password))


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


def find_most_similar_product_prompt(prompt):
    prompt_embedding = get_embedding(prompt)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        products = session.execute_read(fetch_product_and_connected_nodes)

    driver.close()

    similarities = []

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

        similarity = sk_cosine_similarity([prompt_embedding], [combined_embedding])[0][0]
        similarities.append((similarity, {
            'title': product.get('title'),
            'description': product.get('description'),
            'product_type': product.get('product_type'),
            'handle': product.get('handle'),
            'status': product.get('status'),
            'tags': product.get('tags'),
            'image': product.get('image'),
            'age_groups': list(set([a.get('name') for a in age_groups])),
            'colors': list(set([c.get('name') for c in colors])),
            'tags': list(set([t.get('name') for t in tags]))
        }))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_2_similar_products = [sim[1] for sim in similarities[:2]]
    return top_2_similar_products


# Define your API and Neo4j connection details
def get_image_features(image_url: str, base_url: str = ""):
    try:
        params = {'image_url': image_url}
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return np.array(data['features']).flatten()  # Convert to numpy array and flatten

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        raise HTTPException(status_code=500, detail="Image feature extraction failed.")
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise HTTPException(status_code=500, detail="Image feature extraction failed.")


def get_image_features_resnet(image_url: str, base_url: str = ""):
    try:
        params = {'image_url': image_url}
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return np.array(data['embeddings'])  # Convert to numpy array directly

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        raise HTTPException(status_code=500, detail="Image feature extraction failed.")
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise HTTPException(status_code=500, detail="Image feature extraction failed.")


def cosine_similarity(vec1, vec2):
    """ Compute the cosine similarity between two vectors using scikit-learn. """
    vec1 = vec1.reshape(1, -1)  # Reshape for sklearn compatibility
    vec2 = vec2.reshape(1, -1)  # Reshape for sklearn compatibility
    return sk_cosine_similarity(vec1, vec2)[0][0]


def find_most_similar_product(driver, image_features):
    """ Find the top 2 products with the most similar image embedding in Neo4j. """
    with driver.session() as session:
        # Fetch all product nodes with image embeddings
        result = session.run("""
            MATCH (p:Product)
            RETURN p.title AS title, p.embedded_image AS embedded_image, p.image AS image, p.handle AS handle
        """)

        similarities = []

        for record in result:
            product_title = record['title']
            embedded_image = record['embedded_image']
            image = record['image']
            handle = record['handle']

            if embedded_image:  # Check if embedded_image is not empty or None
                try:
                    stored_embedding = np.array(json.loads(embedded_image)).flatten()
                    similarity = cosine_similarity(image_features, stored_embedding)

                    similarities.append((similarity, [product_title, image, handle]))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error decoding embedded image for product '{product_title}': {e}")
            else:
                pass

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_2_similar_products = [sim[1] for sim in similarities[:2]]
        return top_2_similar_products


def find_most_similar_product_resnet(driver, image_features):
    """ Find the top 2 products with the most similar image embedding in Neo4j. """
    with driver.session() as session:
        # Fetch all product nodes with image embeddings
        result = session.run("""
            MATCH (p:Product)
            RETURN p.title AS title, p.embedded_image_resnet AS embedded_image, p.image AS image, p.handle AS handle
        """)

        similarities = []
        for record in result:
            product_title = record['title']
            embedded_image = record['embedded_image']
            image = record['image']
            handle = record['handle']

            if embedded_image:  # Check if embedded_image is not empty or None
                try:
                    # Convert the JSON string to a numpy array
                    stored_embedding = np.array(json.loads(embedded_image)).flatten()
                    similarity = cosine_similarity(image_features, stored_embedding)

                    similarities.append((similarity, [product_title, image, handle]))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error decoding embedded image for product '{product_title}': {e}")
            else:
                pass

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_2_similar_products = [sim[1] for sim in similarities[:2]]
        return top_2_similar_products


@app.get("/find-similar-product-prompt/")
async def find_similar_product_prompt(prompt: str = Query(..., description="The URL of the image to compare")):
    product = find_most_similar_product_prompt(prompt)
    return {"gpt_response": "success", "products": product}


@app.get("/find-similar-product-image/")
async def find_similar_product(image_url: str = Query(..., description="The URL of the image to compare")):
    image_features = get_image_features(image_url)
    most_similar_products = find_most_similar_product(driver, image_features)

    if not most_similar_products:
        raise HTTPException(status_code=404, detail="No similar products found.")

    return {
        "gpt_response": "success",
        "products": [{"product_title": product[0], "image": product[1], "handle": product[2]} for product in
                     most_similar_products]
    }


@app.get("/find-similar-product-image-resnet/")
async def find_similar_product(image_url: str = Query(..., description="The URL of the image to compare")):
    image_features = get_image_features_resnet(image_url)
    most_similar_products = find_most_similar_product_resnet(driver, image_features)

    if not most_similar_products:
        raise HTTPException(status_code=404, detail="No similar products found.")

    return {
        "gpt_response": "success",
        "products": [{"product_title": product[0], "image": product[1], "handle": product[2]} for product in
                     most_similar_products]
    }


@app.on_event("shutdown")
def shutdown_event():
    driver.close()
