from neo4j import GraphDatabase
import json
import openai
import requests

# Load JSON data
with open('final_data_1.json', 'r') as file:
    data = json.load(file)

# Neo4j connection details
uri = "bolt://localhost:7687"
user = ""
password = ""

# List of colors
colors = ['teal', 'pink', 'yellow', 'blue', 'gray', 'red', 'purple', 'charcoal', 'corduroy', 'black', 'green',
          'lavender', 'strawberry', 'brown', 'white', 'orange', 'lilac', 'peach']

# List of age groups
age_groups = ['3-7 months', '3-6 months', '0-3 months', '6-9 months', '9-12 months']

tags = ['comfortable baby clothes', 'baby fashion', 'baby romper', 'baby clothing', 'baby clothes',
        'comfortable baby wear', 'baby outfit', 'baby onesie', 'infant clothing', 'newborn outfit', 'newborn clothes',
        'kids fashion', 'cotton onesie', 'cute baby clothes', 'easy diaper changes', 'short sleeve romper',
        'striped romper', 'soft baby apparel', 'toddler clothing', 'baby dress', 'baby girl outfit', 'cute baby dress',
        'girls dress', 'summer dress', 'soft baby clothes', 'baby girl dress', 'baby sleepwear', 'comfortable dress',
        'toddler dress', 'casual wear', 'breathable fabric', 'green and white romper', 'baby wardrobe essentials',
        'infant bodysuit', 'striped baby romper', '"comfortable baby clothes"', 'boys ethnic outfit',
        'boys festival clothing', 'boys traditional dress', 'comfortable ethnic attire', 'cultural event outfit',
        'embroidered vest for boys', 'high quality boys clothes', 'special occasion wear', 'traditional boys clothing']

openai.api_key = ""


def get_image_features(image_url: str, base_url: str = ""):
    try:
        params = {'image_url': image_url}
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return json.dumps(data['features'])

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


def get_image_features_resnet(image_url: str, base_url: str = ""):
    try:
        params = {'image_url': image_url}
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return json.dumps(data['embeddings'])

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


def get_embedding(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response['data'][0]['embedding']


# Function to create nodes and relationships in Neo4j
def create_graph(tx, products):
    # Create age group nodes
    for age_group in age_groups:
        tx.run("""
            MERGE (a:AgeGroup {name: $name, embedded: $embedded})
        """, name=age_group, embedded=get_embedding(age_group))

    for product in products:
        # Create product node
        tx.run("""
            MERGE (p:Product {
                title: $title, 
                description: $description, 
                product_type: $product_type, 
                handle: $handle,
                status: $status, 
                tags: $tags,
                embedded_title: $embedded_title,
                embedded_description: $embedded_description,
                embedded_tags: $embedded_tags,
                image : $image,
                embedded_image: $embedded_image,
                embedded_image_resnet : $embedded_image_resnet
            })
        """, title=product['title'], description=product['description'],
               product_type=product['product_type'], handle=product['handle'],
               status=product['status'], tags=product['tags'],
               embedded_title=product['Embedded Title'],
               embedded_description=product['Embedded Description'],
               embedded_tags=product['Embedded Tags'],
               image=product['images'][0]['src'] if len(product['images']) != 0 else "",
               embedded_image=get_image_features(product['images'][0]['src']) if len(product['images']) != 0 else "",
               embedded_image_resnet=get_image_features_resnet(product['images'][0]['src']) if len(product['images']) != 0 else "")

        # Create relationships to age group nodes
        for variant in product['variants']:
            if variant['title'] in age_groups:
                tx.run("""
                    MATCH (a:AgeGroup {name: $name})
                    MERGE (p:Product {title: $product_title})
                    MERGE (p)-[:HAS_AGE_GROUP]->(a)
                """, name=variant['title'], product_title=product['title'])

        # Create color nodes and relationships
        if 'colors' in product:
            for color in product['colors']:
                if color in colors:
                    tx.run("""
                        MERGE (c:Color {
                            name: $name,
                            embedded_color: $embedded_color
                        })
                        MERGE (p:Product {title: $product_title})
                        MERGE (p)-[:HAS_COLOR]->(c)
                    """, name=color, product_title=product['title'],
                           embedded_color=product['Embedded Color'].get(color, []))

        # Create tag nodes and relationships
        product_tags = product['tags'].split(', ')
        for tag in product_tags:
            if tag in tags:
                tx.run("""
                    MERGE (t:Tag {
                        name: $name,
                        embedded_tag: $embedded_tag
                    })
                    MERGE (p:Product {title: $product_title})
                    MERGE (p)-[:HAS_TAG]->(t)
                """, name=tag, product_title=product['title'],
                       embedded_tag=get_embedding(tag))


# Connect to Neo4j and create the graph
driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as session:
    session.execute_write(create_graph, data)

driver.close()
