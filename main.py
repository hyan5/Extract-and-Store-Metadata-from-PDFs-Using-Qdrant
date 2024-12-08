from flask import Flask, request, render_template, jsonify
import pymupdf
from PIL import Image
import io
import torch
import bitsandbytes as bnb
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from transformers import AutoModel, AutoTokenizer, pipeline
import numpy as np
import pdfplumber
import pandas as pd
import uuid
import base64
import os
import socket
import pdb


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize Qdrant client
qdrant_client = QdrantClient("http://localhost:6333")
collection_name = 'pdf_content'
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config={"default": VectorParams(size=384, distance="Cosine")},
)

# Load models for embeddings
"""
Using sentence transformer as text embedding model.
"""
text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name)
image_captioner = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")

def generate_text_embedding(text):
    """
    Generate embedding for text.
    """
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().tolist()


def generate_image_caption(image, context):
    """
    Generate a caption for an image using ViT-GPT2. 
    The caption is generated based on both the image and relvent context extracted from the pdf.
    """
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": context
                }
            ]
        }
    ]
    outputs = image_captioner(text=message, max_new_tokens=50, return_full_text=False)
    caption = outputs[0]["generated_text"]
    return caption

def process_pdf(pdf_path):
    """
    Extract headers, footers, and content from a PDF, grouping paragraphs under the same
    subtitle into one block, while processing content in order on each page.
    Args:
        pdf_path: the path to the pdf file
    Returns:
        None
    """
    doc = pymupdf.open(pdf_path)

    for page_num, page in enumerate(doc):
        page_height = page.rect.height

        # Collect all content (text, images) with their positions
        content_blocks = []
        grouped_paragraphs = {}  # Store paragraphs grouped by subtitles
        current_subtitle = None  # Track the current subtitle

        # Extract text blocks
        text_blocks = page.get_text("dict")["blocks"]

        for block in text_blocks:
            # Extract image
            if block['type'] == 1:
                content_blocks.append({
                "type": "image",
                "image": block["image"],
                "bbox": block["bbox"]
                })
                continue
            # Extract text
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip().replace("\n", " ")  # Clean up text
                    font_size = span["size"]
                    bbox = span["bbox"]

                    if not text:
                        continue
                    
                    # Ignore the header and footer of the page
                    if bbox[3] < page_height * 0.05 or bbox[1] > page_height * 0.95:
                       continue
                    else:
                        # Detect subtitles by font size
                        if font_size > 13:  # Adjust threshold as needed
                            current_subtitle = text
                            grouped_paragraphs[current_subtitle] = {
                                "content": [],
                                "bbox": bbox,
                            }
                        elif current_subtitle:
                            # Append text to the current subtitle group
                            grouped_paragraphs[current_subtitle]["content"].append(text)

        for para_num, para in enumerate(grouped_paragraphs):
            para = para
            text = " ".join(grouped_paragraphs[para]['content'])
            bbox = grouped_paragraphs[para]['bbox']
            if text:
                para = f"{para} \n{text}"
            content_blocks.append({
                "type": "paragraph",
                "content": para,
                "bbox": bbox,
                "paragraph_number": para_num + 1
            })

        # Sort all content in order
        content_blocks.sort(key=lambda b: b["bbox"][1])

        # Store content in Qdrant
        for curr_idx, content in enumerate(content_blocks):
            if content["type"] == "paragraph":
                # Generat the embedding for search purpose.
                embeddings = generate_text_embedding(content["content"])
                paragraph = {
                    "content": content["content"],
                    "metadata": {
                        "type": "paragraph",
                        "page_number": page_num + 1,
                        "paragraph_number": content["paragraph_number"],
                    },
                }
                store_in_qdrant(paragraph, embeddings)

            elif content["type"] == "image":
                # Generate a caption using the context text
                context = ""
                if curr_idx != 0 and content_blocks[curr_idx-1]["type"] == 'paragraph':
                    context = content_blocks[curr_idx-1]["content"]
                context = f"Describe this image based on the following context: {context}"
                image = Image.open(io.BytesIO(content['image']))
                caption = generate_image_caption(image, context)

                # Generat the embedding for search purpose.
                embeddings = generate_text_embedding(caption)
                base_image = base64.b64encode(content['image']).decode('utf-8')
                image_data = {
                    "content": caption,
                    "image": base_image,
                    "metadata": {
                        "type": "image",
                        "page_number": page_num + 1
                    },
                }
                store_in_qdrant(image_data, embeddings)

    # Extract all tables
    extract_tables_with_descriptions(pdf_path)

def extract_tables_with_descriptions(pdf_path):
    """
    Extract tables from a PDF along with their descriptions.
    Currently, the description is assumed to be one sentence before the table.
    If the description is empty, the table is treated as an unqulified table.
    Args:
        pdf_path: the path to the pdf file
    Returns:
        None
    """
    tables_with_descriptions = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text from the page
            page_text = page.extract_text()

            # Split the text into lines for better context
            lines = page_text.split("\n") if page_text else []

            # Extract tables from the page
            tables = page.extract_tables()

            for table in tables:
                if not table:
                    continue

                # Convert table into a DataFrame
                df = pd.DataFrame(table[1:], columns=table[0])  # First row as header

                # Find the description (one sentence before the table)
                description = ""
                for i, line in enumerate(lines):
                    if table[0][0] in line:  # Check if table header appears in text
                        if i > 0:
                            description = lines[i - 1].strip()
                        break
                if not description:
                    continue

                embeddings = generate_text_embedding(description)
                table_data = {
                    "content": description,
                    "table": df.to_html(index=False),
                    "metadata": {
                        "type": "table",
                        "page_number": page_num + 1
                    },
                }
                store_in_qdrant(table_data, embeddings)

    return tables_with_descriptions

def store_in_qdrant(data, embedding):
    """
    Store data in Qdrant vector store with embeddings.
    Args:
        data (dict): the extracted content and its metadata.
        embedding (vector): vector embeddings.
    Returns:
        None
    """
    unique_id = str(uuid.uuid4())
    point = PointStruct(id=unique_id, vector={"default": embedding}, payload=data)
    info = qdrant_client.upsert(collection_name=collection_name, points=[point])
    print(info)


def query_qdrant_by_embedding(query, limit=5):
    """
    Query Qdrant for the nearest vectors based on a query embedding.
    Args:
        query_embedding (list): The embedding vector for similarity search.
        limit (int): Number of top results to retrieve.
    Returns:
        list: Results from Qdrant.
    """
    query_embedding = generate_text_embedding(query)
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector={"name": "default", "vector": query_embedding},
        limit=limit,
    )

    return results

def find_available_port(start_port=5000):
    """
    Finds an available port starting from the given port number.
    """
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', start_port))
                return start_port
        except OSError:
            start_port += 1

# Example PDF file to process
# extract_tables_with_descriptions("Algorithms_and_Flowcharts.pdf")

# query = "What is the workflow described in the document?"
# query_embedding = generate_text_embedding(query)

# embedding_results = query_qdrant_by_embedding(query_embedding, limit=5)
# print("Results by Embedding Similarity:")
# for result in embedding_results:
#     print(result.payload)


# Query example
# query_results = qdrant_client.search(
#     collection_name=collection_name,
#     query_vector=[0] * 768,  # Placeholder vector for query embedding
#     limit=5,
# )

# for result in query_results:
#     print(result)

@app.route("/", methods=["GET"])
def index():
    """Render the homepage."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handle PDF upload and parsing."""
    if "pdf" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["pdf"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    process_pdf(file_path)
    return jsonify({"message": "PDF parsed and stored in Qdrant successfully."})

@app.route("/query", methods=["POST"])
def query_qdrant():
    """Handle user queries and format results."""
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Fetch results from Qdrant
    results = query_qdrant_by_embedding(user_query)

    # Format results for better readability
    formatted_results = []
    for result in results:
        payload = result.payload
        content_type = payload.get("metadata", {}).get("type", "unknown")
        page_number = payload.get("metadata", {}).get("page_number", "unknown")

        if content_type == "image":
            # Format image results
            formatted_results.append({
                "type": "Image",
                "content": payload.get("content", "No content available."),
                "image": payload.get("image", ""),
                "page_number": page_number,
            })
        elif content_type == "paragraph":
            # Format paragraph results
            formatted_results.append({
                "type": "Paragraph",
                "content": payload.get("content", "No content available."),
                "page_number": page_number,
                "paragraph_number": payload.get("metadata", {}).get("paragraph_number", "unknown"),
            })
        elif content_type == "table":
            # Format table results
            formatted_results.append({
                "type": "Table",
                "content": payload.get("content", "No description available."),
                "table_html": payload.get("table", ""),
                "page_number": page_number,
            })
    return jsonify({"query": user_query, "results": formatted_results})

if __name__ == "__main__":
    port = find_available_port()
    app.run(host='0.0.0.0', port=port)