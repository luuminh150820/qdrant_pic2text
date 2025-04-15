import os
import base64
import re
import io
import time
import uuid
import traceback
import shutil  # Added for directory removal
import numpy as np
import google.generativeai as genai
from qdrant_client import QdrantClient, models
from PIL import Image, ImageOps

# Configuration
OUTPUT_FOLDER = "images"
QDRANT_COLLECTION = "pdf_collection"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OUTPUT_TEXT_FILE = "extracted_text.txt"
VECTOR_SIZE = 768
MAX_CHUNK_SIZE = 1000
MIN_SIMILARITY_SCORE = 0.7
BATCH_SIZE = 5  # Process 5 images at once
MAX_IMAGE_SIZE = (800, 800)  # Reduced size for faster processing

# Path for local Qdrant storage
QDRANT_LOCAL_PATH = os.path.join(os.getcwd(), "qdrant_storage")

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel("gemini-2.0-flash")

def preprocess_image(image_path):
    """Preprocess image for better OCR performance"""
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if larger than MAX_IMAGE_SIZE
        if img.width > MAX_IMAGE_SIZE[0] or img.height > MAX_IMAGE_SIZE[1]:
            img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        img_gray = ImageOps.grayscale(img)
        
        # Enhance contrast
        img_gray = ImageOps.autocontrast(img_gray)
        
        # Save to buffer
        buffer = io.BytesIO()
        img_gray.save(buffer, format="PNG")
        buffer.seek(0)
        
        return buffer.getvalue(), 'image/png'
    except Exception as e:
        # If preprocessing fails, return the original image
        with open(image_path, "rb") as f:
            return f.read(), f'image/{os.path.splitext(image_path)[1][1:].lower()}'

def get_embedding(text):
    """Generate embedding for text using the correct Gemini embedding API"""
    if not text or len(text.strip()) == 0:
        return [0.0] * VECTOR_SIZE
    
    try:
        # Use the correct embedding model name format with "models/" prefix
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        
        # The result is now a dictionary structure, not an object with embedding attribute
        # Access the embedding values using dictionary methods
        if isinstance(result, dict) and 'embedding' in result:
            embedding_list = list(map(float, result['embedding']))
            
            # Check if the vector has the right dimension
            if len(embedding_list) != VECTOR_SIZE:
                # If too short, pad with zeros
                if len(embedding_list) < VECTOR_SIZE:
                    embedding_list.extend([0.0] * (VECTOR_SIZE - len(embedding_list)))
                # If too long, truncate
                else:
                    embedding_list = embedding_list[:VECTOR_SIZE]
                    
            return embedding_list
        elif isinstance(result, dict) and 'embeddings' in result:
            # Try plural form
            embedding_list = list(map(float, result['embeddings']))
            if len(embedding_list) != VECTOR_SIZE:
                if len(embedding_list) < VECTOR_SIZE:
                    embedding_list.extend([0.0] * (VECTOR_SIZE - len(embedding_list)))
                else:
                    embedding_list = embedding_list[:VECTOR_SIZE]
            return embedding_list
        else:
            # Fall back to zero vector
            return [0.0] * VECTOR_SIZE
    except Exception as e:
        return [0.0] * VECTOR_SIZE

def split_into_sentences(text):
    """Split text into sentences using regex pattern"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def recursive_chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE):
    """Chunk text recursively while preserving semantic boundaries"""
    if not text:
        return []

    # First try to split by paragraphs
    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        elif len(paragraph) > max_chunk_size:
            # If paragraph is too large, split into sentences
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            sentences = split_into_sentences(paragraph)
            sentence_chunk = ""

            for sentence in sentences:
                if len(sentence_chunk) + len(sentence) > max_chunk_size and sentence_chunk:
                    chunks.append(sentence_chunk)
                    sentence_chunk = sentence
                else:
                    separator = " " if sentence_chunk else ""
                    sentence_chunk += separator + sentence

            if sentence_chunk:
                current_chunk = sentence_chunk
        else:
            separator = "\n\n" if current_chunk else ""
            current_chunk += separator + paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def process_image_batch(image_paths):
    """Process a batch of images at once using Gemini"""
    if not image_paths:
        return {}
        
    try:
        # Prepare batch of images
        parts = []
        
        # First, add all the images with preprocessing
        for image_path in image_paths:
            image_data, mime_type = preprocess_image(image_path)
            parts.append({"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_data).decode('utf-8')}})
        
        # Add the prompt text as the last part
        parts.append({"text": "Act like a text scanner. Extract text from each image in order, separated by '---IMAGE BOUNDARY---'. Extract text as it is without analyzing it and without summarizing it."})
        
        # Make a single API call for all images
        response = generation_model.generate_content(parts)
        text_results = response.text.strip()
        
        # Split the response by image boundary marker
        text_segments = text_results.split('---IMAGE BOUNDARY---')
        
        # Clean up and create results dictionary
        results = {}
        for i, (image_path, text) in enumerate(zip(image_paths, text_segments)):
            # Clean up any extra whitespace
            cleaned_text = text.strip()
            if cleaned_text:
                results[image_path] = cleaned_text
                
        return results
    except Exception as e:
        # Fall back to processing images individually
        results = {}
        for image_path in image_paths:
            try:
                text = process_single_image(image_path)
                if text:
                    results[image_path] = text
            except Exception:
                pass
        return results

def process_single_image(image_path):
    """Process a single image if batch processing fails"""
    try:
        image_data, mime_type = preprocess_image(image_path)
        
        parts = [
            {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_data).decode('utf-8')}},
            {"text": "Act like a text scanner. Extract text as it is without analyzing it and without summarizing it."}
        ]

        response = generation_model.generate_content(parts)
        return response.text.strip()
    except Exception:
        return ""

def setup_qdrant_collection(qdrant_client):
    """Create or reset Qdrant collection"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # Delete existing collection if it exists
        if QDRANT_COLLECTION in collection_names:
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)
            # Verify deletion
            collections_after = qdrant_client.get_collections().collections
            if QDRANT_COLLECTION in [c.name for c in collections_after]:
                print(f"Warning: Failed to delete collection {QDRANT_COLLECTION}")
                return False
        
        # Create collection with correct vector configuration
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        return True
    except Exception as e:
        print(f"Error setting up collection: {e}")
        return False

def store_chunks_in_qdrant(qdrant_client, chunks, metadata):
    """Store text chunks and their embeddings in Qdrant"""
    try:
        # Process in smaller batches to avoid memory issues
        batch_size = 10
        total_chunks = len(chunks)
        stored_count = 0
        
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            
            points = []
            for i, chunk in enumerate(batch_chunks):
                if not chunk.strip():
                    continue
                    
                # Get embedding for chunk
                embedding = get_embedding(chunk)
                
                # Debug check - make sure embedding is a list of floats
                if not isinstance(embedding, list):
                    # Try to convert it to a list
                    try:
                        embedding = list(embedding)
                    except Exception:
                        # Skip this chunk
                        continue
                
                # Additional type check for each element
                for j, val in enumerate(embedding):
                    if not isinstance(val, (int, float)):
                        try:
                            embedding[j] = float(val)
                        except:
                            embedding[j] = 0.0
                
                # Create a valid UUID for the point ID
                point_id = str(uuid.uuid4())  # Always use UUID format
                
                # Create point
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,  # Now properly converted to list of floats
                        payload={
                            "text": chunk,
                            "source_image": metadata.get("source_image", "unknown"),
                            "chunk_index": batch_start + i,
                            "chunk_level": metadata.get("chunk_level", "document"),
                            "original_id": f"{metadata.get('id_prefix', 'chunk')}_{batch_start + i}"  # Store original ID in payload
                        }
                    )
                )
            
            # Upload batch to Qdrant
            if points:
                qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points
                )
                stored_count += len(points)
                
                # Small delay to avoid API rate limits if needed
                time.sleep(0.5)
        
        return stored_count
    except Exception:
        return 0

def advanced_search(qdrant_client, query, filters=None, min_score=MIN_SIMILARITY_SCORE, limit=5):
    """Search with filtering and score thresholds"""
    try:
        query_embedding = get_embedding(query)
        
        # Check if embedding is all zeros or has valid values
        is_zeros = all(v == 0.0 for v in query_embedding)
        
        if not query_embedding or is_zeros:
            return []

        # Create filter conditions
        filter_condition = None
        if filters:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
            )

        # Try a search with lower threshold first
        temp_min_score = 0.0  # No threshold to see if we get any results
        
        # Search with filters
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=limit,
            query_filter=filter_condition,
            score_threshold=temp_min_score
        )
        
        # Apply score threshold and return results
        filtered_results = [
            {
                "text": r.payload["text"],
                "score": r.score,
                "source_image": r.payload["source_image"],
                "chunk_index": r.payload["chunk_index"]
            }
            for r in search_results if r.score >= min_score
        ]
        
        return filtered_results
    except Exception:
        return []

def retrieve_and_generate(qdrant_client, query, filters=None):
    """Retrieve relevant context and generate an answer"""
    try:
        results = advanced_search(qdrant_client, query, filters=filters)
        if not results:
            return "No relevant information found."

        # Combine results into context
        context = "\n\n".join([f"[From {r['source_image']}, Score: {r['score']:.2f}]\n{r['text']}" for r in results])
        
        # More explicit prompt to avoid hallucinations
        prompt = f"""
        Based ONLY on the following context, answer this question: {query}
        
        Context:
        {context}
        
        IMPORTANT: Only use information that is explicitly mentioned in the context above. 
        Do not add any information from your training data or other sources.
        If the information is not present in the context, say "The information is not available in the provided context."
        Please provide a concise, accurate answer based only on the information in the context.
        """

        response = generation_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def verify_collection_content(qdrant_client, collection_name):
    """Verify what's actually stored in the collection"""
    try:
        print(f"Checking content in '{collection_name}' collection...")
        result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        
        points = result[0]
        if not points:
            print("Collection is empty.")
            return
        
        print(f"Found {len(points)} sample points:")
        for i, point in enumerate(points):
            source = point.payload.get("source_image", "unknown")
            text_preview = point.payload.get("text", "")[:50] + "..." if point.payload.get("text") else "No text"
            print(f"  {i+1}. From {source}: {text_preview}")
    except Exception as e:
        print(f"Error checking collection: {e}")

if __name__ == "__main__":
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all images
    image_paths = [
        os.path.join(OUTPUT_FOLDER, filename)
        for filename in os.listdir(OUTPUT_FOLDER)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'))
    ]
    
    if not image_paths:
        print(f"No image files found in the '{OUTPUT_FOLDER}' folder.")
        exit(1)
    
    print(f"Processing {len(image_paths)} images in batches of {BATCH_SIZE}...")
    
    extracted_texts = {}
    
    # Create batches
    batches = [image_paths[i:i+BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
    
    # Process each batch
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} images)...")
        batch_results = process_image_batch(batch)
        extracted_texts.update(batch_results)
    
    # Show completion status
    successful_count = len(extracted_texts)
    print(f"✓ Completed text extraction: {successful_count}/{len(image_paths)} images processed successfully")

    if not extracted_texts:
        print("No text extracted from images.")
        exit(1)

    # Save combined text to file
    combined_text = "\n\n".join([f"[From {os.path.basename(img_path)}]\n{text}" for img_path, text in extracted_texts.items()])
    with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"✓ Extracted text saved to {OUTPUT_TEXT_FILE}")
    
    # COMPLETE DATABASE RESET - Remove entire Qdrant directory to ensure fresh start
    if os.path.exists(QDRANT_LOCAL_PATH):
        print(f"Cleaning up previous database at {QDRANT_LOCAL_PATH}...")
        shutil.rmtree(QDRANT_LOCAL_PATH)
    
    # Initialize Qdrant client with local storage
    os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
    qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)
    
    # Setup collection
    if not setup_qdrant_collection(qdrant_client):
        print("Failed to set up Qdrant collection.")
        exit(1)
    
    print("Processing text chunks and storing in Qdrant...")
    total_chunks = 0
    
    # Process each image's text separately
    for image_path, text in extracted_texts.items():
        source_image = os.path.basename(image_path)
        
        # Create semantic chunks with improved chunking algorithm
        chunks = recursive_chunk_text(text)
        
        # Store chunks in Qdrant with metadata
        metadata = {
            "source_image": source_image,
            "chunk_level": "semantic",
            "id_prefix": f"img_{image_paths.index(image_path)}"
        }
        
        stored_chunks = store_chunks_in_qdrant(qdrant_client, chunks, metadata)
        total_chunks += stored_chunks
    
    print(f"✓ Completed storing chunks in Qdrant")
    
    # Verify what's actually in the collection after storing
    verify_collection_content(qdrant_client, QDRANT_COLLECTION)
    
    # Run a sample query
    if total_chunks > 0:
        print("\n--- Sample Query Results ---")
        query = "What is the main topic of these documents?"
        print(f"Query: {query}")
        answer = retrieve_and_generate(qdrant_client, query)
        print(f"Answer: {answer}")
        
        # Example filtered search for a specific image
        if len(image_paths) > 1:
            source_image = os.path.basename(image_paths[0])
            print(f"\n--- Filtered search for image '{source_image}' ---")
            source_filter = {"source_image": source_image}
            filtered_answer = retrieve_and_generate(qdrant_client, query, filters=source_filter)
            print(f"Filtered answer: {filtered_answer}")
    else:
        print("No chunks were stored in Qdrant. Exiting.")