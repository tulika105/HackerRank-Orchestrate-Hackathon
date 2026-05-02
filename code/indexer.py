import os
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

class HybridIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.documents = []
        self.metadata = []
        self.faiss_index = None
        self.sparse_matrix = None

    def load_data(self, data_dir: str):
        """Recursively loads all .md files from the data directory."""
        print(f"Loading data from {data_dir}...")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    # Extract company from path
                    company = "None"
                    if "hackerrank" in root.lower():
                        company = "HackerRank"
                    elif "claude" in root.lower():
                        company = "Claude"
                    elif "visa" in root.lower():
                        company = "Visa"
                    
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Split by double newlines into paragraphs
                        paragraphs = content.split('\n\n')
                        for para in paragraphs:
                            para = para.strip()
                            # Skip if less than 30 words
                            if len(para.split()) >= 30:
                                self.documents.append(para)
                                self.metadata.append({
                                    'path': path,
                                    'filename': file,
                                    'company': company
                                })
        print(f"Loaded {len(self.documents)} documents.")

    def fit(self):
        """Fits TF-IDF and generates Dense Embeddings with FAISS."""
        if not self.documents:
            print("No documents loaded.")
            return

        print("Generating TF-IDF sparse matrix...")
        self.sparse_matrix = self.tfidf.fit_transform(self.documents)

        print("Generating Dense embeddings and building FAISS index...")
        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        embeddings = embeddings.astype('float32')
        # Normalize for Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        # Initialize FAISS index for Inner Product (Cosine Similarity on normalized vectors)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

    def search_dense(self, query: str, top_k: int = 10) -> List[int]:
        """Performs dense vector search using FAISS (Cosine Similarity)."""
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return indices[0].tolist()

    def search_sparse(self, query: str, top_k: int = 10) -> List[int]:
        """Performs sparse TF-IDF search."""
        query_vector = self.tfidf.transform([query])
        similarities = (self.sparse_matrix * query_vector.T).toarray().flatten()
        return np.argsort(similarities)[::-1][:top_k].tolist()

    def rrf(self, dense_results: List[int], sparse_results: List[int], k: int = 60) -> List[int]:
        """Reciprocal Rank Fusion."""
        scores = {}
        for rank, doc_idx in enumerate(dense_results):
            scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (k + rank + 1)
        for rank, doc_idx in enumerate(sparse_results):
            scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (k + rank + 1)
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in sorted_docs]

    def hybrid_search(self, query: str, company: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Combines dense and sparse search with filtering."""
        # Get more results for fusion
        d_results = self.search_dense(query, top_k=50)
        s_results = self.search_sparse(query, top_k=50)
        
        fused_indices = self.rrf(d_results, s_results)
        
        results = []
        for idx in fused_indices:
            meta = self.metadata[idx]
            # Filter by company if specified
            if company and company != "None" and meta['company'] != company:
                continue
            
            results.append({
                'content': self.documents[idx],
                'metadata': meta
            })
            if len(results) >= top_k:
                break
                
        return results

if __name__ == "__main__":
    indexer = HybridIndexer()
    indexer.load_data("../data")
    indexer.fit()
    # Test
    results = indexer.hybrid_search("How to delete account?", company="HackerRank")
    for r in results:
        print(f"File: {r['metadata']['filename']} | Company: {r['metadata']['company']}")
