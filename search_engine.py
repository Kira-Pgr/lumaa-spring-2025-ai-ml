# search_engine.py
# The main logic for the search engine
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse


class SentenceTransformerSearchEngine:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5"):
        """Load the sentence transformer model.
        
        Args:
            model_name (str): The name of the model to use
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.embeddings = None
        self.dataset = None

    def load_data(self, csv_path):
        """Load and prepare the dataset from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing movie data
        """
        self.dataset = pd.read_csv(csv_path)
        documents = []
        for title, summary in zip(self.dataset['Title'], self.dataset['PlotSummary']):
            documents.append(f"search_document: {title} {summary}")

        self.embeddings = self.model.encode(documents, convert_to_tensor=True)

    def search(self, query, top_k=5):
        """Search for movies using the query.
        
        Args:
            query (str): The search query
            top_k (int): The number of top results to return
            
        Returns:
            tuple: (DataFrame with results, similarity scores)
        """
        if self.embeddings is None or self.dataset is None:
            raise ValueError("Dataset not loaded.")

        query_embedding = self.model.encode(
            f"search_query: {query}", convert_to_tensor=True)

        similarities = cosine_similarity(
            query_embedding.cpu().unsqueeze(dim=0).numpy(),
            self.embeddings.cpu().numpy()
        ).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.dataset.iloc[top_indices], similarities[top_indices]


def print_search_results(results_df, sim_scores):
    """Print formatted search results.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing search results
        sim_scores (np.array): Array of similarity scores
    """
    print("Search Results:")
    for idx, score in zip(results_df.index, sim_scores):
        print(
            f"Title: {results_df.loc[idx, 'Title']}, "
            f"Summary: {results_df.loc[idx, 'PlotSummary']}, "
            f"Similarity: {score:.3f}"
        )


def main():
    """run the search engine with command line arguments."""
    parser = argparse.ArgumentParser(
        description='Search for movies using natural language queries')
    parser.add_argument('-q', '--query', type=str,
                        required=True, help='Search query')
    parser.add_argument('-k', '--top_k', type=int, default=5,
                        help='Number of results to return (default: 5)')
    parser.add_argument('-d', '--dataset', type=str, default='dataset.csv',
                        help='Path to the dataset CSV file (default: dataset.csv)')

    # Parse arguments
    args = parser.parse_args()

    # Initialize and run search engine
    search_engine = SentenceTransformerSearchEngine()
    search_engine.load_data(args.dataset)
    results_df, sim_scores = search_engine.search(args.query, args.top_k)
    print_search_results(results_df, sim_scores)


if __name__ == "__main__":
    main()
