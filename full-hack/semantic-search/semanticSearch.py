import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from googletrans import Translator, LANGUAGES
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


# SemanticSearch Class: Manages the backend functionality like data loading, adding, searching, and embeddings.
class SemanticSearch:
    def __init__(self, file_path, model_name='bert-base-uncased'):
        """
        Initialize the SemanticSearch class.

        Args:
        - file_path (str): Path to the dataset file (CSV format).
        - model_name (str): Pretrained BERT model name (default is 'bert-base-uncased').
        """
        self.file_path = file_path  # Path to the CSV dataset
        self.model_name = model_name  # BERT model to be used
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)  # Initialize the BERT tokenizer
        self.model = BertModel.from_pretrained(self.model_name)  # Initialize the BERT model
        self.model.eval()  
        self.df = None  # Placeholder for the dataset (Pandas DataFrame)

        # Load and preprocess the dataset
        self._load_data()

    def _load_data(self):
        """
        Load the dataset and retain only the required columns.

        Ensures compatibility with different file encodings (utf-8, ISO-8859-1, latin1).
        Raises an error if required columns are missing.
        """
        try:
            self.df = pd.read_csv(self.file_path, encoding='utf-8')  
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(self.file_path, encoding='ISO-8859-1')  
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.file_path, encoding='latin1')  

        # Keep only the necessary columns
        required_columns = ['Name', 'Description', 'Price', 'NetQuantity']
        self.df = self.df[required_columns]

        # Ensure all required columns are present in the dataset
        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(f"Missing required column: {column}")

    def add_data(self, name, description, price, netquantity):
        """
        Add a new row of data to the dataset.

        Args:
        - name (str): Name of the product.
        - description (str): Description of the product.
        - price (str): Price of the product.
        - netquantity (str): Quantity of the product.
        """
        # Create a new DataFrame for the row and append it
        new_entry = pd.DataFrame([{
            'Name': name,
            'Description': description,
            'Price': price,
            'NetQuantity': netquantity
        }])
        self.df = pd.concat([self.df, new_entry], ignore_index=True)  # Concatenate to the existing dataset

    def save_data(self):
        """
        Save the updated dataset back to the file.
        """
        self.df.to_csv(self.file_path, index=False)

    def _get_embeddings(self, text):
        """
        Generate BERT embeddings for a given text.

        Args:
        - text (str): Input text to embed.

        Returns:
        - torch.Tensor: Embedding vector for the input text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():  # No gradients required during inference
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  
        return embeddings

    def _precompute_embeddings(self):
        """
        Precompute embeddings for the 'Name' column in the dataset.

        Adds a new column 'Embedding' containing the BERT embeddings for each product name.
        """
        self.df['Embedding'] = self.df['Name'].apply(lambda name: self._get_embeddings(name))

    def search(self, query_embedding, top_k=10):
        """
        Perform semantic search by comparing the query embedding with precomputed embeddings.

        Args:
        - query_embedding (torch.Tensor): Embedding vector for the query.
        - top_k (int): Number of top matches to return.

        Returns:
        - pd.DataFrame: Top-k rows of the dataset with the highest similarity scores.
        """
        # Compute cosine similarity between query embedding and each product name embedding
        similarities = self.df['Embedding'].apply(
            lambda emb: cosine_similarity(query_embedding.numpy(), emb.numpy())[0][0]
        )
        self.df['Similarity'] = similarities  # Add similarity scores to the DataFrame
        return self.df.sort_values(by='Similarity', ascending=False).head(top_k)  # Return top-k matches
