import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from googletrans import Translator, LANGUAGES
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


# SemanticSearchApp Class: Provides a Tkinter-based GUI for the application.
class SemanticSearchApp:
    def __init__(self, root, searcher):
        """
        Initialize the GUI for the Semantic Search Application.

        Args:
        - root (tk.Tk): Root window for the GUI.
        - searcher (SemanticSearch): Instance of the SemanticSearch class for backend operations.
        """
        self.searcher = searcher  
        self.translator = Translator() 

        # Configure the root window
        root.title("Semantic Search App")
        root.geometry("800x750")  # Set the size of the GUI window
        root.resizable(False, False)  # Make the window size fixed
        root.configure(bg="#f0f8ff")  

        # Title Label
        title_label = tk.Label(root, text="Semantic Search Application", font=("Helvetica", 16, "bold"), bg="#f0f8ff", fg="#2f4f4f")
        title_label.pack(pady=10)

        # Mode Selection: Radio buttons for Search and Add Data modes
        self.mode_var = tk.StringVar(value="Search")
        mode_frame = tk.Frame(root, bg="#f0f8ff")
        mode_frame.pack(pady=5)

        ttk.Radiobutton(mode_frame, text="Search", variable=self.mode_var, value="Search", command=self.update_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Add Data", variable=self.mode_var, value="Add Data", command=self.update_mode).pack(side=tk.LEFT, padx=10)

        # Language Selection Dropdown
        language_frame = tk.Frame(root, bg="#f0f8ff")
        language_frame.pack(pady=5)

        tk.Label(language_frame, text="Input Language:", font=("Helvetica", 12), bg="#f0f8ff").pack(side=tk.LEFT, padx=5)
        self.language_var = tk.StringVar()
        language_choices = ["Auto-Detect"] + [f"{code}: {lang}" for code, lang in LANGUAGES.items()]
        self.language_dropdown = ttk.Combobox(language_frame, textvariable=self.language_var, values=language_choices, state="readonly", width=30)
        self.language_dropdown.set("Auto-Detect")
        self.language_dropdown.pack(side=tk.LEFT, padx=5)

        # Widgets for Search Mode
        self.query_label = tk.Label(root, text="Enter Query:", font=("Helvetica", 12), bg="#f0f8ff")
        self.query_entry = ttk.Entry(root, width=50, font=("Helvetica", 12))
        self.search_button = ttk.Button(root, text="Search", command=self.perform_search)

        # Widgets for Add Data Mode
        self.name_label = tk.Label(root, text="Name:", font=("Helvetica", 12), bg="#f0f8ff")
        self.name_entry = ttk.Entry(root, width=50, font=("Helvetica", 12))
        self.description_label = tk.Label(root, text="Description:", font=("Helvetica", 12), bg="#f0f8ff")
        self.description_entry = ttk.Entry(root, width=50, font=("Helvetica", 12))
        self.price_label = tk.Label(root, text="Price:", font=("Helvetica", 12), bg="#f0f8ff")
        self.price_entry = ttk.Entry(root, width=50, font=("Helvetica", 12))
        self.netquantity_label = tk.Label(root, text="Net Quantity:", font=("Helvetica", 12), bg="#f0f8ff")
        self.netquantity_entry = ttk.Entry(root, width=50, font=("Helvetica", 12))
        self.add_button = ttk.Button(root, text="Add Data", command=self.add_data)

        # Results Section
        self.translated_query_label = tk.Label(root, text="Translated Query:", font=("Helvetica", 12, "italic"), bg="#f0f8ff", fg="#4b0082")
        self.translated_query_box = tk.Label(root, text="", font=("Helvetica", 12), bg="#f0f8ff", wraplength=700, justify=tk.LEFT)

        self.results_label = tk.Label(root, text="Results:", font=("Helvetica", 12, "bold"), bg="#f0f8ff")
        self.results_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=20, font=("Helvetica", 10))

        self.update_mode()

    def update_mode(self):
        """
        Update the GUI layout based on the selected mode (Search or Add Data).
        """
        self.clear_widgets()  # Clear all existing widgets

        if self.mode_var.get() == "Search":
            # Display Search mode widgets
            self.query_label.pack(pady=5)
            self.query_entry.pack(pady=5)
            self.search_button.pack(pady=5)
            self.translated_query_label.pack(pady=5)
            self.translated_query_box.pack(pady=5)
            self.results_label.pack(pady=5)
            self.results_box.pack(pady=10)
        else:
            # Display Add Data mode widgets
            self.name_label.pack(pady=5)
            self.name_entry.pack(pady=5)
            self.description_label.pack(pady=5)
            self.description_entry.pack(pady=5)
            self.price_label.pack(pady=5)
            self.price_entry.pack(pady=5)
            self.netquantity_label.pack(pady=5)
            self.netquantity_entry.pack(pady=5)
            self.add_button.pack(pady=5)

    def clear_widgets(self):
        """
        Remove all widgets from the GUI for mode switching.
        """
        self.query_label.pack_forget()
        self.query_entry.pack_forget()
        self.search_button.pack_forget()
        self.translated_query_label.pack_forget()
        self.translated_query_box.pack_forget()
        self.results_label.pack_forget()
        self.results_box.pack_forget()
        self.name_label.pack_forget()
        self.name_entry.pack_forget()
        self.description_label.pack_forget()
        self.description_entry.pack_forget()
        self.price_label.pack_forget()
        self.price_entry.pack_forget()
        self.netquantity_label.pack_forget()
        self.netquantity_entry.pack_forget()
        self.add_button.pack_forget()

    def perform_search(self):
        """
        Handle the search operation, including language detection, translation, and result retrieval.
        """
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showerror("Error", "Query cannot be empty!")
            return

        # Detect language or use selected language for the query
        selected_language = self.language_var.get()
        if selected_language == "Auto-Detect":
            detected_language = self.translator.detect(query).lang
        else:
            detected_language = selected_language.split(":")[0]

        # Translate the query to English
        translated_query = self.translator.translate(query, src=detected_language, dest="en").text
        self.translated_query_box.config(text=translated_query)

        # Perform semantic search
        query_embedding = self.searcher._get_embeddings(translated_query)
        results = self.searcher.search(query_embedding, top_k=5)

        # Display search results
        self.results_box.delete(1.0, tk.END)
        if not results.empty:
            for _, row in results.iterrows():
                self.results_box.insert(
                    tk.END, f"Name: {row['Name']} | Description: {row['Description']} | Price: {row['Price']} | Net Quantity: {row['NetQuantity']} | Similarity: {row['Similarity']:.4f}\n\n"
                )
        else:
            self.results_box.insert(tk.END, "No results found.")

    def add_data(self):
        """
        Add a new data entry to the dataset after validating required fields.
        """
        name = self.name_entry.get().strip()
        description = self.description_entry.get().strip()
        price = self.price_entry.get().strip()
        netquantity = self.netquantity_entry.get().strip()

        # Ensure all required fields are filled
        if not name or not description or not price or not netquantity:
            messagebox.showerror("Error", "All fields are required!")
            return

        # Add the new entry and save the dataset
        self.searcher.add_data(name, description, price, netquantity)
        self.searcher.save_data()
        messagebox.showinfo("Success", "Data added successfully!")
