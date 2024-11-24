from semanticSearch import SemanticSearch
from semanticSearch_App import SemanticSearchApp
import tkinter as tk


# Main Function: Entry point for the application
if __name__ == "__main__":
    searcher = SemanticSearch(file_path="C:\\Users\\I_LOV\\Downloads\\cleaned_data_first_100.csv")
    searcher._precompute_embeddings()
    root = tk.Tk()
    app = SemanticSearchApp(root, searcher)
    root.mainloop()