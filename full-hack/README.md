# Semantic Search Application

This project is a **Semantic Search and Data Management Tool** that allows users to search for items in a dataset using natural language queries in any language. It also enables adding new items to the dataset with required fields.

---

## **Problem Statement**

Many organizations face challenges in retrieving specific items from large datasets due to linguistic barriers and lack of proper search tools. This project aims to:

1. Provide a **user-friendly interface** for searching items in a dataset.
2. Allow **multi-lingual query input** with automatic translation to English.
3. Enable adding new items to the dataset, ensuring mandatory fields are provided.

---

## **Features**

### **Search**
- Search for products using **natural language queries**.
- Input queries in **any language** using auto-detection or manual language selection.
- Display **translated query in English**.
- Show all matching results, including:
  - Name
  - Description
  - Price
  - Net Quantity
  - Similarity Score

### **Add Data**
- Add new data entries with the following mandatory fields:
  - `Name`
  - `Description`
  - `Price`
  - `Net Quantity`
- Automatically save the updated dataset.

---

## **Structure**

### **SemanticSearch Class**
Handles:
- Loading and preprocessing the dataset.
- Converting `Name` field into BERT embeddings.
- Performing semantic search using cosine similarity.
- Adding and saving new data.

### **Tkinter GUI**
Provides:
- **Search Mode**: Input a query, view translated text, and results.
- **Add Data Mode**: Input required fields to add new items.
- **Language Options**: Choose between auto-detection or specific input language.

---

## **Setup and Usage**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/semantic-search-app.git
   cd semantic-search-app
