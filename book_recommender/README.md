# Book Recommender System

A **Content-Based Book Recommendation System** that uses **vector similarity search** to recommend books based on user preferences.

---

## **Key Features**

- **Metadata Enrichment:**  
  Uses **Hugging Face Transformers** for **text classification** and **sentiment analysis** to enhance book metadata.
  
- **Vector Search with LangChain & ChromaDB:**  
  Indexes enriched book descriptions into **ChromaDB** and performs **high-performance similarity search** to retrieve relevant books.

- **Interactive Interface:**  
  Built with **Gradio** frontend and **FastAPI** backend for seamless user interaction.

- **Production Deployment:**  
  - **Containerized with Docker**  
  - **Automated CI/CD with GitHub Actions**  
  - **Deployed to AWS**   