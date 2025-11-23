https://colab.research.google.com/drive/1u8q5P_RLWwDaA8WUMNA4AYYYJNEr2DMN?usp=sharing

# HACKATHON-2
AI-Powered Document Search And Summarization System

RAG-Based Document Search & Summarization System
This repository contains the prototype we developed during the PSTB AI Hackathon 2025:
 a fully local Retrieval-Augmented Generation (RAG) system capable of searching and summarizing long documents directly on CPU, without external APIs.
The goal of this project was simple and ambitious:
 make it possible to find the right information, inside long and complex documents, in just a few seconds.

Background & Motivation
Modern institutions generate thousands of pages of text every year — reports, analyses, guidelines, policy papers. Searching through these documents manually is slow, tiring, and often imprecise. Even when relevant sections are found, they still require manual summarization.
We wanted to build a tool that solves all of this at once:
 a system that can search intelligently through documents and instantly summarize the relevant parts, all running completely locally.

Dataset
To test this idea, we used public UNESCO documents, available in PDF and text format. These documents are long, rich, diverse in structure, and ideal for testing semantic search and summarization.

Approach & Architecture
Our solution follows a simple but powerful pipeline:

Document Ingestion
 We extract raw text from PDF and TXT files using PyPDF2 and standard UTF-8 decoding.

Cleaning & Chunking
 The text is normalized and split into coherent segments (chunks) of 200–500 tokens, which makes them easier to embed and retrieve.

Semantic Embeddings
 Each chunk is transformed into a dense vector using the MiniLM sentence transformer, chosen for its excellent quality/speed ratio on CPU.

Vector Search with FAISS
 All embeddings are indexed in a FAISS structure, allowing us to retrieve semantically similar passages almost instantly.

Summarization
 The top-k retrieved chunks are concatenated and summarized by a lightweight transformer model (T5-small), producing a short, clear, human-readable answer.
This architecture is fully local, efficient, and scalable enough for the hackathon constraints.

Evaluation
Because our ground truth was limited, we combined:

Quantitative evaluation
 using precision@k and recall@k for semantic search

Human-based evaluation
 assessing clarity, relevance, fluency, and coherence of summaries
This hybrid approach gave us a realistic and practical understanding of system quality.

Results
In less than a minute, a user can:
upload one or several documents
ask a natural-language question
retrieve the most relevant passages
obtain a coherent summary of those passages
The prototype demonstrates that a lightweight, local RAG system can provide real added value without needing GPUs or cloud APIs.

Future Improvements
Although functional, our prototype can evolve in several ways:
support for more document formats (DOCX, HTML)
more powerful embedding models
dynamic chunking strategies
improved summarization (T5-large, Pegasus)
a complete web interface
a multi-turn conversational chatbot
cloud indexing for large document collections

Team
This project was developed collaboratively by:
Catherine Maameri
Cristina Moussoungedi
Adel Zitouni
during the PSTB AI Bootcamp Hackathon (2025).

Acknowledgments
Thank you to our instructors and colleagues for feedback, discussions, and support throughout the hackathon.
