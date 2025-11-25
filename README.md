# SIYB GYB Coach (AI Chat-Based Trainer)

This is an AI-powered, chat-based coach for the **Generate Your Business Idea (GYB)** module 
of the ILO's SIYB programme.

## Features

- Guided flow:
  1. Background
  2. Business Idea
  3. Customers
  4. Competitors
  5. Location
  6. Auto-generated idea summary

- Uses the official GYB manual (converted to chunks: `gyb_chunks.json`)
- Retrieval-Augmented Generation (RAG) with Llama 3.1 via Groq
- Simple Streamlit chat UI

## How to run (locally)

1. Create a virtual env (optional but recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
