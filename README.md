# AI-Script-Studio-LangGraph-Project
AI Script Studio is a Python-based application that uses LangGraph, LangChain, Groq, and Streamlit to transform raw text into an engaging video script and translate it into natural Hinglish.

Features:
1. AI Text Editor – Corrects grammar, spelling, and readability.
2. AI Script Writer – Converts edited text into an engaging YouTube video script.
3. Hinglish Translator – Translates the generated script into natural Hinglish.
4. Streamlit UI – Provides a simple, interactive web interface.
5. Download Output – Allows users to download the edited text, video script, and final Hinglish output as text files.

Technologies Used:
- Python
- LangGraph
- LangChain
- Groq LLM
- Streamlit
- Python-dotenv

Project Workflow:
Raw Text Input → AI Editor → Video Script Writer → Hinglish Translator → Final Output

Project Files:
- app.py – Contains the Streamlit UI and LangGraph workflow.
- .env – Stores the Groq API key.

Configuration:
Add your Groq API key to the .env file:

GROQ_API_KEY=your_groq_api_key

How to Run:
Run the following command in your terminal:

streamlit run translator.py

The application will open in your browser.

Purpose:
This project demonstrates how to build a sequential AI workflow using LangGraph, integrate a large language model through Groq, and create an interactive application using Streamlit.
