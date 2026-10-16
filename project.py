import os
from typing import TypedDict

# Lets create a State first, which will hhold the project
class pipelinestate(TypedDict):
    raw_input: str
    editor_text: str
    script_text: str
    final_output: str

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# Creating nodes for the pipeline
def editor_node(state: pipelinestate) -> dict:
    """Stage 1: Cleanup the raw text grammar, remove typos, and make it more readable """
    prompt = (
        "You are an expert copyeditor. Cleanup the following raw text."
        "Fix any gramatical errors, spelling mistakes, and improve the overall readability."
        "While doing so, ensure that the original meaning and intent of the text is preserved. \n\n"
        f"Text:\n{state['raw_input']}\n\n"
    )
    response = llm.invoke(prompt)
    return {"editor_text" : response.content.strip()}

def script_node(state: pipelinestate) -> dict:
    """Stage 2: Format the cleaned text into an engaged script for a video. """
    prompt = (
        "You are a charimastic YouTube content creator. Take this edited text and turn it into a compelling script for a video. "
        "Format the following cleaned text into an engaging script for a video. "
        "Make it sound like a professional YouTuber would speak. Return only the script without any additional text. \n\n"
        f"Edited Text:\n{state['editor_text']}\n\n"
    )
    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}

def translator_node(state: pipelinestate) -> dict:
    """Stage 3: Translate the script into natural flowing Hinglish."""
    prompt = (
        "You are a professional translator for the Indian market. Translate the following script into Hinglish. "
        "Ensure that the translation is accurate and maintains the original tone and style of the script. "
        "Return only the translated script without any additional text. \n\n"
        f"Script:\n{state['script_text']}\n\n"
    )
    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}

# The state and nodes are defined, it is time to create a graph and we can create a pipeline that connects them together
# For that we have to use edges and edges are very important to create the worklows.

from langgraph.graph import StateGraph, START, END

# Create a graph and add the nodes to it
graph = StateGraph(pipelinestate)
graph.add_node("editor", editor_node)
graph.add_node("script", script_node)
graph.add_node("translator", translator_node)

# Add edges to connect the nodes (Sequential - one after another)
graph.add_edge(START, "editor")
graph.add_edge("editor", "script")
graph.add_edge("script", "translator")
graph.add_edge("translator", END)

# Compile the graph
app = graph.compile()
result = app.invoke(
    {
        "raw_input": "AI Agents are future technology. They can think, plan, and act on their own. LangGraph helps you build AI AGents with proper control and memory."
     }
    )

# Output the final result
print("The final output is: \n\n" )
print(result["final_output"])