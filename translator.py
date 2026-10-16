
import os
import streamlit as st
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Script Studio",
    page_icon="🎬",
    layout="wide",
)

load_dotenv()


# --------------------------------------------------
# 2. Define Pipeline State
# --------------------------------------------------
class PipelineState(TypedDict, total=False):
    raw_input: str
    editor_text: str
    script_text: str
    final_output: str


# --------------------------------------------------
# 3. Initialize Groq LLM
# --------------------------------------------------
@st.cache_resource
def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Please configure your .env file."
        )

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.7,
    )


# --------------------------------------------------
# 4. Define Pipeline Nodes
# --------------------------------------------------
def editor_node(state: PipelineState) -> dict:
    """Stage 1: Correct grammar, spelling, and readability."""

    prompt = f"""
    You are an expert copyeditor.
    Correct grammar, spelling mistakes, and readability issues.
    Preserve the original meaning and intent.

    Return only the edited text.

    Text:
    {state['raw_input']}
    """

    response = get_llm().invoke(prompt)

    return {
        "editor_text": response.content.strip()
    }


def script_node(state: PipelineState) -> dict:
    """Stage 2: Convert edited text into an engaging video script."""

    prompt = f"""
    You are a charismatic YouTube content creator.
    Convert the edited text into an engaging, professional video script.
    Make it natural, compelling, and easy to speak aloud.
    Preserve the original facts and meaning.

    Return only the script without additional explanations.

    Edited Text:
    {state['editor_text']}
    """

    response = get_llm().invoke(prompt)

    return {
        "script_text": response.content.strip()
    }


def translator_node(state: PipelineState) -> dict:
    """Stage 3: Translate the script into natural Hinglish."""

    prompt = f"""
    You are a professional translator for the Indian audience.
    Translate the following script into natural, conversational
    Hinglish using Hindi and English in Roman script.
    Preserve the original meaning, tone, and style.
    Do not summarize or omit important details.

    Return only the translated script.

    Script:
    {state['script_text']}
    """

    response = get_llm().invoke(prompt)

    return {
        "final_output": response.content.strip()
    }


# --------------------------------------------------
# 5. Build LangGraph Workflow
# --------------------------------------------------
@st.cache_resource
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("editor", editor_node)
    graph.add_node("script", script_node)
    graph.add_node("translator", translator_node)

    graph.add_edge(START, "editor")
    graph.add_edge("editor", "script")
    graph.add_edge("script", "translator")
    graph.add_edge("translator", END)

    return graph.compile()


# --------------------------------------------------
# 6. Application Header
# --------------------------------------------------
st.title("🎬 AI Script Studio")

st.markdown(
    "Transform your raw text into a polished Hinglish video script."
)

st.caption(
    "Powered by LangGraph, LangChain, Groq, and Streamlit"
)

st.divider()


# --------------------------------------------------
# 7. Input Section
# --------------------------------------------------
st.subheader("📝 Enter Your Content")


def clear_inputs() -> None:
    """Clear the input and generated output before widgets are instantiated."""
    st.session_state["raw_text"] = ""
    st.session_state.pop("pipeline_result", None)


raw_text = st.text_area(
    "Raw Text",
    placeholder=(
        "Enter your content here...\n\n"
        "Example: AI Agents are future technology. "
        "They can think, plan, and act on their own."
    ),
    height=220,
    key="raw_text",
)

col1, col2 = st.columns([1, 1])

with col1:
    generate = st.button(
        "🚀 Generate Script",
        type="primary",
        use_container_width=True,
    )

with col2:
    clear = st.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=clear_inputs,
    )


# --------------------------------------------------
# 8. Execute LangGraph Pipeline
# --------------------------------------------------
if generate:
    if not raw_text.strip():
        st.warning("Please enter some text before generating.")
    else:
        try:
            app = build_graph()

            progress = st.progress(0)
            status = st.empty()

            with st.spinner("Processing your content with AI..."):
                status.info("Running the AI editing and scripting workflow...")

                result = app.invoke({
                    "raw_input": raw_text.strip()
                })

                st.session_state["pipeline_result"] = result

            progress.progress(100)
            status.success("All three pipeline stages completed!")

        except Exception as exc:
            st.error(f"An error occurred: {exc}")


# --------------------------------------------------
# 9. Display Generated Results
# --------------------------------------------------
result = st.session_state.get("pipeline_result")

if result:
    st.divider()
    st.subheader("✨ Generated Results")

    tab1, tab2, tab3 = st.tabs([
        "📝 Edited Text",
        "🎬 Video Script",
        "🌏 Hinglish Output",
    ])

    with tab1:
        st.markdown("#### Grammar-corrected content")

        edited_text = st.text_area(
            "Edited Text",
            value=result.get("editor_text", ""),
            height=280,
            key="edited_result",
        )

        st.download_button(
            "⬇️ Download Edited Text",
            data=edited_text,
            file_name="edited_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tab2:
        st.markdown("#### Engaging YouTube script")

        video_script = st.text_area(
            "Video Script",
            value=result.get("script_text", ""),
            height=320,
            key="script_result",
        )

        st.download_button(
            "⬇️ Download Video Script",
            data=video_script,
            file_name="video_script.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tab3:
        st.markdown("#### Final Hinglish script")

        final_output = st.text_area(
            "Final Hinglish Output",
            value=result.get("final_output", ""),
            height=350,
            key="final_result",
        )

        st.download_button(
            "⬇️ Download Final Script",
            data=final_output,
            file_name="hinglish_script.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True,
        )

    st.success("Your script is ready!")