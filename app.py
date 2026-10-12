import os
import re

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")



def find_answer(question, chunks):
    """Find relevant passages using keyword matching."""

    stop_words = {
        "a", "an", "the", "is", "are", "was", "were",
        "what", "why", "how", "when", "where", "who",
        "which", "explain", "describe", "tell", "me",
        "about", "and", "or", "of", "to", "in", "on",
        "for", "with", "do", "does", "can", "could",
        "would", "should", "it", "this", "that", "be",
        "give", "define", "please", "i", "you", "my"
    }

    def get_keywords(text):
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {
            word for word in words
            if word not in stop_words and len(word) > 1
        }

    question_words = get_keywords(question)

    if not question_words or not chunks:
        return []

    results = []

    for chunk in chunks:
        chunk_words = get_keywords(chunk)
        matched_words = question_words.intersection(chunk_words)

        if matched_words:
            # Reward passages that match more of the question.
            score = len(matched_words) / len(question_words)

            results.append({
                "text": chunk,
                "score": score
            })

    results.sort(key=lambda item: item["score"], reverse=True)

    # Return up to 3 passages, even when the match is partial.
    return results[:3]




def generate_ai_answer(question, context, chat_history=None):
    """Generate an answer using Gemini and PDF context."""

    if not api_key:
        return "API key not found. Please check your .env file."

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=60000,
                retry_options=types.HttpRetryOptions(attempts=1)
            )
        )

        # Include the latest conversation to understand follow-up questions.
        previous_conversation = ""

        if chat_history:
            recent_history = chat_history[-4:]

            previous_conversation = "\n".join(
                f"Student: {item['question']}\n"
                f"Assistant: {item['answer']}"
                for item in recent_history
            )

        prompt = f"""
You are a helpful AI Study Buddy.

Answer the student's question using the provided PDF context.
Use simple language and concise explanations.
If the student asks a follow-up question, use the previous conversation
to understand what they mean.

The PDF is the primary source of factual information.
If the answer is not supported by the PDF context, say so clearly.
Do not invent facts.

Previous conversation:
{previous_conversation or "No previous conversation."}

PDF context:
{context[:4000]}

Current question:
{question}
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=300
            )
        )

        if response.text:
            return response.text

        return "Gemini returned an empty answer. Please try again."

    except Exception as error:
        return f"Could not generate an AI answer: {error}"


    


def get_keywords(text):
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {word for word in words if word not in stop_words}

        question_words = get_keywords(question)

        if not question_words or not chunks:
            return []

        results = []

        for chunk in chunks:
            chunk_words = get_keywords(chunk)
            matched_words = question_words & chunk_words

            if matched_words:
                score = len(matched_words) / len(question_words)

                results.append({
                    "text": chunk,
                    "score": score
                })

            results.sort(
                key=lambda item: item["score"],
                reverse=True
            )

        return results[:3]



st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Study Buddy")
st.write("Upload your study notes and ask questions about them.")

st.divider()
# Keep chat history during the current Streamlit session.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


def extract_pdf_text(pdf_bytes):
    """Extract text from all pages of a PDF."""
    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:
        return "\n".join(page.get_text() for page in pdf)


def split_into_chunks(text, chunk_size=120, overlap=30):
    """Split text into overlapping passages of words."""
    words = text.split()
    chunks = []

    if not words:
        return chunks

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


    def get_keywords(text):
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {word for word in words if word not in stop_words}

    question_words = get_keywords(question)

    if not question_words or not chunks:
        return []

    results = []

    for chunk in chunks:
        chunk_words = get_keywords(chunk)
        matched_words = question_words.intersection(chunk_words)

        if matched_words:
            score = len(matched_words) / len(question_words)
            results.append({
                "text": chunk,
                "score": score
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:3]



uploaded_file = st.file_uploader(
    "📄 Upload your study notes",
    type=["pdf"]
)

if uploaded_file is not None:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("Please upload a PDF smaller than 10 MB.")

    else:
        try:
            pdf_text = extract_pdf_text(
                uploaded_file.getvalue()
            )

            if not pdf_text.strip():
                st.warning(
                    "No readable text found. "
                    "This PDF may contain scanned images."
                )

            else:
                chunks = split_into_chunks(pdf_text)

                st.success(
                    f"PDF ready! Extracted {len(pdf_text):,} "
                    f"characters from your notes."
                )

                with st.expander("Preview extracted notes"):
                    st.text_area(
                        "PDF text",
                        value=pdf_text,
                        height=200,
                        disabled=True
                    )
                if st.session_state.chat_history:
                    with st.expander(
                        f"🕘 Chat history ({len(st.session_state.chat_history)})",
                        expanded=True
                    ):
                        for index, chat in enumerate(
                            st.session_state.chat_history,
                            start=1
                        ):
                            st.markdown(f"**You {index}:** {chat['question']}")
                            st.markdown(f"**🤖 AI Study Buddy:** {chat['answer']}")
                            st.divider()

                        if st.button("Clear chat history", key="clear_chat_history"):
                            st.session_state.chat_history = []
                            st.rerun()


                st.subheader("💬 Ask your study notes")
                st.caption("💡 Need an idea? Try one of these questions.")

                suggestions = [
                    "Summarize the main points",
                    "Explain the key concepts",
                    "What are the important definitions?",
                    "Give me a quick revision summary",
                ]

                suggestion_columns = st.columns(2)

                for index, suggestion in enumerate(suggestions):
                    with suggestion_columns[index % 2]:
                        if st.button(
                            suggestion,
                            key=f"suggestion_{index}",
                            use_container_width=True
                        ):
                            st.session_state.question_input = suggestion
                            st.rerun()

                with st.form("question_form"):
                    question = st.text_input(
                    "Your question",
                    placeholder="Example: What is machine learning?",
                    key="question_input"
)

                    submitted = st.form_submit_button(
                        "Find Answer",
                        type="primary"
                    )

                if submitted:
                    if not question.strip():
                        st.warning("Please enter a question.")
                    else:
                        
                        with st.spinner("Finding relevant passages..."):
                            answers = find_answer(question, chunks)

                        if answers:
                            context = "\n\n".join(
                                answer["text"] for answer in answers
                            )

                            st.subheader("🤖 AI-generated answer")

                            with st.spinner("Generating your answer..."):
                                ai_answer = generate_ai_answer(
                                    question,
                                    context,
                                    st.session_state.chat_history
                                )

                            st.write(ai_answer)
                            # Save the conversation for this session.
                            st.session_state.chat_history.append({
                                "question": question.strip(),
                                "answer": ai_answer
                            })

                            st.divider()
                            st.subheader("📖 Relevant passages from your PDF")

                            for number, answer in enumerate(answers, start=1):
                                st.markdown(
                                    f"**Relevant passage {number}**"
                                )
                                st.write(answer["text"])
                                st.caption(
                                    f"Keyword match: {answer['score']:.1%}"
                                )

                            st.caption(
                                "AI answers are based on the matching passages "
                                "from your uploaded PDF. Always verify important details."
                            )

                        else:
                            st.warning(
                                "I couldn't find a matching passage in your notes. "
                                "Try asking with keywords used in the PDF."
                            )



        except Exception as error:
            st.error("Unable to read this PDF.")
            st.exception(error)