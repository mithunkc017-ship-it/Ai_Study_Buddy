import os
import re
import json

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# Quiz state: preserved during the current Streamlit session.
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False


# Keep chat history during the current Streamlit session.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


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
            score = len(matched_words) / len(question_words)
            results.append({
                "text": chunk,
                "score": score
            })

    results.sort(key=lambda item: item["score"], reverse=True)
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
                retry_options=types.HttpRetryOptions(attempts=3)
            )
        )

        previous_conversation = ""
        if chat_history:
            recent_history = chat_history[-4:]
            previous_conversation = "\n".join(
                f"Student: {item['question']}\nAssistant: {item['answer']}"
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
{previous_conversation or 'No previous conversation.'}

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


def generate_quiz_questions(pdf_text, number_of_questions=5):
    """Generate multiple-choice questions from the uploaded PDF."""

    if not api_key:
        raise ValueError("Gemini API key not found. Check your .env file.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=60000,
            retry_options=types.HttpRetryOptions(attempts=3)
        )
    )

    prompt = f"""
Create {number_of_questions} multiple-choice questions based only on
the study notes below.

Return ONLY a valid JSON array. Do not use Markdown fences.
Each item must have these keys:
- question: the question text
- options: an array of exactly four distinct answer choices
- correct_answer: the exact text of one option
- explanation: a short explanation supported by the notes

Make the questions educational and avoid repeating the same concept.
Do not invent facts that are not in the notes.

STUDY NOTES:
{pdf_text[:5000]}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=2500,
            response_mime_type="application/json"
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty quiz.")

    questions = json.loads(response.text)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Gemini did not return a valid quiz.")

    valid_questions = []
    for item in questions:
        if not isinstance(item, dict):
            continue

        question = item.get("question")
        options = item.get("options")
        correct_answer = item.get("correct_answer")
        explanation = item.get("explanation")

        if (
            isinstance(question, str)
            and question.strip()
            and isinstance(options, list)
            and len(options) == 4
            and all(isinstance(option, str) and option.strip() for option in options)
            and len(set(options)) == 4
            and correct_answer in options
            and isinstance(explanation, str)
            and explanation.strip()
        ):
            valid_questions.append({
                "question": question.strip(),
                "options": options,
                "correct_answer": correct_answer,
                "explanation": explanation.strip()
            })

    if len(valid_questions) != number_of_questions:
        raise ValueError(
            f"Expected {number_of_questions} valid questions, but received {len(valid_questions)}. Please try again."
        )

    return valid_questions


st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Study Buddy")
st.write("Upload your study notes and ask questions about them.")
st.divider()


def extract_pdf_text(pdf_bytes):
    """Extract text from all pages of a PDF."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
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


def render_quiz_section(pdf_text):
    """Display the quiz generator and score history."""

    st.divider()
    st.subheader("📝 AI Quiz Generator")
    st.write("Test your understanding of the uploaded study notes.")

    number_of_questions = st.selectbox(
        "Number of questions",
        options=[3, 5, 10],
        index=1,
        key="quiz_question_count"
    )

    if st.button("Generate Quiz", type="primary", key="generate_pdf_quiz"):
        try:
            with st.spinner("Creating questions from your PDF..."):
                questions = generate_quiz_questions(pdf_text, number_of_questions)

            st.session_state.quiz_questions = questions
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False

            for index in range(20):
                st.session_state.pop(f"quiz_answer_{index}", None)

        except Exception as error:
            st.error(f"Could not generate the quiz: {error}")

    questions = st.session_state.quiz_questions

    if questions:
        st.markdown("### Choose your answers")

        if not st.session_state.quiz_submitted:
            with st.form("pdf_quiz_form"):
                for index, item in enumerate(questions):
                    st.markdown(f"**Question {index + 1}. {item['question']}**")
                    st.radio(
                        "Select one answer:",
                        options=item["options"],
                        index=None,
                        key=f"quiz_answer_{index}"
                    )

                submitted = st.form_submit_button("Submit Quiz", type="primary")

            if submitted:
                selected_answers = {
                    index: st.session_state.get(f"quiz_answer_{index}")
                    for index in range(len(questions))
                }

                score = sum(
                    selected_answers[index] == item["correct_answer"]
                    for index, item in enumerate(questions)
                )

                st.session_state.quiz_answers = selected_answers
                st.session_state.quiz_results.append({
                    "score": score,
                    "total": len(questions)
                })
                st.session_state.quiz_submitted = True
                st.rerun()

        if st.session_state.quiz_submitted:
            answers = st.session_state.quiz_answers
            score = sum(
                answers.get(index) == item["correct_answer"]
                for index, item in enumerate(questions)
            )
            total = len(questions)

            st.success(f"Your score: {score}/{total} ({score / total:.0%})")
            st.progress(score / total)

            st.markdown("### Answer review")

            for index, item in enumerate(questions):
                selected = answers.get(index)
                correct = item["correct_answer"]

                if selected == correct:
                    st.markdown(f"**Question {index + 1}: Correct**")
                else:
                    st.markdown(f"**Question {index + 1}: Incorrect**")
                    st.write(f"Your answer: {selected or 'Not answered'}")

                st.write(f"Correct answer: {correct}")
                st.write(f"Explanation: {item['explanation']}")
                st.divider()

            if st.button("Try a New Quiz", key="reset_pdf_quiz"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False

                for index in range(20):
                    st.session_state.pop(f"quiz_answer_{index}", None)

                st.rerun()

    if st.session_state.quiz_results:
        st.markdown("### 📊 Quiz Score History")
        for attempt, result in enumerate(reversed(st.session_state.quiz_results), start=1):
            st.write(
                f"Attempt {attempt}: {result['score']}/{result['total']} "
                f"({result['score'] / result['total']:.0%})"
            )


uploaded_file = st.file_uploader("📄 Upload your study notes", type=["pdf"])

if uploaded_file is not None:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("Please upload a PDF smaller than 10 MB.")
    else:
        try:
            pdf_text = extract_pdf_text(uploaded_file.getvalue())

            if not pdf_text.strip():
                st.warning("No readable text found. This PDF may contain scanned images.")
            else:
                chunks = split_into_chunks(pdf_text)

                st.success(
                    f"PDF ready! Extracted {len(pdf_text):,} characters from your notes."
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
                        for index, chat in enumerate(st.session_state.chat_history, start=1):
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
                    submitted = st.form_submit_button("Find Answer", type="primary")

                if submitted:
                    if not question.strip():
                        st.warning("Please enter a question.")
                    else:
                        with st.spinner("Finding relevant passages..."):
                            answers = find_answer(question, chunks)

                        if answers:
                            context = "\n\n".join(answer["text"] for answer in answers)
                            st.subheader("🤖 AI-generated answer")

                            with st.spinner("Generating your answer..."):
                                ai_answer = generate_ai_answer(
                                    question,
                                    context,
                                    st.session_state.chat_history
                                )

                            st.write(ai_answer)
                            st.session_state.chat_history.append({
                                "question": question.strip(),
                                "answer": ai_answer
                            })

                            st.divider()
                            st.subheader("📖 Relevant passages from your PDF")

                            for number, answer in enumerate(answers, start=1):
                                st.markdown(f"**Relevant passage {number}**")
                                st.write(answer["text"])
                                st.caption(f"Keyword match: {answer['score']:.1%}")

                            st.caption(
                                "AI answers are based on the matching passages from your uploaded PDF. "
                                "Always verify important details."
                            )
                        else:
                            st.warning(
                                "I couldn't find a matching passage in your notes. Try asking with keywords used in the PDF."
                            )
                render_quiz_section(pdf_text)
                
        except Exception as error:
            st.error("Unable to read this PDF.")
            st.exception(error)
else:
    st.info("Upload a PDF to get started.")
