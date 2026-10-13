
# 📚 AI Study Buddy – Smart PDF Chatbot

AI Study Buddy is an AI-powered learning assistant that helps students study more effectively by interacting with their PDF notes. Students can upload study materials, ask questions in natural language, and receive simple explanations based on the content of their documents.

The application combines PDF text extraction, keyword-based passage retrieval, and Google's Gemini AI model to provide contextual answers and support follow-up questions.

## ✨ Features

- **PDF Upload:** Upload study notes in PDF format.
- **Text Extraction:** Extract readable text from PDF documents using PyMuPDF.
- **AI-Powered Answers:** Generate concise, easy-to-understand answers using Google Gemini.
- **Smart Passage Retrieval:** Find relevant sections of notes using Python keyword matching.
- **Follow-Up Questions:** Use previous questions and conversation history to improve context for follow-up queries.
- **Chat History:** Review previous questions and answers during the current session.
- **Suggested Questions:** Start learning with ready-made question suggestions.
- **PDF Preview:** View extracted study notes directly in the application.
- **Secure API Configuration:** Load the Gemini API key from environment variables.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend and UI:** Streamlit
- **AI Model:** Google Gemini (`gemini-3.5-flash-lite`)
- **PDF Processing:** PyMuPDF
- **Environment Variables:** python-dotenv
- **Search Logic:** Python regular expressions and keyword matching

## 🎯 Project Objective

The goal of AI Study Buddy is to make studying easier by allowing students to interact with their own learning materials instead of manually searching through lengthy PDF documents.

## 🚀 Future Enhancements

- AI-generated quizzes and practice tests
- Automated revision flashcards
- More advanced semantic PDF search
- Persistent chat history
- Support for multiple document formats

## 👨‍💻 Learning Outcomes

This project provides practical experience with Python functions, text processing, PDF extraction, API integration, prompt engineering, session state management, and building interactive AI applications.

**Note:** AI-generated answers depend on the available document context and should be verified for accuracy.