# AI Engineering Prep Project Overview

This project is a hands-on preparation project for learning AI Engineering.

The project is designed to build a production-shaped AI assistant step by step. The first phase creates a clean Python application with environment configuration, structured LLM responses, CLI execution, logging, and tests.

The second phase adds retrieval-augmented generation, also known as RAG. The assistant will load local documents, split them into chunks, create embeddings, retrieve relevant chunks for a user question, and answer using only the retrieved context.

The third phase adds tool calling and mini-agent behavior. The assistant will be able to search documents, inspect document metadata, and generate Jira-style implementation tickets from grounded context.

The fourth phase adds evaluation, observability, Docker support, and project polish.

The long-term goal is to build an AI Technical Discovery Assistant that helps transform project notes, architecture decisions, technical documents, and tickets into implementation-ready outputs such as specs, acceptance criteria, risk lists, and test scenarios.