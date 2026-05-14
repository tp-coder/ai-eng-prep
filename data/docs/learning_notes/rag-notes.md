# RAG Notes

Retrieval-augmented generation, or RAG, is a pattern where an application retrieves relevant information before asking the language model to answer.

A basic RAG pipeline has five steps:

1. Load source documents.
2. Split documents into chunks.
3. Create embeddings for each chunk.
4. Retrieve chunks similar to the user question.
5. Ask the model to answer using the retrieved context.

RAG helps reduce hallucination because the model is not expected to answer only from its general training data. Instead, it receives specific context from trusted documents.

A good RAG answer should cite which documents or chunks were used. If the retrieved context does not contain enough information, the assistant should say that the answer is not available from the provided documents.