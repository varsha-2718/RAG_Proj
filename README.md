## 📄 Policy Document Question Answering using RAG

This project implements a Retrieval-Augmented Generation (RAG) based question-answering assistant over company policy documents. The system retrieves relevant information from policy files and generates grounded, non-hallucinated answers using a large language model.

##📁 Project Overview##

The system is designed to:
. Load and preprocess policy documents
. Store semantic embeddings in a vector database
. Retrieve relevant document chunks based on user queries
. Generate accurate answers strictly from retrieved context
. Gracefully handle missing or out-of-scope information

##🧩 1. Data Preparation
1.1 Loading Policy Documents
The policy documents are provided in text (.txt) format and are loaded using LangChain’s TextLoader. This loader preserves paragraph boundaries, which is important for maintaining semantic coherence during retrieval. Although text files are used in this implementation for simplicity and compatibility, the same approach can be extended to PDF or Markdown documents.

1.2 Cleaning and Chunking Strategy

After loading, the documents are split into smaller chunks using the RecursiveCharacterTextSplitter. This splitter prioritizes breaking text at natural boundaries such as paragraphs and sentences before falling back to word or character-level splits. This ensures that each chunk retains meaningful context.
Chunking Parameters:
Chunk size: 500 characters
Chunk overlap: 50 characters
This overlap helps preserve continuity across chunk boundaries.

1.3 Explanation of Chunk Size Choice
A chunk size of 500 characters provides a balance between retrieval precision and contextual completeness. Smaller chunks improve semantic similarity matching, while larger chunks preserve enough context for accurate answer generation. The 50-character overlap reduces the risk of losing important information that may appear at the boundaries of chunks.

##🔗 2. RAG Pipeline
2.1 Embedding Generation
Each text chunk is converted into a numerical vector using the Sentence Transformers all-MiniLM-L6-v2 model. This lightweight and efficient model is well-suited for semantic similarity and retrieval tasks.

2.2 Vector Storage
The generated embeddings are stored in a Qdrant vector database using a persistent local storage path. This allows embeddings to persist across multiple runs without requiring Docker or a cloud-based setup.

2.3 Semantic Retrieval
When a user submits a query, the system performs top-k semantic retrieval (k = 3) using cosine similarity to identify the most relevant document chunks from the vector store.

2.4 Context Injection into LLM
The retrieved document chunks are concatenated and passed as explicit context to the language model (Ollama with LLaMA 3). The model is instructed to generate answers strictly based on this retrieved context, ensuring grounded and relevant responses.

##🧠 3. Prompt Engineering
3.1 Initial Prompt
The initial prompt instructed the language model to answer user questions using the retrieved context. However, it did not strongly restrict hallucinations or enforce structured output.

3.2 Improved Prompt
. The prompt was improved by:
. Explicitly instructing the model to answer only from the provided context
. Adding a rule to gracefully handle missing information
. Requesting clear and concise responses
. Improved Prompt Logic:
. If the answer is present → respond clearly
. If the answer is missing → explicitly state that the information is unavailable in the documents

3.3 Reason for Improvements
These improvements were made to:
. Reduce hallucinations
. Improve answer reliability
. Ensure transparency when information is unavailable
. Make responses more consistent and easier to evaluate

##📊 4. Evaluation
4.1 Evaluation Set
A set of 7 evaluation questions was created, including:
. Fully answerable questions
. Partially answerable questions
. Completely unanswerable questions
. This mix helps assess system robustness across different scenarios.

4.2 Evaluation Criteria
Each response was evaluated manually using:
Accuracy, Hallucination avoidance and Answer clarity

A simple rubric was applied:
✅ Correct and grounded
⚠️ Partially correct but non-hallucinated
❌ Incorrect or hallucinated

4.3 Evaluation Outcome
The system performed well on answerable questions, avoided hallucinations on unanswerable queries, and produced clear, concise responses. This demonstrates that the RAG approach significantly improves factual reliability compared to standalone language models.

##⚠️ 5. Edge Case Handling
5.1 No Relevant Documents Found
When semantic retrieval returns no relevant chunks, the system informs the user that no relevant documents were found, preventing misleading or fabricated responses.

5.2 Question Outside Knowledge Base
If a question is outside the scope of the provided documents, the model responds with:
“The information is not available in the provided documents.”
This ensures transparency and prevents hallucinations.

###✅ Final Conclusion
This project successfully implements a Retrieval-Augmented Generation (RAG) pipeline for policy-based question answering. By combining semantic retrieval with controlled language model generation, the system achieves strong grounding, effective hallucination avoidance, and robustness across a wide range of query types.

##🚀 Technologies Used
LangChain
Sentence Transformers
Qdrant (Local Persistent Storage)
Ollama (LLaMA 3)
Streamlit
