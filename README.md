### 🧠 RAG Prototype: Notes & Design Choices

> **Just click the Colab link**, upload the processed text file (attached in this repo), and test the methods directly in your browser.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shohog/bella/blob/main/RagProto10Min.ipynb)



#### 📌 Key Implementation Notes

* ✅ I used **Gemini** to transcribe the text from storybook images, extracted from the PDF using `ffmpeg`.

* 🧱 For chunking the text:

  * Initially, I used **`RecursiveCharacterTextSplitter` from LangChain**.
  * However, I believe **semantic chunking** is more appropriate, especially for stories.

* 🌐 **Embedding model**:

  * I’m using the **Gemini embedding model** (free and easy to integrate).
  * Previously, we tried **Voyage Multilingual 2** for Bengali.
  * I also plan to explore **`intfloat/e5`** embeddings for richer semantic understanding.

* 🔍 For retrieval:

  * I'm using **vector similarity search** (based on embedding similarity).
  * Though simple, this works reasonably well for short stories like *Kollyani*.
  * For complex knowledge bases, **hybrid search** (dense + keyword) would be better.
  * Building a proper **knowledge base** is a realistic and scalable next step.


#### ⚠️ Challenges

> The most challenging part is retrieving the most relevant chunk.
> Semantic chunking and a well-structured knowledge base are crucial for effective and accurate RAG.

