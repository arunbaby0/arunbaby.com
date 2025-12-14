---
title: "Document Processing for Agents"
day: 10
collection: ai_agents
categories:
  - ai-agents
tags:
  - rag
  - pdf-parsing
  - ocr
  - multimodal
  - unstructured
  - llamaparse
difficulty: Medium-Easy
---

**"Garbage In, Garbage Out. The Art of Reading Messy Data."**

## 1. Introduction: The PDF Trap

We like to think of enterprise data as clean rows in a SQL database. In reality, **80% of enterprise knowledge** is locked in "Unstructured Documents": PDF contracts, PowerPoint slides, Excel financial models, and PNG screenshots of dashboards.

For a human, a PDF is easy to read. For an LLM, a PDF is a nightmare.
*   **No Semantic Structure:** A PDF doesn't know what a "Paragraph" or a "Header" is. It only knows "Place character 'A' at coordinates x=10, y=20."
*   **Tables:** A table in a PDF is just a grid of lines and floating text. Reconstructing the row/column relationship is an NP-hard problem for traditional parsers.
*   **Layout:** Multi-column layouts confuse standard text extractors (reading across columns instead of down).

If your Agent's retrieval system (RAG) feeds it garbage text from a broken PDF parse, the Agent will fail to answer even simple questions. **Document Processing** is the unsexy but critical precursor to intelligence.

---

## 2. Extraction Strategies: The Hierarchy of Power

How do we get text out? There are three main strategies, ordered by cost and quality.

### 2.1 Strategy 1: Text-Based Extraction (The Cheap Way)
*   **Tools:** `pypdf`, `PyMuPDF`, `pdfplumber`.
*   **Mechanism:** Extracts the underlying text stream embedded in the file.
*   **Pros:** Extremely fast (milliseconds). Cheap (CPU only).
*   **Cons:**
    *   Fails on **Scanned PDFs** (images wrapped in PDF).
    *   Fails on complex layouts (merges columns).
    *   Fails on tables (flattens them into a string mess).

### 2.2 Strategy 2: OCR (Optical Character Recognition)
*   **Tools:** `Tesseract` (Open Source), Amazon Textract, Google Document AI, Azure Layout Analysis.
*   **Mechanism:** Renders the PDF as an image, then looks for shapes of letters.
*   **Pros:** Works on scanned documents and screenshots. Can detect "Forms" and key-value pairs.
*   **Cons:** Slow. Expensive. Often misreads "0" (zero) as "O" (letter) or "1" as "l".

### 2.3 Strategy 3: Vision-Language Models (The New Gold Standard)
*   **Tools:** **GPT-4o Vision**, **LlamaParse (LlamaIndex)**, **Unstructured.io**.
*   **Mechanism:**
    1.  Take a screenshot of the PDF page.
    2.  Send it to a Multi-Modal LLM.
    3.  Prompt: "Transcribe this page into Markdown, carefully preserving all tables and headers."
*   **Pros:**
    *   **Understanding:** It "sees" that bold text is a header.
    *   **Tables:** It reconstructs tables perfectly into Markdown syntax (`| col | col |`).
    *   **Charts:** It can describe "Revenue is uptrending" from a bar chart.
*   **Cons:** Expensive (Vision tokens cost money). High latency (seconds per page).

---

## 3. The Table Problem: The Final Boss

Tables are the nemesis of RAG Agents.
*   *Snippet:* "Revenue | 2020 | 10M"
*   *Bad Parse:* "Revenue 2020 10M" (Lost the relationship).
*   *Agent Query:* "What was revenue in 2020?"
*   *Retrieved Chunk:* "Revenue 2020 10M"
*   *Agent Answer:* "The revenue is 10M." (Lucky guess, but robust agents fail).

**Solution: Markdown Conversion.**
We must force the extractor to output Markdown Tables. LLMs are trained heavily on Markdown (from GitHub READMEs) and can reason about them exceptionally well.
*   **LlamaParse** is currently the state-of-the-art for this. It utilizes a trained model just to detect table borders and reconstruct the grid structure before generating text.

---

## 4. Chunking Strategies: Cutting the Cake

Once you have text, you must split it.

### 4.1 Recursive Character Splitting
The default.
1.  Split by Paragraph `\n\n`.
2.  If too big, split by Sentence `.`.
3.  If too big, split by Word ` `.
*   **Overlap:** Always keep 50-100 tokens of overlap so sentences aren't cut in half.

### 4.2 Semantic Chunking
Instead of splitting by size, split by **Meaning**.
1.  Embed every sentence.
2.  Calculate cosine similarity between S1 and S2.
3.  If similarity drops below a threshold (e.g., 0.7), start a new chunk.
*   *Result:* You get coherent "Topics".

---

## 5. Multi-Modal RAG

What about charts? A Vector Database cannot "search" a bar chart.

### 5.1 Pattern: Image-to-Text Indexing
1.  **Extraction:** Detect images in the PDF. Crop them.
2.  **Captioning:** Send the image to a Vision Model. "Describe this chart in detail, including data points."
    *   *Result:* "A bar chart showing Q3 Sales rising by 20% to $12M."
3.  **Embedding:** Embed the **Caption** (Text) into the Vector DB.
4.  **Storage:** Store the original Image path in metadata.

### 5.2 Retrieval Flow
1.  User: "Did sales go up?"
2.  Search matches the *Caption* ("Sales rising...").
3.  Agent retrieves the caption and says "Yes, sales rose 20%."
4.  Optional: Agent displays the original image to the user.

---

## 6. Code: Modern Parsing Pipeline (Conceptual)

How an "Agentic Ingestion Pipeline" looks in pseudocode.

```python
def ingest_document(file_path):
    # 1. Routing
    if is_scanned(file_path):
        mode = "vision"
    else:
        mode = "text"
        
    # 2. Parsing (LlamaParse or Unstructured)
    text = parser.parse(file_path, mode=mode, output_format="markdown")
    
    # 3. Image Extraction
    images = extract_images(file_path)
    for img in images:
        caption = vision_model.caption(img)
        text += f"\n\n![Image]({caption})"
        
    # 4. Semantic Chunking
    chunks = semantic_chunker(text)
    
    # 5. Indexing
    vector_db.add(chunks)
```

---

## 7. Summary

Document Processing determines the **Ceiling** of your agent's intelligence.
*   **Text Extraction:** Use `pypdf` for simple text, `LlamaParse` for everything else.
*   **Tables:** Must be converted to Markdown.
*   **Charts:** Must be captioned by Vision models.

In the next section (Day 11), we will look at **Vector Search Algorithms** in depth and how to scale this to millions of documents.
