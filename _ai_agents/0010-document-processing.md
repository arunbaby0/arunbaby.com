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
difficulty: Medium-Easy
---

**"Garbage In, Garbage Out. The Art of Reading Messy Data."**

## 1. Introduction: The PDF Trap

We like to think of data as clean JSON or SQL. In reality, 80% of enterprise knowledge is locked in **Unstructured Documents**: mostly PDFs, PowerPoints, and Excel sheets.

For an human, a PDF is easy to read. For an LLM, a PDF is a nightmare.
*   **No Structure:** A PDF doesn't know what a "Paragraph" is. It only knows "Place character 'A' at coordinates x=10, y=20."
*   **Tables:** A table in a PDF is just a grid of lines and floating text. Reconstructing the row/column relationship is an NP-hard problem for traditional parsers.
*   **Layout:** Multi-column layouts confuse standard text extractors (reading across columns instead of down).

If your Agent's retrieval system (RAG) feeds it garbage text from a broken PDF parse, the Agent will fail. **Document Processing** is the unsexy but critical precursor to intelligence.

---

## 2. Extraction Strategies

How do we get text out?

### 2.1 Strategy 1: Text-Based Extraction (Fast)
*   **Tools:** `pypdf`, `PyMuPDF`.
*   **Mechanism:** Extracts the underlying text stream embedded in the file.
*   **Pros:** Extremely fast (milliseconds). Cheap.
*   **Cons:** Fails on Scanned PDFs (images). Fails on complex layouts (tables, forms).

### 2.2 Strategy 2: OCR (Optical Character Recognition)
*   **Tools:** `Tesseract` (Open Source), Amazon Textract, Google Document AI.
*   **Mechanism:** Renders the PDF as an image, then looks for shapes of letters.
*   **Pros:** Works on scanned documents and screenshots.
*   **Cons:** Slow. Expensive. Often misreads "0" as "O" or "1" as "l".

### 2.3 Strategy 3: Vision-Language Models (The New Standard)
*   **Tools:** GPT-4V, LlamaParse (LlamaIndex), Unstructured.io.
*   **Mechanism:**
    1.  Take a screenshot of the PDF page.
    2.  Send it to a Vision Model (GPT-4o).
    3.  Prompt: "Transcribe this page into Markdown, preserving tables."
*   **Pros:** **Understand Layout.** It "sees" the difference between a header and a footer. It reconstructs tables perfectly into Markdown.
*   **Cons:** Most expensive (Vision tokens are pricey). Slower.

---

## 3. The Table Problem

Tables are the nemesis of Agents.
*   *Snippet:* "Revenue | 2020 | 10M"
*   *Bad Parse:* "Revenue 2020 10M" (Lost the relationship).
*   *Agent Query:* "What was revenue in 2020?"
*   *Result:* Agent fails.

**Solution: Markdown Conversion.**
We must force the extractor to output Markdown Tables (`| Col | Col |`). LLMs are trained heavily on Markdown tables and can reason about them.
*   **LlamaParse** is currently the state-of-the-art for this, utilizing a specialized model to detect table borders and reconstruct the grid.

---

## 4. Multi-Modal RAG

What about charts? A Vector Database cannot "search" a bar chart.

### 4.1 Indexing Images
1.  **Extraction:** Detect images in the PDF. Crop them.
2.  **Captioning:** Send the image to a Vision Model. "Describe this chart in detail."
    *   *Result:* "A bar chart showing Q3 Sales rising by 20%."
3.  **Embedding:** Embed the **Caption** (Text) into the Vector DB.
4.  **Storage:** Store the original Image path in metadata.

### 4.2 Retrieval
1.  User: "Did sales go up?"
2.  Search matches the *Caption* ("Sales rising...").
3.  Agent retrieves the caption (and optionally displays the image to the user).

---

## 5. Code: Using LlamaParse

Example of using a modern parser (Conceptual).

```python
# pip install llama-parse
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-...",
    result_type="markdown"  # The magic setting
)

# Parse a complex PDF
documents = parser.load_data("./annual_report.pdf")

print(documents[0].text[:500])
# Output:
# # Annual Report 2023
# 
# ## Financial Highlights
# | Metric | 2022 | 2023 |
# |--------|------|------|
# | Revenue| 10B  | 12B  |
```

The output is clean Markdown, ready for chunking and embedding.

---

## 6. Summary

Don't cheap out on your parser.
*   If you use `pypdf` on a financial report, your agent will be stupid.
*   If you use **Vision/Markdown parsing**, your agent can read tables and charts like a human analyst.

Data quality > Model quality.

In the next post, we will look at **Vector Search** algorithms in depth: HNSW, IVF, and metrics.
