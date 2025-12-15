---
title: "Multimodal LLMs (GPT-4V, LLaVA)"
day: 22
collection: ai_agents
categories:
  - ai-agents
tags:
  - multimodal
  - gpt-4v
  - llava
  - visual-question-answering
  - ocr
  - high-res-scaling
difficulty: Medium
related_dsa_day: 22
related_ml_day: 22
related_speech_day: 22
---

**"One model to rule them all: Unifying Text and Vision."**

## 1. Introduction: The End of Separate Systems

In Day 21, we discussed "Composing" vision tools: running a detector, then a segmenter, then a classifier. This works, but it is brittle. It's the same problem as the early NLP pipelines (Tokenizer -> POS Tagger -> Parser -> Sentiment).

**Multimodal Large Language Models (MLLMs)** effectively collapsed the entire vision pipeline into a single Transformer.
Instead of:
`Image -> Detector -> [List of Boxes] -> Logic Code -> Answer`

We now have:
`Image + Text Prompt -> MLLM -> Text Answer`

You can simply pass an image of a dashboard and ask, "Is the server overheating?" The model does the detection, reading (OCR), and reasoning in one pass.

In this post, we will explore the architecture of these giants (GPT-4V, LLaVA) and the specific engineering challenges of using them for agents.

---

## 2. Architecture: How do you feed an Image to an LLM?

LLMs (Llama, GPT) only understand text tokens. How do we shove an image into the prompt?

### 2.1 The Projector Strategy (LLaVA)
**LLaVA (Large Language and Vision Assistant)** pioneered a simple, open-source approach that is now the standard.

1.  **Vision Encoder:** Take a pre-trained Vision Encoder (like CLIP ViT-L/14). This turns an image into a set of vectors (e.g., 256 vectors of size 1024).
2.  **The Projector (The Adapter):** These vision vectors live in "Vision Space". The LLM expects vectors in "Text Space".
    *   We train a simple **Linear Projection Layer** (matrix multiplication) to map Vision Vector $V$ to Text Vector $T$.
    *   $T = W \cdot V + b$.
3.  **The LLM:** We treat these projected vectors as if they were just "word embeddings" for a weird foreign language. We prepend them to the user's text prompt.
    *   Input: `[Image_Token_1] ... [Image_Token_256] "Describe this image."`

**Training:**
We freeze the Vision Encoder (it's already good at seeing). We freeze the LLM (it's already good at talking). We only train the **Projector** (the translation layer) and fine-tune the system on image-caption pairs.

### 2.2 The GPT-4V Approach (Hypothesized)
Proprietary models like GPT-4V likely use a more complex, massive "Cross-Attention" mechanism or are trained natively from scratch on interleaved text and images, allowing for much deeper integration than a simple projection.

---

## 3. The Resolution Dilemma

The biggest bottleneck for Vision Agents is **Resolution**.

### 3.1 The 224px Problem
Most CLIP models were trained on 224x224 or 336x336 pixel images to save compute.
*   **Scenario:** You screenshot a 1080p Excel spreadsheet.
*   **Compression:** Squishing 1080p to 336p turns text into unreadable blur. "Profit: $5000" becomes a gray smudge.

### 3.2 Dynamic High-Res Cropping (The Solution)
Models like **GPT-4o** and **LLaVA-Next** use a tiling strategy.
1.  **The Global View:** Feed the resized (low-res) image to get the "Gist" ("This is a spreadsheet").
2.  **The Tile View:** Cut the 1080p image into fixed patches (e.g., 512x512 tiles).
3.  **Process:** Encode each tile separately.
4.  **Concatenate:** Feed *all* tile embeddings into the context window.
    *   Input: `[Global_Tokens] [Tile_1_Tokens] [Tile_2_Tokens] ...`

**Cost Implication:**
This is why GPT-4V pricing is per-tile. A detailed high-res image might consume 1000+ tokens just to "see".

---

## 4. Visual Capabilities for Agents

What can these models actually *do*?

### 4.1 OCR on Steroids (OCR-free Reading)
Traditional OCR (Tesseract) gives you text but loses *structure*.
*   *Tesseract:* "Name Age Bob 24 Alice 30" (Is "Bob" the age or the name?).
*   *MLLM:* "This is a table. The column headers are Name and Age. The first row is Bob, aged 24."
MLLMs understand **Layout**. They can read charts, handwriting, and complex forms.

### 4.2 Spatial Reasoning (The Weakness)
Ask GPT-4V: "What are the exact pixel coordinates of the 'Submit' button?"
It often fails or gives an approximation (`[500, 500]` instead of `[512, 530]`).
*   **Why?** The patch system destroys fine-grained spatial information.
*   **Workaround:** **Set-of-Marks (SoM)**. We use a separate tool (Grounding DINO) to draw red numbered boxes on the image *before* feeding it to GPT-4V.
    *   *Prompt:* "Which number is the submit button?"
    *   *Image:* Has a Red Box labeled '1' over the button.
    *   *GPT-4V:* "Number 1."
    *   *Agent:* Looks up coordinates of Box 1.

---

## 5. Vision Hallucinations

Vision models hallucinate differently than text models.
1.  **Object Hallucination:** Claiming an object exists when it doesn't (usually because it often appears in similar contexts). E.g., seeing a "sink" in a kitchen photo where there is no sink.
2.  **OCR Hallucination:** Misreading blurry text. Reading "Il" as `11` or `ll`.
3.  **Directional Blindness:** Confusing Left and Right is surprisingly common in cheaper models.

**Defense:**
Always verify critical visual actions (e.g., extracting a bank account number) with a traditional deterministic OCR tool or by asking the model to "Check step-by-step".

---

## 6. Summary

Multimodal LLMs are the "Prefrontal Cortex" of a Vision Agent. They provide reasoning and high-level understanding.

*   **Architecture:** Vision Encoder + Projector + LLM.
*   **Resolution:** Critical for UI tasks. Use tiling strategies.
*   **Limits:** Weak at exact coordinates. Strong at semantic understanding.

In the next post, we will apply these MLLMs to a specific domain: **Screen Agents**. Building agents that can control your computer, browse the web, and use software.
