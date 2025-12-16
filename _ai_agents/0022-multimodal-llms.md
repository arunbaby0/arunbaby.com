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

**"One model to rule them all: The Unification of Text and Vision."**

## 1. Introduction: The End of Separate Systems

In Day 21, we discussed the "Composability" approach to vision. We built a stack: `Image -> GroundingDINO -> Bounding Box -> SAM -> Mask -> Classifier -> Label`.
This works, but it is **brittle**. It suffers from the "Telephone Game" problem (error propagation). If the Detector misses the box by 10 pixels, the Segmenter fails, and the Classifier sees noise. The agent fails.

This parallels the early days of NLP (2010s), where we had separate pipelines for Tokenization, Part-of-Speech Tagging, Dependency Parsing, and Sentiment Analysis. Transformers killed that pipeline by merging everything into one model.

**Multimodal Large Language Models (MLLMs)** effectively collapse the entire vision pipeline into a single Transformer key.
Instead of extensive Python glue code coordinating 4 different neural networks, we now have:
`Image + Text Prompt -> MLLM -> Text Answer`

You can simply pass an image of a server dashboard and ask, "Is the memory utilization trending up?" The model performs detection (finding the chart), reading (OCR on the axis), and reasoning (trend analysis) in a single forward pass.

In this deep dive, we will explore the architecture of these giants (GPT-4V, LLaVA, Gemini Vision), how they actually "eat" images, and the specific engineering challenges of using them in production agents.

---

## 2. Architecture: How do you feed an Image to an LLM?

LLMs (Llama 3, GPT-4) are fundamentally text-processing machines. They take a sequence of integers (tokens) and predict the next integer. They do not have a slide slot for a JPEG. So, how do we shove an image into the prompt?

### 2.1 The Projector Strategy (The LLaVA Approach)
**LLaVA (Large Language and Vision Assistant)** is the pioneering open-source architecture that demystified this process. Most open MLLMs (BakLLaVA, Yi-VL) follow this recipe.

1.  **The Vision Encoder (Frozen):**
    We take a pre-trained Vision Transformer, usually **CLIP (ViT-L/14)**. This model is already excellent at turning images into dense semantic vectors.
    *   *Input:* 1 Image.
    *   *Output:* A grid of feature vectors. For a 336x336 image, CLIP outputs a 24x24 grid = **576 vectors**. Each vector is size 1024.
2.  **The Projector (The Adapter):**
    These 576 vectors live in "CLIP Vision Space". But our LLM (e.g., Llama 3) expects vectors in "Llama Text Space". These are different mathematical languages.
    *   We train a simple **Linear Projection Layer** (a matrix multiplication, sometimes a 2-layer MLP).
    *   $Vector_{text} = W \cdot Vector_{vision} + b$.
    *   This "translates" the visual concept of "Cat" into the vector that Llama understands as the word "Cat".
3.  **The LLM (Frozen-ish):**
    We treat these 576 projected vectors as if they were just 576 text tokens. We **Proppend** them to the user's text prompt.
    *   *Input Sequence:* `[Img_Tok_1] [Img_Tok_2] ... [Img_Tok_576] "Describe this image."`
    *   The LLM attends to these visual tokens exactly as it attends to text context.

**Training Strategy:**
*   **Stage 1 (Feature Alignment):** Freeze LLM and Vision Encoder. Train *only* the Projector on massive image-caption pairs (CC3M). This teaches the model that "Image of Cat" = "Text Cat".
*   **Stage 2 (Visual Instruction Tuning):** Unfreeze the LLM (or use LoRA). Train on complex instructions ("Look at this image and write a poem"). This teaches reasoning.

### 2.2 The Native Approach (GPT-4V / Gemini)
While LLaVA glues two models together, models like **GPT-4V** and **Gemini 1.5 Pro** are hypothesized to be more **Native**. They are likely trained from scratch on interleaved documents containing both text and images.
*   *Advantage:* Deeper integration. The model doesn't just "see" the image as a prefix; it understands the interplay of text *inside* the image and the surrounding text.
*   *Capability:* This enables analyzing massive PDFs with charts, where the text refers to the chart ("As seen in Figure 1...").

---

## 3. The Resolution Dilemma

The single biggest bottleneck for Vision Agents today is **Resolution**.

### 3.1 The 224px / 336px Problem
Recall that CLIP is trained on low-res images (224x224 or 336x336). This is tiny.
*   **Scenario:** You screenshot an AWS Billing Console to ask the agent "Why is my bill high?".
*   **The Crunch:** The browser screenshot is 1920x1080. The model squashes this to 336x336.
*   **The Result:** The text "$5,403.20" becomes a gray smudge of 4 pixels. The agent says, "I cannot see the bill amount."

### 3.2 Dynamic High-Res Cropping (The Solution)
State-of-the-art models (GPT-4o, LLaVA-Next) use a **Tiling Strategy** to overcome this without retraining the core vision encoder on massive images (which is computationally prohibitive).

**The Algorithm:**
1.  **Global View:** Resize the full image to 336x336. Pass it to the encoder. This gives the "Gist" or context ("This is a billing dashboard").
2.  **Tile Generation:** Cut the original high-res 1080p image into fixed patches, say 512x512 tiles.
    *   A 1080p image might result in 6 tiles.
3.  **Encoding:** Pass *each* of the 6 tiles through the encoder separately.
4.  **Concatenation:** The prompt to the LLM becomes:
    `[Global_Tokens] [Tile_1_Tokens] [Tile_2_Tokens] ... [Tile_6_Tokens] "Analyze the cost."`

**The Trade-off:**
*   **Pros:** It can read tiny text and see fine details.
*   **Cons:** **Token Explosion.**
    *   Low Res: 576 tokens.
    *   High Res (6 tiles): 576 * 7 = ~4,000 tokens.
    *   *Cost:* Viewing one image might cost $0.04 instead of $0.005. Latency increases from 2s to 10s.

---

## 4. Visual Capabilities for Agents

What can these MLLM agents actually *do* that simpler models couldn't?

### 4.1 OCR-Free Reading (Document Intelligence)
Traditional OCR (Tesseract) gives you a "Bag of Words". It loses structure.
*   *Tesseract Output:* "Invoice Date Total 12/01/2023 $500". (Is $500 the total or a line item? Is the date the invoice date or due date?).
*   *MLLM Output:* "This is a table. Row 1 matches column 'Total' with '$500'."
MLLMs inherently understand **Visual Layout**. They can read charts, handwriting, forms, and complex PDFs by seeing the spatial relationships between text blocks.

### 4.2 Spatial Reasoning (The Weakness)
Ask GPT-4V: "What are the exact pixel coordinates of the 'Submit' button?"
It often fails. It might say `[500, 500]` when the button is at `[512, 530]`.
*   **Why?** The patch system destroys fine-grained spatial information. The LLM knows the button is "in the bottom right", but it doesn't know the exact pixel.
*   **Workaround: Set-of-Marks (SoM).**
    We use a separate tool (Grounding DINO) to pre-process the image. We draw Red Bounding Boxes with numbers `[1]`, `[2]`, `[3]` on every button.
    *   *Prompt:* "Which number is the submit button?"
    *   *Image:* Has a Red Box labeled '1' over the button.
    *   *GPT-4V:* "Number 1."
    *   *Agent:* Looks up the coordinates of Box 1 in a dictionary.

---

## 5. Vision Hallucinations

Vision models hallucinate differently than text models. You must build defenses.

1.  **Object Hallucination:** Claiming an object exists when it doesn't.
    *   *Cause:* Statistical co-occurrence. In the training data, "Kitchen" images almost always have "Sinks". If you show a blurry kitchen corner without a sink, the model might "hallucinate" a sink because its priors are so strong.
2.  **OCR Hallucination:** Misreading blurry text.
    *   Common errors: Reading "Il" (Capital i, lowercase L) as `11` or `ll`. Reading serial numbers wrong.
    *   *Defense:* If reading a critical number (e.g., Bank Account), ask the model to output it, but *also* verify it using a Regex or a specialized OCR tool.
3.  **Directional Blindness:** Confusing Left and Right.
    *   "The cup is to the left of the plate." (It's actually to the right).
    *   *Why:* Random flipping is a common data augmentation technique in training vision models. This inadvertently teaches the model that "Orientation doesn't matter".

---

## 6. Summary

Multimodal LLMs are the "Prefrontal Cortex" of a Vision Agent. They provide reasoning and high-level understanding that specialized tools lack.

*   **Projection:** The magic trick that turns Image Vectors into Text Tokens.
*   **High-Res Tiling:** The definition of "Seeing clearly". Without it, agents are legally blind to UI details.
*   **Document Intelligence:** The ability to understand layout, not just text.
*   **Set-of-Marks:** The bridge between "Seeing" and "Clicking".

In the next post, we will apply these MLLMs to a specific domain: **Screen Agents**. We will build agents that can control your computer, browse the web, and use software by "seeing" the screen just like you do.
