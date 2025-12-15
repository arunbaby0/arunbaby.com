---
title: "Object Detection & Segmentation (YOLO, SAM)"
day: 24
collection: ai_agents
categories:
  - ai-agents
tags:
  - computer-vision
  - yolo
  - sam
  - object-detection
  - segmentation
  - on-device-ai
difficulty: Medium
related_dsa_day: 24
related_ml_day: 24
related_speech_day: 24
---

**"Knowing WHAT is where, and precisely WHERE it acts."**

## 1. Introduction: Bounding Boxes vs. Pixels

In Day 21, we introduced "Vision Fundamentals" (ViT/CLIP). Those give us *semantic* vectors. They tell us "There is a cat in this image".
But for an agent to *act* (pick up the cat, pet the cat, avoid the cat), it needs geometry.

There are two levels of geometric understanding:
1.  **Object Detection:** "The cat is in the box defined by `(x1, y1)` and `(x2, y2)`."
2.  **Segmentation:** "These exact 45,201 pixels belong to the cat."

For agents, Detection is usually enough for "Clicking" (UI). Segmentation is required for "Manipulation" (Robotics/Photo Editing).

---

## 2. Real-Time Detection: The YOLO Family

**YOLO (You Only Look Once)** is the standard for fast object detection. Unlike Transformers which are heavy, YOLO is a CNN designed for **Speed**.

### 2.1 How it works
*   **Grid:** Divide image into an SxS grid.
*   **Prediction:** Each cell predicts `B` bounding boxes and `C` class probabilities.
*   **Single Pass:** It does this detection in a single forward pass of the network (hence "Once").
*   **Speed:** YOLOv8 or YOLOv10 can run at 100+ FPS on a GPU, or 30 FPS on a Raspberry Pi.

### 2.2 Agent Use-Case: Video Understanding
If your agent is watching a video stream (e.g., Security Camera Agent), you cannot afford GPT-4V ($0.01/frame).
*   **Pipeline:**
    *   Run YOLO (cheap/fast) on every frame.
    *   If `Person` detected in `Restricted Zone`:
    *   Trigger `GPT-4V` (expensive/smart) to analyze: "Is this person a delivery driver or a burglar?"

This **Cascade Architecture** (Cheap Model -> Expensive Model) is the key to economic viability.

---

## 3. The Segmentation Revolution: SAM

**SAM (Segment Anything Model)** by Meta changed segmentation from a "Training" problem to a "Prompting" problem.
Before SAM, if you wanted to segment "Screws", you had to hand-label 10,000 screws.

### 3.1 Promptable Segmentation
SAM takes an image + a prompt.
*   **Point Prompt:** User clicks on a pixel. SAM expands outward to find the object boundary.
*   **Box Prompt:** User draws a box. SAM fills it.
*   **Text Prompt:** (With extensions like Grounded-SAM) "Find the screw".

### 3.2 The Usage for Agents
1.  **Robotics:**
    *   Agent sees a messy table.
    *   Proprioception: "I need to pick up the red cup."
    *   Detection: Finds bounding box of "red cup".
    *   SAM: Generates a mask for the cup.
    *   Grasp Planner: Calculates a grasp point *inside* the mask, avoiding the handle.
2.  **Photo Editing Agents:**
    *   User: "Make the sky blue."
    *   Agent: `SAM.predict(prompt="sky")`.
    *   Agent: Applies blue filter only to the mask.

---

## 4. Open Vocabulary Detection (Grounding DINO)

Standard YOLO is trained on COCO (80 classes: Person, Car, Dog...). It doesn't know what a "WiFi Router" is.
**Grounding DINO** allows you to detect objects using arbitrary text prompts.

*   **Mechanism:** It fuses a BERT (Language) model with a Transformer (Vision) model.
*   **Input:** Image + Text: "Find me the green lego block."
*   **Output:** Bounding Box.

**Recipe for a General Purpose Vision Tool:**
```python
def find_item(image_path, text_query):
    # 1. Use Grounding DINO to get the Box
    boxes = grounding_dino.predict(image_path, text_query)
    
    # 2. Use SAM to get the Mask (if needed)
    masks = sam.predict(image_path, boxes)
    
    return boxes, masks
```
This function allows an agent to find *anything* without retraining.

---

## 5. Fine-Tuning for Specialized Domains

Sometimes "General" isn't enough.
*   **Medical:** Detecting distinct types of tumors.
*   **Manufacturing:** Detecting microscopic defects on a PCB.
*   **Satellite:** Detecting specific tank models.

In these cases, we must Fine-Tune.
1.  **Data Engine:** Use a large model (SAM) to "Pre-label" data.
2.  **Human Verification:** Human corrects the masks (much faster than drawing from scratch).
3.  **Distillation:** Train a smaller, faster model (like YOLO) on this new dataset to run on the edge device.

---

## 6. Summary

*   **YOLO:** Use for high-speed, fixed-class detection (Security, Traffic).
*   **Grounding DINO:** Use for open-world, text-based detection (Agents).
*   **SAM:** Use when you need pixel-perfect precision (Robotics, Editing).

Combining these creates the "Spatial Awareness" loop. The agent knows *what* is there and exactly *where* it is. In the final post of this vision section, we will look at **Video Analysis and Real-World Perception**.
