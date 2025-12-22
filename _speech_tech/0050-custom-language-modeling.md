---
title: "Custom Language Modeling"
day: 50
collection: speech_tech
categories:
  - speech-tech
tags:
  - asr
  - language-model
  - biasing
  - contextual-asr
  - production
difficulty: Hard
subdomain: "ASR Customization"
tech_stack: Kaldi, Whisper, Google Speech API
scale: "Adapting generic models to specific domains"
companies: Nuance, Google Cloud, Amazon Transcribe
related_dsa_day: 50
related_ml_day: 50
related_agents_day: 50
---

**"The model knows 'Apple' the fruit. It needs to learn 'Apple' the stock ticker."**

## 1. Introduction: The "Unique Name" Problem

You deploy a generic ASR model for a medical clinic.
Doctor says: "Patient needs 50mg of **Xarelto**."
ASR output: "Patient needs 50mg of **the rel toe**."

The model isn't "bad". It just mapped the sounds to the most probable English words it knows. "Xarelto" is rare. "The rel toe" is common.
To fix this without retraining the massive acoustic model ($1M cost), we use **Custom Language Models** (CLM) or **Contextual Biasing**.

---

## 2. How ASR Works (Simplified)

ASR is mathematically: `P(Text | Audio)`.
Using Bayes theorem, this splits into:
`P(Audio | Text) * P(Text)`

1. **Acoustic Model `P(Audio | Text)`**: "Does this sound match these phonemes?"
2. **Language Model `P(Text)`**: "Is this a valid sequence of words?"

The generic LM says `P("the rel toe") >> P("Xarelto")`.
We need to hack the LM to say: "In *this* clinic, `P("Xarelto")` is very high."

---

## 3. Techniques for Customization

### 3.1 Vocabulary Expansion (The "Word List")
The simplest customization. You upload a list of 500 domain-specific terms (Product names, employee names).
The ASR engine compiles a small, dynamic graph of these words and "boosts" their probability scores during the beam search decoding.

**Pros**: Fast, easy API update.
**Cons**: Can over-trigger. If you add "Cash", the model might hear "Cash" every time someone says "Catch".

### 3.2 Corpus Fine-Tuning
You upload 10,000 text documents (Medical Journals, Legal Briefs) from your domain.
You train a *text-only* Language Model (unsupervised) on this data.
You **interpolate** this small, specific LM with the huge, generic LM.
`Final_Score = 0.8 * Generic_LM + 0.2 * Custom_LM`.

**Pros**: Captures grammar and context ("Patient presents with..."), not just names.
**Cons**: Slower to deploy.

### 3.3 Contextual ASR (Real-Time hints)
This is the cutting edge (used by Siri/Alexa).
When you open your "Contacts" app, the phone sends your contact list to the ASR engine *for that session only*.
The ASR temporarily boosts `P("Arun")` because "Arun" is in your contacts.
Five minutes later, if you say "Arun" while in the Maps app, it might fail if it's not a common location.

This "Contextual Biasing" is dynamic and fleeting.

---

## 4. Architecture Implementation

How do we actually implement this with modern tools like **Whisper**?
Whisper is an "End-to-End" model (Acoustic + LM combined). It's hard to inject an external LM.

**Option A: Prompt Engineering**
Whisper accepts a `prompt` string.
If we pass `prompt="Xarelto, Ibuprofen, Tylenol"`, the Transformer's attention mechanism attends to those words.
When it hears ambiguity, it "copies" from the prompt.
*This works surprisingly well!*

**Option B: Shallow Fusion**
We run an external n-gram LM alongside Whisper's beam search.
At every decoding step, we check the external LM.
`Score = Whisper_Logit + Beta * External_LM_Logit`.

---

## 5. Metrics: WER isn't enough

If you just measure standard Word Error Rate (WER) on "The cat sat on the mat", your medical model looks fine.
We need **Entity-WER**.
- Accuracy on specific "entities" (Drug names, Locations).
- Finding "Xarelto" correctly is worth 100x more than missing "the".

---

## 6. Summary

Custom Language Modeling is the bridge between a "Demo" and a "Product".
A Demo works on generic YouTube videos.
A Product works for *your* users, speaking *your* jargon, in *your* noise conditions.
It allows us to "teach" the AI new vocabulary instantly, without the massive cost of teaching it new sounds.

---

**Originally published at:** [arunbaby.com/speech-tech/0050-custom-language-modeling](https://www.arunbaby.com/speech-tech/0050-custom-language-modeling/)

*If you found this helpful, consider sharing it with others who might benefit.*
