# Benchmarking and Understanding LLM Fine-Tuning Techniques

This repository contains resources and summaries from presentations on fine-tuning Large Language Models (LLMs). The goal is to provide a clear overview of the different techniques, their applications, and the practical considerations involved in adapting pre-trained models for specific tasks.

## Table of Contents
* [Introduction to Fine-Tuning](#introduction-to-fine-tuning)
* [Fine-Tuning Techniques](#fine-tuning-techniques)
    * [Supervised Fine-Tuning](#supervised-fine-tuning)
    * [Unsupervised Fine-Tuning](#unsupervised-fine-tuning)
    * [Reinforcement Learning-Based Tuning](#reinforcement-learning-based-tuning)
* [Classification of Techniques](#classification-of-techniques)
    * [By Model Size](#classification-by-model-size)
    * [By Use Case](#classification-by-use-case)
* [The Fine-Tuning Process](#the-fine-tuning-process )
* [Evaluation](#evaluation)
* [Platforms and Frameworks](#platforms-and-frameworks)
* [Hardware Requirements](#hardware-requirements)
* [Challenges](#challenges)
* [Conclusion](#conclusion)


## Introduction to Fine-Tuning
Large Language Models (LLMs) are powerful AI models trained on vast amounts of text data, enabling them to understand and generate human-like language. While pre-trained models are highly capable, fine-tuning is the process of adapting them to more specific tasks or domains. This involves further training the model on a smaller, targeted dataset, which enhances its performance for specialized applications while retaining its foundational knowledge.

The primary challenge is selecting the most efficient fine-tuning technique in terms of cost, performance, and resource utilization.

## Fine-Tuning Techniques
There are several approaches to fine-tuning, each with its own trade-offs. They can be broadly categorized as follows:

### Supervised Fine-Tuning
This method uses labeled datasets to adapt the model to a specific task.

* **Full Fine-Tuning**: Updates all the weights and biases of the pre-trained model. While comprehensive, it is computationally expensive.

* **Instruction Fine-Tuning**: Trains the model on examples formatted as instructions and desired responses, teaching it to follow specific commands.

* **Adapter**s: Adds small, trainable layers to the existing transformer model and only fine-tunes these new layers.

* **Parameter-Efficient Fine-Tuning (PEFT)**:
These methods update only a small subset of the model's parameters, reducing computational and memory costs significantly.
  
  * **Low-Rank Adaptation (LoRA)**: Injects small, trainable low-rank matrices into the transformer layers. Only these matrices are updated, drastically reducing the number of trainable parameters.

  * **Prefix Tuning & Prompt Tuning**: Involves adding trainable tokens to the input prompts without changing the underlying model weights.

### Unsupervised Fine-Tuning
This approach is used when labeled data is unavailable. The model is trained on a large corpus of unlabeled text from a target domain (e.g., legal or medical documents) to improve its understanding of that specific language.

* **Domain-Specific Fine-Tuning**: Adapts a general LLM to a particular industry or subject by training it on domain-specific texts.
  
* **Distillation** : it  is a technique where a smaller model is trained to mimic the behavior of a larger model.

### Reinforcement Learning-Based Tuning
This advanced technique uses human feedback to align the model's outputs with human values and subjective quality.

* **Reinforcement Learning from Human Feedback (RLHF)**: Humans rank different model responses, and this feedback is used to train a "reward model." The LLM is then fine-tuned to maximize the score from this reward model.
  
---

## Classification of Techniques
With the growing variety and diverse use cases of fine-tuning, the challenge of selecting the most efficient technique has become a major consideration. In this presentation, we will explore this critical problem and discuss how to make an informed choice.

### Classification by Model Size
The choice of fine-tuning technique often depends on the size of the LLM. Large language models (LLMs) differ significantly in size, this size directly affects their fine-tuning approaches. Generally we categorize these models as small, medium, or large.

| Size   | Parameters         | Notes                          |	Recommended Techniques      |
|--------|--------------------|--------------------------------|------------------------------|
| Small  | < 300M             | Fast training, limited power   | 	Full Fine-Tuning            |
| Medium | 300M – 3B          | Balanced performance and cost  |  Full FT, LoRA, Adapters     |
| Large  | > 3B               | High accuracy, costly training |  LoRA, QLoRA, Adapters, RLHF |



### Classification by Use Case
There are several use cases of the techniques of fine-tuning, but we can summarize them in the following table:

| Use Case                  | Typical Model Size | Common Fine-Tuning Methods                         |
|---------------------------|--------------------|----------------------------------------------------|
| Instruction Following     | Small to Large     | Full FT (small models), LoRA/Adapters (large models) |
| Chatbots/Conversational AI| Medium to Large    | RLHF, SFT, LoRA                                   |
| Text Classification       | Small to Medium    | Full FT, LoRA, Adapters                           |
| Text Generation           | Medium to Large    | Full FT, LoRA, Prompt Tuning                      |
| Multimodal (Text+Image)   | Medium to Large    | LoRA, Adapters                                    |

---

## The Fine-Tuning Process 


1. **Install Dependencies**  
   `transformers`, `datasets`, `accelerate`, `torch`, etc.

2. **Prepare Dataset**  
   Load and preprocess your task-specific data.

3. **Load Pretrained Model and Tokenizer**  
   Use Hugging Face or another framework.

4. **Set Training Arguments**  
   Define `output_dir`, `batch_size`, `learning_rate`, `num_train_epochs`, etc.

5. **Train the Model**  
   Fine-tune using the appropriate technique.

6. **Evaluate and Save**  
   Assess performance and export the trained model.

---


## Evaluation
Evaluating an LLM's performance is crucial and involves both automated and human-centric methods.

* Automatic Metrics:

  * Text Classification: Accuracy, F1 Score
  
  * Text Generation: Perplexity, BLEU Score
  
  * Summarization: ROUGE Score
  
  * Translation: BLEU Score, METEOR

* Human Evaluations: Used when automatic metrics are insufficient.

  * Pairwise Comparisons: Raters choose the better of two model responses.
  
  * Rating Scales: Raters score outputs based on criteria like coherence and relevance.
  
  * A/B Testing: Comparing two model versions in a live environment.

Popular evaluation frameworks include Hugging Face's evaluate library, LangChain's evaluation modules, and OpenAI Evals.

---

##  Platforms and Frameworks

Several platforms and tools can facilitate the fine-tuning process:

- **Hugging Face Hub**  
  A comprehensive platform with tools, libraries, pre-trained models, and datasets.
  
- **Cloud Platforms**  
  - Google Vertex AI : User-friendly, excellent for multimedia projects.
  - AWS SageMaker : Highly customizable for complex configurations.
  - Azure AI Studio : Offers a suite of tools for model development.

- **Frameworks**  
  - PyTorch / TensorFlow 
  - LLaMA Factory  
  - OpenAI API 

---

## Hardware Requirements
The choice of hardware is dictated by the model size and fine-tuning technique.

| Model Size   | CPU                      | GPU (VRAM)                   | RAM             |
| :----------- | :----------------------- | :--------------------------- | :-------------- |
| Small (<300M)  | Intel i7/i9, Ryzen 7/9   | NVIDIA RTX 3090/4090 (24 GB) | 64 GB+          |
| Medium (300M-3B)| Server-class Xeon/EPYC   | NVIDIA A100 (40 GB), H100 (80 GB) | 256 GB - 1 TB   |
| Large (3B+) | Multiple high-end CPUs   | 4-16+ A100/H100 (80 GB)      | Multiple TBs    |

Example: LoRA on a LLaMA-7B Model
* **Sub-optimal Hardware**: RTX 3060 (12GB), 16GB RAM -> **10-14 hours** training time.

* **Appropriate Hardware**: RTX 3090 (24GB), 64GB RAM ->** 3-5 hours** training time.

---

## Challenges

- Limited hardware
- Lack of labeled datasets
- Long training time for large models
- High cloud costs

---

## Conclusion

Choosing the right fine-tuning strategy requires balancing performance, cost, and infrastructure. This repository helps you benchmark techniques, understand their requirements, and implement them effectively for your LLM projects.

---

