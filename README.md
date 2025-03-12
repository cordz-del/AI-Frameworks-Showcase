# AI-Frameworks-Showcase
Showcasing AI frameworks and certifications with code examples and practical applications.

## Introduction

This repository showcases the AI frameworks and certifications that Robert Aaron Graham has mastered. It includes code examples and practical applications that demonstrate the skills acquired through various certifications and experiences.

## Skills & Certifications

### LLM Frameworks
- OpenAI GPT (GPT-3, GPT-4, ChatGPT)
- Anthropic Claude
- Meta's LLaMA
- Google PaLM / Gemini
- Hugging Face Transformers
- Cohere AI

### RAG (Retrieval-Augmented Generation) Frameworks
- LangChain (Your strong proficiency and favorite)
- LlamaIndex (formerly GPT-Index)
- Haystack
- Chroma (Vector Database Integration)
- Pinecone, Weaviate, FAISS (Vector DB backends)

### AI Automation & CI/CD
- Azure DevOps (Strongly experienced)
- GitHub Actions
- Jenkins
- Terraform
- AWS SageMaker (Strong experience)
- Google Vertex AI
- Docker & Kubernetes

### Additional Key AI Tools & APIs
- FastAPI / Flask
- AWS AI Services (Amazon Polly, Amazon Lex)
- Google Cloud AI APIs (Speech-to-Text, Text-to-Speech)
- Deepgram (Speech recognition & transcription)
- LiveKit (Real-time communication)

### Relevant Certifications Highlighting Expertise
- Stanford Machine Learning Certification
- Google Cloud Generative AI Certification
- Duke University LLMOps Certification
- API Testing Foundations Certification
- Test Automation Certification

## Code Snippet Examples

### OpenAI GPT Integration Example
```python
import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Define a prompt for the GPT model
prompt = "What is the capital of France?"

# Make an API call to OpenAI
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=50
)

# Print the response from the model
print(response.choices[0].text.strip())
```
This example demonstrates how to integrate with OpenAI's GPT model to generate text based on a given prompt.

### GitHub Actions CI/CD Example
```yaml
name: Python application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest
```
This GitHub Actions workflow sets up a Python environment, installs dependencies, and runs tests whenever code is pushed to the main branch or a pull request is made.
