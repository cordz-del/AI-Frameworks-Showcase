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

### LangChain RAG Example
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Initialize the vector store with Chroma
vector_store = Chroma("my_collection", embeddings=OpenAIEmbeddings())

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_vector_store(vector_store)

# Ask a question and get an answer
question = "What is the capital of France?"
answer = qa_chain.run(question)

print(f"Question: {question}")
print(f"Answer: {answer}")
```
This example demonstrates how to use LangChain for retrieval-augmented generation, integrating a vector store with Chroma and OpenAI embeddings.

### AWS SageMaker Model Deployment Example
```python
import boto3
from sagemaker import Session
from sagemaker.tensorflow import TensorFlowModel

# Initialize a SageMaker session
sagemaker_session = Session()

# Define the S3 path to the model artifact
model_artifact = "s3://my-bucket/my-model.tar.gz"

# Create a TensorFlow model
model = TensorFlowModel(
    model_data=model_artifact,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    framework_version="2.3",
    sagemaker_session=sagemaker_session
)

# Deploy the model to a SageMaker endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

print("Model deployed successfully.")
```
This example demonstrates how to deploy a TensorFlow model to an AWS SageMaker endpoint.

### FastAPI Example
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "value": "This is item " + str(item_id)}

# To run the app, use the command: uvicorn main:app --reload
```
This example demonstrates how to create a simple API with FastAPI, including a root endpoint and a parameterized endpoint.

### Dockerfile Example for Python Application
```Dockerfile
# Use the official Python image as a base
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install the required Python packages
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python", "app.py"]
```
This Dockerfile sets up a Python environment, installs dependencies, and runs a Python application.

### Terraform AWS EC2 Instance Example
```hcl
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2 AMI
  instance_type = "t2.micro"

  tags = {
    Name = "TerraformExample"
  }
}
```
This Terraform configuration provisions an AWS EC2 instance using the Amazon Linux 2 AMI in the us-west-2 region.
