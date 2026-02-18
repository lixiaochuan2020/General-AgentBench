#!/bin/bash
#
# Local Test Script for MathHay DeepSeek-V3.2
# This script runs on a local server without SLURM
# Purpose: Test if haystack_len=128000 works with original MathHay + HuggingFace/Novita
#

set -e

echo "========================================="
echo "MathHay Local Test: DeepSeek-V3.2"
echo "========================================="
echo "Model: deepseek-ai/DeepSeek-V3.2-Exp"
echo "Provider: HuggingFace (Novita)"
echo "Context: 128k tokens (haystack_len=128000)"
echo "========================================="
echo ""

# Change to MathHay directory
cd /home/ubuntu/agentic-long-bench/MathHay

# Load environment from mathhay_keys
if [ -f /home/ubuntu/agentic-long-bench/MathHay/scripts/mathhay_keys ]; then
    echo "Loading API keys from mathhay_keys..."
    set -a
    source /home/ubuntu/agentic-long-bench/MathHay/scripts/mathhay_keys
    set +a
fi

# Verify HuggingFace API key
if [ -z "$HUGGINGFACE_API_KEY" ] && [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HUGGINGFACE_API_KEY or HF_TOKEN environment variable is not set!"
    exit 1
fi

# Export HF_TOKEN if only HUGGINGFACE_API_KEY is set
if [ -z "$HF_TOKEN" ] && [ -n "$HUGGINGFACE_API_KEY" ]; then
    export HF_TOKEN="$HUGGINGFACE_API_KEY"
fi

echo "HuggingFace API key: Configured ✓"
echo ""

# Check if data directory exists
DATA_DIR="./outputs/data/March-2024-to-September-2024"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo ""
    echo "The original MathHay requires benchmark data in this directory."
    echo "Required files:"
    echo "  - documents.json"
    echo "  - full_haystack_question_3s3d.json"
    exit 1
fi

# Always use direct API test with real data (skip run_multimodel_evaluation.py)
echo "Testing API with real MathHay data..."
echo ""
    
    # Run a direct API test using real MathHay data
    echo "Testing DeepSeek-V3.2 API with real MathHay data (via LiteLLM)..."
    python3 << 'PYTHON_SCRIPT'
import os
import sys
import json
import asyncio
import tiktoken

# Use LiteLLM like all_in_one does
import litellm
litellm.drop_params = True
litellm.modify_params = True

# Get API key
hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_KEY')
if not hf_token:
    print("ERROR: No HuggingFace token found")
    sys.exit(1)

# Set HuggingFace token for LiteLLM
os.environ['HUGGINGFACE_API_KEY'] = hf_token

print(f"Token: {hf_token[:10]}...{hf_token[-4:]}")

# LiteLLM model format (requires huggingface/ prefix and :provider suffix)
model_id = "huggingface/deepseek-ai/DeepSeek-V3.2:fireworks-ai"

print(f"\nLiteLLM Settings:")
print(f"  model: {model_id}")
print(f"  drop_params: {litellm.drop_params}")
print(f"  modify_params: {litellm.modify_params}")

enc = tiktoken.encoding_for_model('gpt-4o')

# Load real MathHay data
data_dir = "/home/ubuntu/agentic-long-bench/MathHay/outputs/data/March-2024-to-September-2024"
questions_file = os.path.join(data_dir, "full_haystack_question_3s3d.json")
documents_file = os.path.join(data_dir, "documents.json")

print(f"\nLoading MathHay data from {data_dir}...")

with open(questions_file, 'r') as f:
    questions = json.load(f)

with open(documents_file, 'r') as f:
    documents = json.load(f)

print(f"Loaded {len(questions)} questions and {len(documents)} documents")

# Use first question as test case
row = questions[0]
question = row["Task"]["question"]
print(f"\nTest Question: {question[:100]}...")

# Build context - use 200K to test context limit
haystack_len = 200000  # Test 200K context limit

# Get relevant documents (3 docs for 3s3d task)
relevant_docs = row.get("Documents", [])
relevant_text = "\n\n".join(relevant_docs)
relevant_tokens = enc.encode(relevant_text)
print(f"Relevant document tokens: {len(relevant_tokens):,}")

# Get irrelevant documents to fill haystack
irrelevant_indices = row.get("Irrelevant_Documents_Indexs", [])
irrelevant_docs = [documents[idx]["Document"] for idx in irrelevant_indices]
irrelevant_text = "\n\n".join(irrelevant_docs)
irrelevant_tokens = enc.encode(irrelevant_text)
print(f"Irrelevant document tokens: {len(irrelevant_tokens):,}")

# Calculate how many irrelevant tokens we need
prompt_len = len(enc.encode(f"Question: {question}")) + 500
rest_tokens = max(0, haystack_len - len(relevant_tokens) - prompt_len)
print(f"Rest tokens needed: {rest_tokens:,}")

# Build full haystack (place relevant docs in middle of irrelevant docs)
irrelevant_tokens_trimmed = irrelevant_tokens[:rest_tokens]
placement_index = len(irrelevant_tokens_trimmed) // 2

# Insert relevant tokens in the middle
full_tokens = (irrelevant_tokens_trimmed[:placement_index] + 
               relevant_tokens + 
               irrelevant_tokens_trimmed[placement_index:])
full_text = enc.decode(full_tokens)

print(f"Full haystack tokens: {len(full_tokens):,}")

# Build final prompt
prompt = f"""Here are some documents:

{full_text}

Based on the above documents, answer the following question:
{question}

Please show your reasoning step by step."""

prompt_tokens = len(enc.encode(prompt))
print(f"\nFinal prompt tokens: {prompt_tokens:,}")

# Test the API call using LiteLLM
async def test_litellm(messages, tools=None, max_tokens=2000, temperature=0.7, verbose=False):
    """Test LiteLLM acompletion"""
    kwargs = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    
    if verbose:
        print("\n" + "="*60)
        print("FULL REQUEST TO LITELLM:")
        print("="*60)
        print(f"model: {kwargs['model']}")
        print(f"temperature: {kwargs['temperature']}")
        print(f"max_tokens: {kwargs['max_tokens']}")
        print(f"messages: [{len(messages)} messages]")
        for i, m in enumerate(messages):
            content_preview = m['content'][:200] + '...' if len(m['content']) > 200 else m['content']
            print(f"  [{i}] role={m['role']}, content_len={len(m['content'])}, preview: {content_preview[:100]}...")
        if tools:
            print(f"tools: [{len(tools)} tools]")
            print(f"tool_choice: {kwargs['tool_choice']}")
            print("\nFULL TOOLS JSON:")
            print(json.dumps(tools, indent=2, ensure_ascii=False))
        print("="*60 + "\n")
    
    response = await litellm.acompletion(**kwargs)
    return response

# Define tools like all_in_one does
tools = [
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit your final answer to the math problem",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final numerical answer"
                    },
                    "reasoning": {
                        "type": "string", 
                        "description": "Step by step reasoning"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]

print(f"\nSending request to DeepSeek-V3.2 via LiteLLM (WITH TOOLS)...")
messages = [{"role": "user", "content": prompt}]

# First test: WITH tools (like all_in_one does)
print("\n" + "#"*60)
print("# TEST 1: WITH TOOLS")
print("#"*60)
try:
    response = asyncio.run(test_litellm(messages, tools=tools, verbose=True))
    print(f"\n✅ SUCCESS (with tools)!")
    print(f"Response ({len(response.choices[0].message.content or '')} chars):")
    content = response.choices[0].message.content or ""
    print(content[:500] + "..." if len(content) > 500 else content)
except Exception as e:
    error_msg = str(e)
    print(f"\n❌ FAILED (with tools): {error_msg}")

# Second test: WITHOUT tools
print("\n" + "#"*60)
print("# TEST 2: WITHOUT TOOLS")
print("#"*60)
try:
    response = asyncio.run(test_litellm(messages, tools=None, verbose=True))
    print(f"\n✅ SUCCESS (without tools)!")
    print(f"Response ({len(response.choices[0].message.content or '')} chars):")
    content = response.choices[0].message.content or ""
    print(content[:500] + "..." if len(content) > 500 else content)
except Exception as e:
    error_msg = str(e)
    print(f"\n❌ FAILED (without tools): {error_msg}")

print("\n========================================")
print("Test complete!")
print("========================================")
PYTHON_SCRIPT

