import ollama
from retriever import retrieve_context
from context_builder import build_context

def ask_agent(query: str, top_k=5, model="mistral"):
    print(f"\n🤖 Querying agent: {query}\n")

    context_blocks = retrieve_context(query, top_k=top_k)
    context_text = build_context(context_blocks)

    prompt = f"""
You're an assistant that has watched thousands of hours of fitness-related videos.
Your task is to answer the question based on actual quotes from these transcripts.

### CONTEXT:
{context_text}

### QUESTION:
{query}

### ANSWER:
"""

    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt.strip()}
    ])

    return response['message']['content']
