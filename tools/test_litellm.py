import litellm

# Define the custom model and local endpoint
model = "openai/cwm" # Use 'openai/' prefix for local OpenAI-compatible servers
api_base = "http://93.91.156.83:50120/v1"
api_key = "colin7d17d1"

response = litellm.completion(
    model=model,
    api_base=api_base,
    api_key=api_key,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a bash command showing the running containers created within 1 hour."}
    ],
    # Pass your extra template arguments here
    # chat_template_kwargs={"enable_thinking": False}
)

print(response)
print('------------------------------')
print(response.choices[0].message.content)
