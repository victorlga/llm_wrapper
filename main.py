import argparse
from custom_llm_wrapper import CustomLLMWrapper  # Assume this is your implemented class

def main():
    parser = argparse.ArgumentParser(description="Interact with different LLMs through a unified interface.")
    parser.add_argument("--provider", help="Select the LLM provider", required=True, choices=['openai', 'huggingface'])
    parser.add_argument("--stream", help="Use streaming for responses (if supported by the LLM)", action="store_true")

    args = parser.parse_args()

    # Placeholder for dynamic model selection based on provider
    if args.provider == "openai":
        model_options = ["text-davinci-003", "gpt-3.5-turbo"]  # Example options for OpenAI
    elif args.provider == "huggingface":
        model_options = ["gpt2", "EleutherAI/gpt-neo-2.7B"]  # Example options for Hugging Face

    # Dynamically ask user to choose a model from the selected provider
    model_name = input(f"Select a model from {args.provider} ({', '.join(model_options)}): ")

    # Ensure the selected model is valid
    if model_name not in model_options:
        print("Invalid model selected. Please choose from the available options.")
        return

    # Example prompt
    prompt = input("Enter your prompt: ")

    # Initialize the wrapper
    wrapper = CustomLLMWrapper(provider=args.provider, model_name=model_name)

    # Check if streaming is requested and supported
    if args.stream and hasattr(wrapper, 'stream'):
        print("Streaming responses:")
        for chunk in wrapper.stream(prompt):
            print(chunk, end="", flush=True)
    else:
        # Fall back to the standard invoke method
        response = wrapper.invoke(prompt)
        print("Response:", response)

if __name__ == "__main__":
    main()
