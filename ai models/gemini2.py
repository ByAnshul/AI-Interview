import os
import google.generativeai as genai

# --- Configuration ---
# It's good practice to get the API key from environment variables
# for security reasons, rather than hardcoding it directly in the script.
# Ensure your environment variable is set before running this script.
# For example, on Linux/macOS: export GEMINI_API_KEY="YOUR_API_KEY"
# On Windows (CMD): set GEMINI_API_KEY="YOUR_API_KEY"
# On Windows (PowerShell): $env:GEMINI_API_KEY="YOUR_API_KEY"
GEMINI_API_KEY = "AIzaSyDdYo1C1ka1ptCL_Q5oBIFyFOtQBnAITUU"
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable is not set.")
    print("Please set the GEMINI_API_KEY environment variable with your Gemini API key.")
    print("Example (Linux/macOS): export GEMINI_API_KEY='your_api_key_here'")
    print("Example (Windows CMD): set GEMINI_API_KEY='your_api_key_here'")
    exit(1)

# --- Configure the Generative AI library with your API Key ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit(1)

# --- Test API Functionality (Initial Check) ---
print("\nAttempting to list available models to verify API key...")
try:
    found_model = False
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods and "gemini-1.5-pro" in m.name:
            print(f"- Found suitable model: {m.name}")
            found_model = True
            break
    if not found_model:
        print("Warning: 'gemini-1.5-pro' or similar model not found among supported models.")
        print("Please check available models or try a different model name if you encounter issues.")
    print("\nGemini API key appears to be working correctly for basic access!")
except Exception as e:
    print(f"\nError: Failed to list models. The API key might be invalid or there's a network issue.")
    print(f"Details: {e}")
    print("Please double-check your GEMINI_API_KEY and your network connection.")
    exit(1) # Exit if the initial API check fails

# --- Research Paper Generation Functionality ---

def generate_research_paper_text():
    """
    Prompts the user for a template and current text, then uses a Gemini model
    to generate expanded or refined research paper content.
    """
    print("\n--- Research Paper Text Generation ---")
    print("Please provide the sample template for your research paper.")
    print("You can paste multiple lines; press Enter twice to finish input.")
    
    research_paper_template = []
    while True:
        line = input()
        if not line:
            break
        research_paper_template.append(line)
    research_paper_template = "\n".join(research_paper_template)

    print("\nNow, please provide your current research paper text.")
    print("You can paste multiple lines; press Enter twice to finish input.")
    
    current_research_paper_text = []
    while True:
        line = input()
        if not line:
            break
        current_research_paper_text.append(line)
    current_research_paper_text = "\n".join(current_research_paper_text)

    if not research_paper_template and not current_research_paper_text:
        print("No template or current text provided. Exiting text generation.")
        return

    print("\nGenerating final text output with Gemini...")

    # Choose a suitable model for long text generation
    # Using 'gemini-1.5-pro' for its large context window and strong capabilities
    model_name = "models/gemini-1.5-pro" 
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error: Could not load model '{model_name}'. Please ensure it's available and accessible.")
        print(f"Details: {e}")
        return

    # Construct the prompt for the AI
    prompt = f"""
You are an AI assistant specialized in academic writing. Your task is to help the user generate a comprehensive research paper text.
The user will provide a 'Sample Template' which outlines the structure or style desired for the research paper.
The user will also provide 'Current Research Paper Text' which is their existing content.

Your goal is to integrate the 'Current Research Paper Text' into the 'Sample Template' or expand upon it following the template's guidelines.
Ensure coherence, logical flow, and academic rigor. If the 'Current Research Paper Text' is sparse, use the template to generate relevant placeholder content or expand upon existing points.
If the 'Sample Template' is very detailed, adhere to its structure strictly. If it's general, use it as a guiding style.

--- Sample Template ---
{research_paper_template}

--- Current Research Paper Text ---
{current_research_paper_text}

--- Generated Research Paper Text ---
"""

    try:
        response = model.generate_content(prompt)
        # Access the text from the response. If candidates exist, pick the first one.
        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content:
            final_text = response.candidates[0].content.parts[0].text
            print("\n--- Final Generated Research Paper Text ---")
            print(final_text)
        else:
            print("\nError: No text content received from the model.")
            print(f"Full response: {response}") # Print full response for debugging
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Content was blocked due to: {response.prompt_feedback.block_reason}")
                print(f"Block metadata: {response.prompt_feedback.block_reason_metadata}")

    except Exception as e:
        print(f"\nAn error occurred during text generation: {e}")
        print("Please check your input and ensure the model is functioning correctly.")

# Call the generation function
if __name__ == "__main__":
    generate_research_paper_text()
