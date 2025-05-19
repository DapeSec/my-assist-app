from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
import re

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import PubmedQueryRun
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Import BaseCallbackHandler for capturing agent thoughts
from langchain_core.callbacks import BaseCallbackHandler

try:
    from langchain_deepseek import ChatDeepseek
except ImportError:
    print("Warning: langchain_deepseek is not installed. DeepSeek LLM will not be available. Run 'pip install langchain-deepseek'")
    ChatDeepseek = None


# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# --- LangChain Setup ---

google_api_key = os.environ.get("GOOGLE_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")
ticketmaster_api_key = os.environ.get("TICKETMASTER_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

# --- API Key Validation ---
if not google_api_key:
    print("Warning: GOOGLE_API_KEY environment variable not set. Gemini LLM and Google Search tool may not be available.")
if not anthropic_api_key:
    print("Warning: ANTHROPIC_API_KEY environment variable not set. Claude LLM for code generation will not be available.")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not set. OpenAI LLM (ChatGPT) for code generation will not be available.")
if not deepseek_api_key:
    print("Warning: DEEPSEEK_API_KEY environment variable not set. DeepSeek LLM for code generation will not be available.")
if not ticketmaster_api_key:
     print("Warning: TICKETMASTER_API_KEY environment variable not set. Ticketmaster tool will not be available.")


# --- Initialize LLMs ---

llm_gemini = None
if google_api_key:
    try:
        # Using gemini-1.5-flash-latest for potentially better code generation capabilities
        llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, temperature=0.2)
    except Exception as e:
        print(f"Error initializing Gemini LLM: {e}")

llm_claude = None
if anthropic_api_key:
    try:
        llm_claude = ChatAnthropic(model="claude-3-7-sonnet-latest", anthropic_api_key=anthropic_api_key, temperature=0.2)
    except Exception as e:
        print(f"Error initializing Claude LLM: {e}")

llm_openai = None
if openai_api_key:
    try:
        llm_openai = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.2)
    except Exception as e:
        print(f"Error initializing OpenAI LLM: {e}")

llm_deepseek = None
if deepseek_api_key and ChatDeepseek:
    try:
        llm_deepseek = ChatDeepseek(model="deepseek-coder", api_key=deepseek_api_key, temperature=0.2)
    except Exception as e:
        print(f"Error initializing DeepSeek LLM: {e}")
elif not ChatDeepseek:
    print("DeepSeek LLM not initialized because langchain_deepseek library is missing.")


# --- Initialize Tools ---

search = None
if google_api_key and google_cse_id:
    try:
        search = GoogleSearchAPIWrapper(google_api_key=google_api_key, google_cse_id=google_cse_id)
    except Exception as e:
        print(f"Warning: Google Search tool could not be initialized. Ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are correct. Error: {e}")
        search = None
else:
     print("Warning: GOOGLE_API_KEY or GOOGLE_CSE_ID is missing. Google Search tool will not be available.")

pubmed = PubmedQueryRun()


# --- Define Ticketmaster Tool ---
def search_ticketmaster_events(query: str) -> str:
    """
    Searches the Ticketmaster API for events based on a query.
    Returns a formatted string summary of the events found.
    """
    if not ticketmaster_api_key:
        return "Ticketmaster API key is not set. Cannot perform event search."

    base_url = "https://app.ticketmaster.com/discovery/v2/events.json"
    params = {
        "apikey": ticketmaster_api_key,
        "keyword": query,
        "countryCode": "US",
        "size": 5
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        events = data.get('_embedded', {}).get('events', [])

        if not events:
            return f"No events found for '{query}' on Ticketmaster."

        event_list = []
        for event in events:
            name = event.get('name', 'N/A')
            date = event.get('dates', {}).get('start', {}).get('localDate', 'N/A')
            time = event.get('dates', {}).get('start', {}).get('localTime', '')
            venue_name = event.get('_embedded', {}).get('venues', [{}])[0].get('name', 'N/A')
            city = event.get('_embedded', {}).get('venues', [{}])[0].get('city', {}).get('name', 'N/A')
            state = event.get('_embedded', {}).get('venues', [{}])[0].get('state', {}).get('stateCode', 'N/A')
            event_url = event.get('url', 'N/A')

            event_list.append(
                f"- {name} on {date} {time} at {venue_name} ({city}, {state}). More info: {event_url}"
            )

        return f"Found the following events on Ticketmaster for '{query}':\n" + "\n".join(event_list)

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ticketmaster API: {e}")
        return f"An error occurred while searching for events on Ticketmaster for '{query}'."
    except Exception as e:
        print(f"An unexpected error occurred processing Ticketmaster data: {e}")
        return f"An unexpected error occurred while processing event data for '{query}'."

# --- Define Code Generation Tool Function ---
def generate_code_from_multiple_models(prompt: str) -> str:
    """
    Uses Gemini, Claude, OpenAI (ChatGPT), and DeepSeek LLMs to generate code based on a natural language prompt.
    This function attempts to call all available code generation models concurrently or sequentially
    and aggregates their results into a single formatted string.
    Returns a formatted string containing code from the available models, clearly labeled with status (Result, Error, Not_initialized).
    This string format is designed to be parsed by the Flask backend to structure the response.
    """
    results = {}
    # Instruction to guide the LLMs to produce only the code block
    code_gen_instruction = "Generate code based on the following request. Provide only the code block within markdown fences (```), and no other text:\n\n"
    full_prompt = code_gen_instruction + prompt

    # Helper to ensure content is wrapped in markdown code fences
    def ensure_markdown_code_block(text_content):
        # Check if the content already looks like a markdown code block
        if re.match(r"```(?:\w+)?\n.*?\n```", text_content.strip(), re.DOTALL):
            return text_content.strip()
        # If not, wrap it in a generic markdown code block
        return f"```\n{text_content.strip()}\n```"

    # Attempt generation with each model if initialized
    if llm_gemini:
        try:
            print("Generating code with Gemini...")
            generated_code_gemini = llm_gemini.invoke(full_prompt)
            results['gemini'] = {
                'status': 'result',
                'content': ensure_markdown_code_block(generated_code_gemini.content if hasattr(generated_code_gemini, 'content') else str(generated_code_gemini))
            }
        except Exception as e:
            print(f"Error during Gemini code generation: {e}")
            results['gemini'] = {'status': 'error', 'content': f"```\nError: {e}\n```"}
    else:
         results['gemini'] = {'status': 'not_initialized', 'content': "```\nGemini LLM is not initialized (API key missing or error).\n```"}

    if llm_claude:
        try:
            print("Generating code with Claude...")
            generated_code_claude = llm_claude.invoke(full_prompt)
            results['claude'] = {
                 'status': 'result',
                 'content': ensure_markdown_code_block(generated_code_claude.content if hasattr(generated_code_claude, 'content') else str(generated_code_claude))
            }
        except Exception as e:
            print(f"Error during Claude code generation: {e}")
            results['claude'] = {'status': 'error', 'content': f"```\nError: {e}\n```"}
    else:
        results['claude'] = {'status': 'not_initialized', 'content': "```\nClaude LLM is not initialized (API key missing or error).\n```"}

    if llm_openai:
        try:
            print("Generating code with OpenAI...")
            generated_code_openai = llm_openai.invoke(full_prompt)
            results['openai'] = {
                'status': 'result',
                'content': ensure_markdown_code_block(generated_code_openai.content if hasattr(generated_code_openai, 'content') else str(generated_code_openai))
            }
        except Exception as e:
            print(f"Error during OpenAI code generation: {e}")
            results['openai'] = {'status': 'error', 'content': f"```\nError: {e}\n```"}
    else:
        results['openai'] = {'status': 'not_initialized', 'content': "```\nOpenAI LLM is not initialized (API key missing or error).\n```"}

    if llm_deepseek:
        try:
            print("Generating code with DeepSeek...")
            generated_code_deepseek = llm_deepseek.invoke(full_prompt)
            results['deepseek'] = {
                'status': 'result',
                'content': ensure_markdown_code_block(generated_code_deepseek.content if hasattr(generated_code_deepseek, 'content') else str(generated_code_deepseek))
            }
        except Exception as e:
            print(f"Error during DeepSeek code generation: {e}")
            results['deepseek'] = {'status': 'error', 'content': f"```\nError: {e}\n```"}
    else:
        results['deepseek'] = {'status': 'not_initialized', 'content': "```\nDeepSeek LLM is not initialized (API key missing, library error, or other issue).\n```"}

    # Format the results into a single string with clear markers
    output_string = "###CODE_GENERATION_START###\n"
    for model, data in results.items():
        # Include the model name and its status (Result, Error, Not_initialized)
        output_string += f"--- {model.capitalize()} {data['status'].capitalize()} ---\n"
        # Include the content (code block or error message)
        output_string += data['content'] + "\n\n"

    # Add a message if no models were available or all failed
    if not any(r.get('status') == 'result' for r in results.values()):
         output_string += "No code generation models were available or all encountered errors."

    return output_string.strip()


# Create a list of tools the agent can use
tools = []
if search:
    tools.append(Tool(
        name="Google Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or find general information online"
    ))
if ticketmaster_api_key:
    tools.append(Tool(
        name="Ticketmaster Event Search",
        func=search_ticketmaster_events,
        description="useful for finding information about live events, concerts, sports, and theater performances. Input should be a query like 'concerts in London' or 'Lakers games'."
    ))
tools.append(Tool(
    name="PubMed Search",
    func=pubmed.run,
    description="useful for when you need to answer questions about medical literature, health, and biomedical topics. Input should be a medical query like 'causes of diabetes' or 'treatment for migraines'."
))
# Add the Code Generator tool if at least one code generation model is available
if llm_gemini or llm_claude or llm_openai or llm_deepseek:
     tools.append(Tool(
        name="Code Generator",
        func=generate_code_from_multiple_models,
        description="useful for when you need to generate code snippets or functions in various programming languages. This tool uses Gemini, Claude, OpenAI (ChatGPT), and DeepSeek models and provides results from each. Input should be a clear description of the code needed, e.g., 'Python function for quicksort' or 'JavaScript code for a button click event'."
    ))
else:
    print("Warning: No code generation models (Gemini, Claude, OpenAI, DeepSeek) are available. Code Generator tool will not be added.")


# Updated base prefix to reinforce the multi-model nature of the code tool
base_personalized_prefix = """You are a helpful AI assistant with access to several tools to assist users with their queries.
You are designed to be friendly and informative.
You respond in a witty and playful tone.
When asked about general knowledge or current events, use Google Search if available.
When asked about live events like concerts or sports, use the Ticketmaster Event Search tool.
When asked about medical or health-related topics, use the PubMed Search tool.
When the user explicitly asks for code or a script, use the Code Generator tool. The input to the Code Generator should be the user's request for the code. This tool provides code results from multiple models (Gemini, Claude, OpenAI, DeepSeek).
When you use the Code Generator tool, you will receive output containing code generated by the available models, clearly labeled. Present ALL the code results you receive from the tool to the user. Before listing the code blocks, provide a brief explanation of the potential differences or characteristics you observe between the code generated by the different models, if multiple results are available. You can comment on style, structure, approach, or any other notable variations.
Always try to provide a concise and helpful answer based on the information you find or generate.

Conversation History:
"""

agent = None
if llm_gemini: # Using Gemini as the orchestrator agent
    agent = initialize_agent(
        tools,
        llm_gemini,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": base_personalized_prefix
        }
    )
else:
    print("Error: Gemini LLM is not initialized. Cannot initialize the agent. Please check GOOGLE_API_KEY.")

# --- Custom Callback Handler to Capture Agent Thoughts ---
class AgentThoughtCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.thoughts = []

    def on_agent_action(self, action, **kwargs):
        # Capture the agent's thought process (the `Thought` and `Action` parts)
        self.thoughts.append(f"Thought: {action.log.strip()}")

    def on_tool_end(self, output, **kwargs):
        # Capture the tool's output
        self.thoughts.append(f"Tool Output: {output.strip()}")

    # Removed on_llm_end to avoid capturing intermediate LLM calls within the agent's reasoning
    # The final result is captured directly from agent.invoke()

    def clear_thoughts(self):
        self.thoughts = []

# --- Flask Routes ---
@app.route('/api/search', methods=['POST'])
def perform_search():
    data = request.get_json()
    query = data.get('query')
    chat_history = data.get('chat_history', [])

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    if not agent:
         return jsonify({'error': 'AI agent is not initialized. Check backend logs for API key issues.'}), 500

    # Initialize the callback handler for this specific request
    callback_handler = AgentThoughtCallbackHandler()

    try:
        # Format chat history for the agent's prefix
        history_string = "\n".join([f"{msg['sender'].capitalize()}: {msg['content']}" for msg in chat_history])
        full_prompt_for_agent = f"{base_personalized_prefix}\n{history_string}\nUser: {query}"

        # Invoke the LangChain agent with the callback handler
        # Use .invoke() with config to pass callbacks
        # The output key holds the final string result from the agent
        result = agent.invoke(
            {"input": full_prompt_for_agent},
            config={"callbacks": [callback_handler]}
        )['output']

        agent_thoughts = callback_handler.thoughts # Get the captured thoughts

        # Check if the result contains the code generation marker
        code_gen_marker = "###CODE_GENERATION_START###"
        if code_gen_marker in result:
            # Split the agent's response into introductory text and the raw code block output
            parts = result.split(code_gen_marker, 1)
            agent_intro_text = parts[0].strip()
            code_block_raw_text = parts[1].strip()

            code_results_list = []
            # Regex to find individual model outputs within the raw text
            # It looks for lines starting with --- ModelName Status --- and captures content until the next such line or end of string
            model_output_pattern = re.compile(r"--- (Gemini|Claude|OpenAI|DeepSeek) (Result|Error|Not_initialized) ---\n(.*?)(?=\n--- (?:Gemini|Claude|OpenAI|DeepSeek) (?:Result|Error|Not_initialized) ---|\Z)", re.DOTALL)

            # Iterate through all matches and parse them
            for match in model_output_pattern.finditer(code_block_raw_text):
                model_name = match.group(1).strip().lower()
                status_type = match.group(2).strip().lower()
                content = match.group(3).strip()

                status = 'success' if status_type == 'result' else status_type

                # Attempt to extract language if specified in markdown fence (e.g., ```python)
                language = 'text' # default language
                # Check if the content starts with a markdown code block fence
                code_block_match = re.match(r"```(\w+)?\n(.*)\n```", content, re.DOTALL)
                if code_block_match:
                    # If language is specified (e.g., python), capture it
                    if code_block_match.group(1):
                        language = code_block_match.group(1).strip().lower()
                    # Keep the full markdown block content
                    content = code_block_match.group(0).strip()
                else:
                    # If it's not a clear markdown block (e.g., an error message),
                    # ensure it's still treated as a code-like string or plain text.
                    # For simplicity, we'll keep the raw content and let the frontend handle rendering.
                     content = content.strip()


                code_results_list.append({
                    'model': model_name,
                    'content': content, # This will contain the full markdown block or error text
                    'status': status,
                    'language': language # Include detected language (or 'text' if none)
                })

            # Prepare the response data structure for code generation results
            code_response_data = {
                'type': 'code_response', # Indicate this is a code response
                'explanation': agent_intro_text if agent_intro_text else "Here's the code I generated for you:", # Agent's intro text
                'code_results': code_results_list, # List of results from each model
                'agent_thoughts': agent_thoughts # Include agent thoughts
            }

            # Create a summary for the chat history display
            history_summary = "Assistant provided code generation results."
            # Only store the final assistant message (code results summary) in history
            updated_history_for_frontend = chat_history + [
                {"sender": "user", "content": query},
                {"sender": "assistant", "content": history_summary, "type": "text"} # Store summary in history, marked as text
            ]
            code_response_data['updated_history'] = updated_history_for_frontend

            # Return the structured JSON response
            return jsonify(code_response_data)

        else:
            # If the code generation marker is not present, it's a regular text response from the agent
            updated_history_for_frontend = chat_history + [
                {"sender": "user", "content": query},
                {"sender": "assistant", "content": result, "type": "text"} # Store agent's text response, marked as text
            ]

            # Prepare the response data structure for text response
            response_data = {
                'type': 'text_response', # Indicate this is a text response
                'result': result, # The agent's final text output
                'updated_history': updated_history_for_frontend,
                'agent_thoughts': agent_thoughts # Include agent thoughts
            }

            # Return the JSON response
            return jsonify(response_data)

    except Exception as e:
        # Catch any exceptions during the agent's processing
        print(f"Error during LangChain processing: {e}")
        # Return an error response with agent thoughts up to the point of failure
        return jsonify({'type': 'error', 'error': f'An error occurred during the search: {str(e)}', 'agent_thoughts': callback_handler.thoughts}), 500

if __name__ == '__main__':
    # Run the Flask app if the agent was successfully initialized
    if agent:
        print("Flask app starting on port 5000...")
        # debug=True is useful for development, disable in production
        app.run(debug=True, port=5000)
    else:
        print("Flask app not started because the AI agent (Gemini for reasoning) could not be initialized. Check API keys and backend logs.")
