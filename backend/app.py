from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
import re
import json # Import json for parsing potential tool output
from typing import Union, Dict, Any # Import Union, Dict, Any for type hinting compatibility
import time # Import time for rate limiting

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import PubmedQueryRun
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate # For YouTube summarization prompt

# Import BaseCallbackHandler for capturing agent thoughts
from langchain_core.callbacks import BaseCallbackHandler

# Import for YouTube Transcript API
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

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

# --- Rate Limiting Configuration for YouTube Summarizer ---
# Simple global rate limiting: max_requests in a time_window_seconds
YOUTUBE_RATE_LIMIT_MAX_REQUESTS = 3 # Example: Allow 3 requests
YOUTUBE_RATE_LIMIT_TIME_WINDOW_SECONDS = 60 # Example: per 60 seconds

# Variables to track rate limit state
youtube_request_count = 0
youtube_last_reset_time = time.time()


# --- LangChain Setup ---

google_api_key = os.environ.get("GOOGLE_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")
ticketmaster_api_key = os.environ.get("TICKETMASTER_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY") # Alpha Vantage API Key

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
if not alpha_vantage_api_key: # Validation for Alpha Vantage API Key
    print("Warning: ALPHA_VANTAGE_API_KEY environment variable not set. Financial data and analysis tools will not be available.")


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

# --- Define Financial Data Tool ---
def get_stock_data(symbol: str) -> str:
    """
    Fetches real-time stock data for a given stock ticker symbol using Alpha Vantage.
    Returns a formatted string summary of the stock's current price and other metrics.
    """
    if not alpha_vantage_api_key:
        return "Alpha Vantage API key is not set. Cannot fetch stock data."

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol.upper(),
        "apikey": alpha_vantage_api_key
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()

        global_quote = data.get("Global Quote", {})
        if not global_quote:
            return f"No real-time data found for stock symbol '{symbol}'. It might be an invalid symbol or data is not available."

        # Extract relevant data
        symbol_out = global_quote.get("01. symbol", "N/A")
        open_price = global_quote.get("02. open", "N/A")
        high_price = global_quote.get("03. high", "N/A")
        low_price = global_quote.get("04. low", "N/A")
        price = global_quote.get("05. price", "N/A")
        volume = global_quote.get("06. volume", "N/A")
        latest_trading_day = global_quote.get("07. latest trading day", "N/A")
        previous_close = global_quote.get("08. previous close", "N/A")
        change = float(data.get("09. change", 0)) # Ensure change is float for analysis
        change_percent = data.get("10. change percent", "0%")

        # Return as a JSON string to make parsing easier for analysis tool
        return json.dumps({
            "symbol": symbol_out,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "price": price,
            "volume": volume,
            "latest_trading_day": latest_trading_day,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent
        })

    except requests.exceptions.RequestException as e:
        print(f"Error calling Alpha Vantage API for {symbol}: {e}")
        return json.dumps({"error": f"An error occurred while fetching stock data for '{symbol}'. Please try again later."})
    except Exception as e:
        print(f"An unexpected error occurred processing Alpha Vantage data for {symbol}: {e}")
        return json.dumps({"error": f"An unexpected error occurred while processing stock data for '{symbol}'."})

# --- Define Stock Analysis Tool ---
def analyze_stock(symbol: str) -> str:
    """
    Analyzes real-time stock data for a given stock ticker symbol.
    Fetches data using the Financial Data tool and provides a brief analysis.
    Input should be a stock ticker symbol, e.g., 'AAPL'.
    """
    # Use the get_stock_data function to fetch the raw data
    raw_data_json = get_stock_data(symbol)

    try:
        data = json.loads(raw_data_json)

        if "error" in data:
            return f"Could not analyze stock {symbol}: {data['error']}"
        if not data or data.get("symbol") == "N/A":
             return f"Could not analyze stock {symbol}: No data found or invalid symbol."

        symbol_out = data.get("symbol", "N/A")
        price = float(data.get("price", 0))
        open_price = float(data.get("open", 0))
        high_price = float(data.get("high", 0))
        low_price = float(data.get("low", 0))
        previous_close = float(data.get("previous_close", 0))
        change = float(data.get("change", 0))
        change_percent_str = data.get("change_percent", "0%")
        volume = data.get("volume", "N/A")
        latest_trading_day = data.get("latest_trading_day", "N/A")

        # Basic Analysis
        analysis_report = f"Analysis for {symbol_out} (as of {latest_trading_day}):\n"
        analysis_report += f"  Current Price: ${price:.2f}\n"
        analysis_report += f"  Change Today: {change:.2f} ({change_percent_str})\n"

        if previous_close != 0:
            if price > previous_close:
                analysis_report += "  The stock is trading higher than its previous close.\n"
            elif price < previous_close:
                analysis_report += "  The stock is trading lower than its previous close.\n"
            else:
                analysis_report += "  The stock is trading at the same price as its previous close.\n"

        price_range = high_price - low_price
        if price_range > 0:
            # Calculate position within the day's range
            position_in_range = (price - low_price) / price_range
            if position_in_range > 0.75:
                analysis_report += "  It is currently trading near the high of the day.\n"
            elif position_in_range < 0.25:
                analysis_report += "  It is currently trading near the low of the day.\n"
            else:
                analysis_report += "  It is trading within the mid-range of the day.\n"
        else:
             analysis_report += "  High and low prices for the day are the same.\n"


        analysis_report += f"  Volume: {volume}\n"
        # Add a disclaimer
        analysis_report += "\nDisclaimer: This analysis is based on real-time data and provides a snapshot of the stock's performance today. It is not financial advice. Consult a qualified financial advisor for personalized investment recommendations."

        return analysis_report

    except json.JSONDecodeError:
        return f"Error processing financial data for {symbol}."
    except ValueError:
        return f"Error converting data to numbers for analysis for {symbol}. Data might be incomplete or malformed."
    except Exception as e:
        print(f"An unexpected error occurred during stock analysis for {symbol}: {e}")
        return f"An unexpected error occurred while analyzing stock data for '{symbol}'."


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

# --- Define YouTube Summarizer/Notes Tool ---
def get_youtube_video_id(url: str) -> Union[str, None]:
    """
    Extracts the YouTube video ID from a given URL.
    Supports various YouTube URL formats.
    """
    # Regex patterns for different YouTube URL formats
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def summarize_and_take_notes_youtube_video(video_url: str) -> Union[Dict[str, Any], str]:
    """
    Fetches the transcript of a YouTube video, summarizes it, and takes key notes using an LLM.
    Requires a YouTube video URL as input.
    Includes basic rate limiting.
    Returns a dictionary containing 'summary' and 'notes' if successful, or a string error message.
    """
    global youtube_request_count, youtube_last_reset_time

    # --- Apply Rate Limiting ---
    current_time = time.time()
    # Reset count if time window has passed
    if current_time - youtube_last_reset_time > YOUTUBE_RATE_LIMIT_TIME_WINDOW_SECONDS:
        youtube_request_count = 0
        youtube_last_reset_time = current_time

    # Check if rate limit is exceeded
    if youtube_request_count >= YOUTUBE_RATE_LIMIT_MAX_REQUESTS:
        wait_time = youtube_last_reset_time + YOUTUBE_RATE_LIMIT_TIME_WINDOW_SECONDS - current_time
        return f"Rate limit exceeded for YouTube summarization and notes. Please wait {wait_time:.1f} seconds before trying again."

    # Increment request count if within limit
    youtube_request_count += 1
    print(f"YouTube summarizer/notes request count: {youtube_request_count} in the current window.")
    # --- End Rate Limiting ---


    video_id = get_youtube_video_id(video_url)
    if not video_id:
        return "Could not extract a valid YouTube video ID from the provided URL. Please provide a valid YouTube video URL."

    try:
        # Get transcript
        # Set languages to try, including English ('en') and potentially auto-generated ('a.en')
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'a.en'])
        full_transcript = " ".join([entry['text'] for entry in transcript_list])

        if not full_transcript.strip():
            return "The YouTube video has no detectable transcript or the transcript is empty."

        # Use an LLM to summarize and take notes from the transcript
        if llm_gemini:
            # Using a prompt that asks for both summary and notes
            # Instruct the LLM to use clear markers for parsing
            summarization_notes_prompt = PromptTemplate(
                template="Summarize the following YouTube video transcript concisely and then provide key notes or bullet points from the content. Use '###SUMMARY###' before the summary and '###NOTES###' before the notes.\n\nTranscript:\n{transcript}\n\n---\n\n###SUMMARY###\n\n###NOTES###\n",
                input_variables=["transcript"]
            )
            chain = summarization_notes_prompt | llm_gemini
            summary_notes_result = chain.invoke({"transcript": full_transcript})

            # Attempt to parse the response into summary and notes using markers
            response_text = summary_notes_result.content if hasattr(summary_notes_result, 'content') else str(summary_notes_result)

            summary = "Summary not found."
            notes = "Notes not found."

            summary_match = re.search(r"###SUMMARY###\s*(.*?)(?:\n###NOTES###|\Z)", response_text, re.DOTALL)
            if summary_match:
                summary = summary_match.group(1).strip()

            notes_match = re.search(r"###NOTES###\s*(.*?)\Z", response_text, re.DOTALL)
            if notes_match:
                notes = notes_match.group(1).strip()

            # Return a structured dictionary
            return {"summary": summary, "notes": notes}

        else:
            return "Gemini LLM is not initialized, cannot summarize or take notes from the YouTube video. Please ensure GOOGLE_API_KEY is set."

    except NoTranscriptFound:
        return "No transcript found for this YouTube video. The video might not have captions, or they are not in an accessible format."
    except TranscriptsDisabled:
        return "Transcripts are disabled for this YouTube video by the creator."
    except Exception as e:
        print(f"Error summarizing/taking notes from YouTube video {video_id}: {e}")
        # Provide a more specific error message for unexpected exceptions during transcript fetching
        return f"An error occurred while fetching the YouTube transcript for video ID '{video_id}': {str(e)}. This might be due to YouTube blocking the request from your IP address or a temporary issue."


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
if alpha_vantage_api_key: # Add Financial Data and Analysis Tools
    tools.append(Tool(
        name="Financial Data",
        func=get_stock_data,
        description="useful for getting real-time stock market data for a given ticker symbol. Input should be a stock ticker symbol, e.g., 'AAPL' or 'MSFT'. Returns raw JSON data."
    ))
    tools.append(Tool(
        name="Stock Analyzer",
        func=analyze_stock,
        description="useful for analyzing real-time stock market data for a given stock ticker symbol and providing a brief summary. Input should be a stock ticker symbol, e.g., 'AAPL' or 'MSFT'."
    ))
else:
    print("Warning: Alpha Vantage API key is missing. Financial Data and Stock Analyzer tools will not be available.")

# Add the Code Generator tool if at least one code generation model is available
if llm_gemini or llm_claude or llm_openai or llm_deepseek:
     tools.append(Tool(
        name="Code Generator",
        func=generate_code_from_multiple_models,
        description="useful for when you need to generate code snippets or functions in various programming languages. This tool uses Gemini, Claude, OpenAI (ChatGPT), and DeepSeek models and provides results from each. Input should be a clear description of the code needed, e.g., 'Python function for quicksort' or 'JavaScript code for a button click event'."
    ))
else:
    print("Warning: No code generation models (Gemini, Claude, OpenAI, DeepSeek) are available. Code Generator tool will not be added.")

# Add the YouTube Summarizer/Notes tool
if llm_gemini: # YouTube summarization depends on an initialized LLM, Gemini is preferred here
    tools.append(Tool(
        name="YouTube Summarizer/Notes",
        func=summarize_and_take_notes_youtube_video,
        description="useful for summarizing the content of a YouTube video and taking key notes. Input should be the full YouTube video URL, e.g., '[https://www.youtube.com/watch?v=exampleid](https://www.youtube.com/watch?v=exampleid)'. I will extract the transcript and provide a summary and notes in a structured format."
    ))
else:
    print("Warning: Gemini LLM is not initialized. YouTube Summarizer/Notes tool will not be available.")


# Updated base prefix to reinforce the multi-model nature of the code tool, financial data, and analysis
base_personalized_prefix = """You are a helpful AI assistant with access to several tools to assist users with their queries.
You are designed to be friendly and informative.
You respond in a witty and playful tone.
When asked about general knowledge or current events, use Google Search if available.
When asked about live events like concerts or sports, use the Ticketmaster Event Search tool.
When asked about medical or health-related topics, use the PubMed Search tool.
When the user explicitly asks for code or a script, use the Code Generator tool. The input to the Code Generator should be a clear description of the code needed. This tool provides code results from multiple models (Gemini, Claude, OpenAI, DeepSeek).
When you use the Code Generator tool, you will receive output containing code generated by the available models, clearly labeled. Present ALL the code results you receive from the tool to the user. Before listing the code blocks, provide a brief explanation of the potential differences or characteristics you observe between the code generated by the different models, if multiple results are available. You can comment on style, structure, approach, or any other notable variations.
When the user asks for *real-time financial data* for a specific stock, use the Financial Data tool. The input should be a stock ticker symbol (e.g., 'AAPL').
When the user asks for an *analysis* or *insights* about a specific stock based on real-time data, use the Stock Analyzer tool. The input should be a stock ticker symbol (e.g., 'AAPL').
When the user asks for *general financial advice* (e.g., "Should I invest in X?", "What's a good investment strategy?", "How do I save for retirement?"), leverage the Google Search tool to find general, reliable financial advice, and then summarize it. **Crucially, explicitly state that you are an AI and cannot provide personalized financial advice, and that they should consult a qualified financial advisor for their specific situation.**
When the user provides a YouTube video URL and asks for a summary or notes, use the YouTube Summarizer/Notes tool. The input to this tool should be the full YouTube video URL.
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
        # Store the raw output for potential parsing later
        self.thoughts.append(f"Tool Output: {output}") # Store raw output

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

        # --- Check for YouTube Summarizer/Notes Tool Output ---
        # We need to inspect the tool output captured in the thoughts
        youtube_tool_output = None
        for thought in agent_thoughts:
            if thought.startswith("Tool Output: "):
                # Attempt to parse the tool output as JSON (dictionary)
                try:
                    # The tool output is stored as a string representation of the dictionary
                    # We need to carefully extract and parse it.
                    # A simple approach is to look for the start and end of the dictionary representation.
                    # This is fragile and depends on the exact string format returned by the tool.
                    # A more robust approach might involve the tool returning a specific marker + JSON string.
                    # For now, let's assume the tool output string is the direct string representation of the dict or an error string.

                    # Check if the output looks like our structured dictionary output
                    # A simple check: does it contain "summary" and "notes" keys?
                    if '"summary":' in thought and '"notes":' in thought:
                         # Extract the dictionary string part
                         json_string_match = re.search(r"(\{.*\})", thought, re.DOTALL)
                         if json_string_match:
                             json_string = json_string_match.group(1)
                             youtube_tool_output = json.loads(json_string)
                             print("Successfully parsed YouTube tool output as JSON.")
                             break # Found the YouTube tool output, stop searching
                         else:
                             print("Could not find JSON string pattern in YouTube tool output.")
                    elif "Rate limit exceeded" in thought or "No transcript found" in thought or "Transcripts are disabled" in thought or "An error occurred while fetching the YouTube transcript" in thought:
                         # It's a known error message string from the YouTube tool
                         youtube_tool_output = thought.replace("Tool Output: ", "").strip()
                         print(f"Detected YouTube tool error message: {youtube_tool_output}")
                         break # Found the YouTube tool output (error), stop searching


                except json.JSONDecodeError as e:
                    print(f"Could not decode YouTube tool output as JSON: {e}")
                    # Keep youtube_tool_output as None, proceed to check for other response types
                except Exception as e:
                    print(f"An unexpected error occurred while processing YouTube tool output: {e}")
                    # Keep youtube_tool_output as None


        if youtube_tool_output and isinstance(youtube_tool_output, dict):
            # It's a successful structured YouTube summary/notes response
            summary = youtube_tool_output.get('summary', 'Summary not available.')
            notes = youtube_tool_output.get('notes', 'Notes not available.')

            updated_history_for_frontend = chat_history + [
                {"sender": "user", "content": query},
                # Store a simplified representation in history, frontend can render full details
                {"sender": "assistant", "content": "Assistant provided YouTube summary and notes.", "type": "text"}
            ]

            # Prepare the response data structure for YouTube summary/notes
            response_data = {
                'type': 'youtube_summary_notes_response', # Indicate this is a YouTube summary/notes response
                'summary': summary,
                'notes': notes,
                'updated_history': updated_history_for_frontend,
                'agent_thoughts': agent_thoughts # Include agent thoughts
            }

            return jsonify(response_data)

        elif youtube_tool_output and isinstance(youtube_tool_output, str):
             # It's an error message string from the YouTube tool
             updated_history_for_frontend = chat_history + [
                {"sender": "user", "content": query},
                {"sender": "assistant", "content": youtube_tool_output, "type": "text"} # Store the error message
             ]

             response_data = {
                'type': 'text_response', # Treat error as a text response
                'result': youtube_tool_output, # The error message
                'updated_history': updated_history_for_frontend,
                'agent_thoughts': agent_thoughts # Include agent thoughts
            }
             return jsonify(response_data)

        # --- Check for Code Generation Output ---
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
            # If neither YouTube nor Code Generation output is detected, it's a regular text response
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
