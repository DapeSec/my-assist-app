MyAssist: An AI-Powered Code and Text Generation AssistantDescriptionMyAssist is a full-stack AI assistant designed to streamline your workflow by providing both text-based answers and code generation capabilities. The application features a responsive React frontend with a modern user interface, integrating speech recognition for voice input and text-to-speech for audible responses. It communicates with a Python Flask backend that leverages multiple cutting-edge AI models (Google Gemini, Anthropic Claude, OpenAI, and DeepSeek) and specialized tools (Google Search, PubMed, Ticketmaster) to process queries and deliver relevant text explanations or functional code snippets.Key features include:Voice Input: Use speech recognition to interact with the AI hands-free.Text-to-Speech: Hear the AI's responses read aloud.Multi-Model Code Generation: Receive code snippets from various leading AI models, presented in a syntax-highlighted, copyable format with model-specific comparisons.Intelligent Tool Use: The AI can leverage Google Search for general knowledge, PubMed for medical literature, and Ticketmaster for event information.Conversation History: View a chronological record of your interactions.Agent Thoughts: Optionally view the underlying thought process of the AI agent for deeper insight into its reasoning.Theming: Toggle between light and dark modes for comfortable viewing.SummaryMyAssist acts as an intelligent companion, empowering users to quickly generate code, obtain information, and interact with AI through both voice and text. By orchestrating multiple powerful AI models and specialized tools, it enhances productivity for developers and general users alike, providing comprehensive and context-aware assistance.Table of ContentsFeaturesTechnologies UsedSetup and InstallationPrerequisitesAPI KeysFrontend SetupBackend SetupRunning the ApplicationUsageProject StructureContributingLicenseFeaturesInteractive Chat Interface: A clean and intuitive UI for seamless conversations.Speech Recognition: Powered by react-speech-recognition for hands-free query input.Text-to-Speech (TTS): Uses window.speechSynthesis to vocalize AI responses, with mute functionality.Dynamic UI Animations: Siri-like pulse effects indicate listening or speaking states.Code Highlighting: Code blocks are rendered with react-syntax-highlighter using the Dracula theme for readability.Copy to Clipboard: Easily copy generated code snippets with a single click.Theming: Toggle between light and dark modes.Error Handling: Displays user-friendly error messages and provides verbose backend logging.Agent Thought Visibility: Option to expand and review the AI's thought process during complex tasks.Technologies UsedFrontendReact.js: A JavaScript library for building user interfaces.Tailwind CSS: A utility-first CSS framework for rapid UI development.Lucide React: A set of beautiful and customizable open-source icons.react-speech-recognition: React hooks for speech recognition.regenerator-runtime: A runtime for async/await functions, required for react-speech-recognition.react-syntax-highlighter: For syntax highlighting of code blocks.BackendPython: The core programming language.Flask: A lightweight web framework for building the API.Flask-CORS: Enables Cross-Origin Resource Sharing for frontend-backend communication.python-dotenv: For loading environment variables (API keys).requests: For making HTTP requests to external APIs (e.g., Ticketmaster).LangChain: A framework for developing applications powered by language models.langchain-google-genai: Integration for Google Gemini models.langchain-anthropic: Integration for Anthropic Claude models.langchain-openai: Integration for OpenAI models (ChatGPT).langchain-deepseek: Integration for DeepSeek models (optional, requires separate installation).langchain-community: Provides tools like GoogleSearchAPIWrapper and PubmedQueryRun.langchain-core: Core LangChain components, including BaseCallbackHandler for capturing agent thoughts.Setup and InstallationFollow these steps to get MyAssist up and running on your local machine.PrerequisitesBefore you begin, ensure you have the following installed:Node.js (LTS version recommended) and npm (or yarn)Python 3.8+pip (Python package installer)API KeysMyAssist relies on several external APIs for its functionality. You will need to obtain API keys for the services you wish to enable. Create a .env file in the backend directory and populate it with your keys:# Google API Keys (for Gemini LLM and Google Search)
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
GOOGLE_CSE_ID=YOUR_CUSTOM_SEARCH_ENGINE_ID

# Anthropic API Key (for Claude LLM)
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY

# OpenAI API Key (for OpenAI LLM)
OPENAI_API_KEY=YOUR_OPENAI_API_KEY

# DeepSeek API Key (for DeepSeek LLM - optional)
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY

# Ticketmaster API Key (for Event Search tool)
TICKETMASTER_API_KEY=YOUR_TICKETMASTER_API_KEY

Google API Key & CSE ID: Needed for the GoogleSearchAPIWrapper tool and ChatGoogleGenerativeAI (Gemini). You can get these from the Google Cloud Console by enabling the Custom Search API and generating an API key. For GOOGLE_CSE_ID, you'll need to set up a Custom Search Engine and link it to your API key.Anthropic API Key: Obtain from the Anthropic console.OpenAI API Key: Obtain from the OpenAI platform.DeepSeek API Key: Obtain from the DeepSeek API platform. This LLM and its integration are optional.Ticketmaster API Key: Register as a developer and get a key from the Ticketmaster Developer Portal.Frontend SetupClone the repository:git clone <your-repository-url>
cd <your-repository-name>
Navigate to the frontend directory:cd frontend
Install dependencies:npm install
# or
yarn install
Backend SetupNavigate to the backend directory:cd backend
Create a virtual environment:It's highly recommended to use a virtual environment to manage Python dependencies.python -m venv venv
Activate the virtual environment:macOS/Linux:source venv/bin/activate
Windows (Command Prompt):venv\Scripts\activate.bat
Windows (PowerShell):.\venv\Scripts\Activate.ps1
Install backend dependencies:The requirements.txt file (which you'll need to create based on the pip install commands below) lists all necessary Python packages.pip install Flask Flask-CORS python-dotenv requests langchain-google-genai langchain-anthropic langchain-openai langchain-community
# Optional: for DeepSeek LLM
pip install langchain-deepseek
Make sure to create a requirements.txt file by running pip freeze > requirements.txt after installing all dependencies.Running the ApplicationAfter setting up both the frontend and backend, you can run the application.Start the Backend Server:Open your terminal, navigate to the backend directory, and activate your virtual environment. Then run:python app.py
The backend server should start on http://127.0.0.1:5000. You'll see warnings in the console if any API keys are missing or if LLMs/tools fail to initialize.Start the Frontend Development Server:Open a new terminal window, navigate to the frontend directory.npm start
# or
yarn start
The frontend application will open in your browser, usually at http://localhost:3000.UsageType your query: Enter your question or command into the input field at the bottom.Use Voice Input: Click the microphone icon to start speaking your query. Click it again to stop.Search/Submit: Press Enter or click the search icon to send your query to the AI.Listen to Responses: If not muted, the AI will automatically read its responses aloud. You can click the Volume2 icon to replay the last assistant message or the VolumeX icon to mute/unmute.Copy Code: For code generation responses, use the "Copy" button in the code block header to quickly copy the code to your clipboard.Toggle Dark Mode: Click the Sun/Moon icon in the top right to switch themes.View Agent Thoughts: Click the "Agent Thoughts" button to expand and see the detailed reasoning steps of the AI agent for a given response.Project Structure.
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js           # Main React component, contains all UI logic
│   │   ├── index.js         # React entry point
│   │   └── ...              # Other frontend assets like index.css
│   ├── package.json         # Frontend dependencies and scripts
│   ├── tailwind.config.js   # Tailwind CSS configuration
│   └── ...                  # Other frontend configuration files
└── backend/
    ├── app.py               # Flask application, LangChain agent, LLM/Tool initialization
    ├── requirements.txt     # Python dependencies (generated after setup)
    ├── .env                 # Your environment variables (API keys)
    ├── .env.example         # Example of the .env file structure
    └── ...                  # Any other backend files
ContributingContributions are welcome! If you'd like to improve MyAssist, please feel free to open issues or submit pull requests.LicenseThis project is open-source and available under the MIT License.