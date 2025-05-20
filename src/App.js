import React, { useState, useEffect } from 'react';
import 'tailwindcss/tailwind.css';
import { Mic, Search, Volume2, VolumeX, Moon, Sun, Copy, ChevronDown, ChevronUp } from 'lucide-react';
import 'regenerator-runtime';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism'; // Using Dracula theme for code highlighting

// Custom CSS for animations and toggle switch
const customStyles = `
  @keyframes siri-pulse-outer {
    0% {
      transform: scale(0.7);
      opacity: 0.6;
      box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
    }
    50% {
      transform: scale(1.1);
      opacity: 1;
      box-shadow: 0 0 25px 18px rgba(59, 130, 246, 0.4);
    }
    100% {
      transform: scale(0.7);
      opacity: 0.6;
      box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
    }
  }

  @keyframes siri-pulse-outer-dark {
    0% {
      transform: scale(0.7);
      opacity: 0.6;
      box-shadow: 0 0 0 0 rgba(96, 165, 250, 0.7);
    }
    50% {
      transform: scale(1.1);
      opacity: 1;
      box-shadow: 0 0 25px 18px rgba(96, 165, 250, 0.4);
    }
    100% {
      transform: scale(0.7);
      opacity: 0.6;
      box-shadow: 0 0 0 0 rgba(96, 165, 250, 0.7);
    }
  }

  @keyframes siri-pulse-inner {
    0% {
      transform: scale(0.6);
      opacity: 0.8;
    }
    50% {
      transform: scale(1.3);
      opacity: 0.4;
    }
    100% {
      transform: scale(0.6);
      opacity: 0.8;
    }
  }

  .animate-siri-pulse-outer {
    animation: siri-pulse-outer 2s infinite ease-in-out;
  }

  .dark .animate-siri-pulse-outer {
     animation: siri-pulse-outer-dark 2s infinite ease-in-out;
  }

   .animate-siri-pulse-inner {
    animation: siri-pulse-inner 1.5s infinite ease-in-out;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  .animate-spin-custom {
    animation: spin 1s linear infinite;
  }

  .icon-static {
    opacity: 1;
    transform: scale(1);
    box-shadow: none;
    transition: opacity 0.3s ease-in-out, transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
  }

  .siri-inner-circle {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      border-radius: 50%;
      background-color: rgba(255, 255, 255, 0.3);
      transform: scale(0.6);
      opacity: 0.8;
  }

  .dark .siri-inner-circle {
       background-color: rgba(255, 255, 255, 0.1);
  }

  .toggle-switch {
    position: relative;
    display: inline-block;
    width: 60px;
    height: 34px;
  }

  .toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    transition: .4s;
    border-radius: 34px;
  }

  .slider:before {
    position: absolute;
    content: "";
    height: 26px;
    width: 26px;
    left: 4px;
    bottom: 4px;
    background-color: white;
    transition: .4s;
    border-radius: 50%;
  }

  input:checked + .slider {
    background-color: #4CAF50;
  }

  .dark input:checked + .slider {
     background-color: #4CAF50;
  }

  input:focus + .slider {
    box-shadow: 0 0 1px #4CAF50;
  }

  input:checked + .slider:before {
    transform: translateX(26px);
  }

  /* Custom styles for code editor blocks */
  .code-editor-block {
      border: 1px solid #e2e8f0; /* light-gray-200 */
      border-radius: 0.5rem; /* rounded-lg */
      overflow: hidden;
      margin-top: 0.5rem; /* my-2 */
      margin-bottom: 0.5rem;
      background-color: #f7fafc; /* gray-100 */
  }

  .dark .code-editor-block {
      border-color: #4a5568; /* dark-gray-600 */
      background-color: #1a202c; /* dark-gray-900 */
  }

  .code-editor-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.5rem 0.75rem; /* p-2 px-3 */
      background-color: #edf2f7; /* gray-200 */
      border-bottom: 1px solid #e2e8f0; /* light-gray-200 */
      color: #4a5568; /* gray-700 */
      border-top-left-radius: 0.5rem;
      border-top-right-radius: 0.5rem;
  }

  .dark .code-editor-header {
      background-color: #2d3748; /* dark-gray-700 */
      border-bottom-color: #4a5568; /* dark-gray-600 */
      color: #e2e8f0; /* dark-gray-200 */
  }

  .code-editor-header span {
      font-size: 0.875rem; /* text-sm */
      font-weight: 500; /* font-medium */
  }

  .code-editor-copy-button {
      padding: 0.25rem 0.5rem; /* px-2 py-1 */
      background-color: #3b82f6; /* blue-600 */
      color: white;
      border-radius: 0.375rem; /* rounded-md */
      display: flex;
      align-items: center;
      font-size: 0.75rem; /* text-xs */
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
      transition: background-color 0.2s ease-in-out;
  }

  .code-editor-copy-button:hover {
      background-color: #2563eb; /* blue-700 */
  }

   .code-editor-copy-button:focus {
      outline: none;
      ring: 2;
      ring-color: #3b82f6; /* blue-500 */
      ring-opacity: 50;
   }

  .copy-success-message {
      font-size: 0.875rem; /* text-sm */
      color: #10b981; /* green-600 */
      margin-right: 0.5rem; /* mr-2 */
      transition: opacity 0.3s ease-in-out;
      opacity: 1;
  }

  .dark .copy-success-message {
      color: #34d399; /* green-400 */
  }

  .code-syntax-highlighter {
      padding: 0.8rem !important; /* Override default padding */
      font-size: 0.85rem !important; /* text-sm */
      overflow-x: auto !important;
      margin-top: 0 !important;
      border-radius: 0 0 0.5rem 0.5rem !important;
  }

  .dark .code-syntax-highlighter {
       background-color: #1a202c !important; /* dark-gray-900 */
  }
`;


function App() {
  const [query, setQuery] = useState('');
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  // State to manage copy success messages for each code block
  const [copySuccessMessages, setCopySuccessMessages] = useState({});
  const [showThoughts, setShowThoughts] = useState(false); // State to toggle thought visibility

  // Speech recognition hooks
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();
  const [isVoiceInputActive, setIsVoiceInputActive] = useState(false);

  // Text-to-speech state
  const [synth, setSynth] = useState(null);
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [isMuted, setIsMuted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Dark mode state
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme) {
      return storedTheme === 'dark';
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  // Apply dark mode class to HTML root element
  useEffect(() => {
    const root = window.document.documentElement;
    if (isDarkMode) {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  // Update query state with transcript when voice input is active
  useEffect(() => {
    if (isVoiceInputActive) {
      setQuery(transcript);
    }
  }, [transcript, isVoiceInputActive]);

  // Initialize speech synthesis
  useEffect(() => {
    const synth = window.speechSynthesis;
    setSynth(synth);

    const loadVoices = () => {
      const availableVoices = synth.getVoices();
      setVoices(availableVoices);
      // Attempt to find a suitable English voice, default to the first if none found
      const defaultVoice = availableVoices.find(voice => voice.lang.startsWith('en'));
      setSelectedVoice(defaultVoice || availableVoices[0]);
    };

    // Load voices immediately if available, otherwise wait for the 'voiceschanged' event
    if (synth && synth.getVoices().length > 0) {
      loadVoices();
    } else if (synth) {
      synth.addEventListener('voiceschanged', loadVoices);
    }

    // Cleanup function to remove event listener and cancel speech on unmount
    return () => {
      if (synth) {
        synth.removeEventListener('voiceschanged', loadVoices);
        synth.cancel();
        setIsSpeaking(false);
      }
    };
  }, [synth]); // Added synth to dependency array

  // Handle sending the search query to the backend
  const handleSearch = async (searchQuery) => {
    if (!searchQuery.trim()) {
      setError('Please enter a query or use voice input.');
      return;
    }

    setLoading(true);
    setError('');
    setCopySuccessMessages({}); // Clear copy messages on new search

    // Cancel any ongoing speech before starting a new search
    if (synth && synth.speaking) {
      synth.cancel();
      setIsSpeaking(false);
    }

    // Add user query to conversation history
    setConversation(prevConv => [...prevConv, { sender: 'user', content: searchQuery }]);

    try {
      // Send query and chat history to the backend API
      const response = await fetch('http://127.0.0.1:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: searchQuery, chat_history: conversation }),
      });

      // Check if the response was successful
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Process the response based on its type
      if (data.type === 'code_response') {
          // Filter out previous code generation summaries if they exist to avoid duplicates
          setConversation(prevConv => [
            ...prevConv.filter(msg => msg.sender !== 'assistant' || msg.content !== 'Assistant provided code generation results.'),
            {
              sender: 'assistant',
              type: 'code',
              explanation: data.explanation,
              code_results: data.code_results,
              agent_thoughts: data.agent_thoughts || [] // Capture thoughts
            }
          ]);
      } else if (data.type === 'youtube_summary_notes_response') {
           // Handle the new YouTube summary/notes response type
           setConversation(prevConv => [
               ...prevConv,
               {
                   sender: 'assistant',
                   type: 'youtube_summary_notes', // New type identifier for frontend rendering
                   summary: data.summary,
                   notes: data.notes,
                   agent_thoughts: data.agent_thoughts || [] // Capture thoughts
               }
           ]);
      }
      else if (data.type === 'text_response') {
          // Add regular text response to conversation
          setConversation(prevConv => [
            ...prevConv,
            { sender: 'assistant', type: 'text', content: data.result, agent_thoughts: data.agent_thoughts || [] } // Capture thoughts
          ]);
      } else if (data.type === 'error') {
          // Display error message and add to conversation history
          setError(data.error);
          setConversation(prevConv => [...prevConv, { sender: 'system', type: 'text', content: `Error: ${data.error}`, agent_thoughts: data.agent_thoughts || [] }]); // Capture thoughts even on error
      }

      setQuery(''); // Clear the input field after search

    } catch (err) {
      console.error('Error fetching search result:', err);
      setError(`Failed to fetch search results: ${err.message}`);
      setConversation(prevConv => [...prevConv, { sender: 'system', type: 'text', content: `Error: ${err.message}` }]);
    } finally {
      setLoading(false); // End loading state
    }
  };

  // Handle input field changes
  const handleInputChange = (event) => {
    setQuery(event.target.value);
    setError(''); // Clear error when user starts typing
  };

  // Trigger search on Enter key press
  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSearch(query);
    }
  };

  // Start speech recognition
  const startListening = () => {
    if (!browserSupportsSpeechRecognition) {
      setError("Browser doesn't support speech recognition.");
      return;
    }
    setIsVoiceInputActive(true);
    resetTranscript(); // Clear previous transcript
    SpeechRecognition.startListening({ continuous: true }); // Start listening continuously
    setError(''); // Clear any previous error

     // Cancel any ongoing speech
     if (synth && synth.speaking) {
      synth.cancel();
      setIsSpeaking(false);
    }
  };

  // Stop speech recognition and trigger search with the transcript
  const stopListening = () => {
    setIsVoiceInputActive(false);
    SpeechRecognition.stopListening();
    if (transcript) {
      handleSearch(transcript); // Use the collected transcript as the query
    } else {
      setError('No speech detected.'); // Show error if no speech was transcribed
    }
  };

  // Toggle between starting and stopping voice input
  const toggleVoiceInput = () => {
    if (isVoiceInputActive) {
      stopListening();
    } else {
      startListening();
    }
  };

  // Display a message if browser doesn't support speech recognition
  useEffect(() => {
    if (!browserSupportsSpeechRecognition) {
      setError("Your browser does not support Speech Recognition. Text input is still available.");
    }
  }, [browserSupportsSpeechRecognition]);

  // Function to read message content aloud
  const readMessageContent = (message) => {
    if (!synth || !message || isMuted) {
      setIsSpeaking(false);
      return;
    }

    let textToSpeak = '';
    if (message.type === 'text') {
      textToSpeak = message.content;
    } else if (message.type === 'code') {
      // For code responses, speak the explanation and a brief summary of each code result
      textToSpeak = message.explanation || "Here are the code generation results.";
      message.code_results.forEach(res => {
        if (res.status === 'success') {
          // Speak the model name and a snippet of the code
          textToSpeak += ` Code from ${res.model}: ${res.content.substring(0, Math.min(res.content.length, 50))}...`;
        } else {
           // Speak the model name and the error/status
          textToSpeak += ` ${res.model} model reported a status: ${res.status.replace('_', ' ')}.`;
        }
      });
    } else if (message.type === 'youtube_summary_notes') {
        // For YouTube summary/notes, speak the summary and mention the notes
        textToSpeak = `Here is the summary of the YouTube video: ${message.summary}. Key notes are also available.`;
    }


    if (!textToSpeak.trim()) {
      setIsSpeaking(false);
      return;
    }

    // Cancel any previous speech and start the new one
    synth.cancel();
    setIsSpeaking(false);

    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }

    // Update speaking state based on utterance events
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    synth.speak(utterance);
  };

  // Automatically read the latest assistant message if not muted
  useEffect(() => {
    const latestMessage = conversation[conversation.length - 1];
    if (latestMessage && latestMessage.sender === 'assistant' && synth && !isMuted) {
        let hasSpeakableContent = false;
        // Check if the message has content that should be spoken
        if (latestMessage.type === 'text' && latestMessage.content.trim()) {
            hasSpeakableContent = true;
        } else if (latestMessage.type === 'code' && (latestMessage.explanation.trim() || latestMessage.code_results.some(res => res.status === 'success' && res.content.trim()))) {
             // For code, speak if there's an explanation or at least one successful code result
            hasSpeakableContent = true;
        } else if (latestMessage.type === 'youtube_summary_notes' && (latestMessage.summary.trim() || latestMessage.notes.trim())) {
            // For YouTube summary/notes, speak if there's a summary or notes
            hasSpeakableContent = true;
        }


        if (hasSpeakableContent) {
            readMessageContent(latestMessage);
        } else {
             // If no speakable content, ensure speech is cancelled
            if (synth.speaking) {
                synth.cancel();
                setIsSpeaking(false);
            }
        }
    } else if (synth && synth.speaking) {
        // If the latest message is not from the assistant or is muted, cancel ongoing speech
        synth.cancel();
        setIsSpeaking(false);
    }
  }, [conversation, synth, isMuted]); // Re-run effect when conversation, synth, or mute state changes


  // Toggle mute state and cancel speech if speaking
  const toggleMute = () => {
    if (synth && synth.speaking) {
      synth.cancel();
      setIsSpeaking(false);
    }
    setIsMuted(!isMuted);
  };

  // Toggle dark mode
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Helper function to extract raw code from markdown fences
  const extractRawCode = (markdownContent) => {
      const codeBlockMatch = markdownContent.match(/^```(?:\w+)?\n([\s\S]*?)\n```$/);
      if (codeBlockMatch && codeBlockMatch[1]) {
          return codeBlockMatch[1].trim();
      }
      // If it doesn't match the expected markdown format, return the original content
      // This might happen for error messages or non-code text within the content field
      return markdownContent.trim();
  };


  // Handle copying code to clipboard
  const handleCopyCode = (codeContent, modelName, messageIndex, codeIndex) => {
    // Extract raw code before copying
    const rawCodeToCopy = extractRawCode(codeContent);

    if (rawCodeToCopy) {
      // Create a unique ID for the copy success message for this specific code block
      const segmentId = `code-copy-${modelName}-${messageIndex}-${codeIndex}`;
      navigator.clipboard.writeText(rawCodeToCopy)
        .then(() => {
          // Show "Copied!" message
          setCopySuccessMessages(prev => ({ ...prev, [segmentId]: 'Copied!' }));
          // Hide the message after 2 seconds
          setTimeout(() => {
            setCopySuccessMessages(prev => {
                const newState = { ...prev };
                delete newState[segmentId];
                return newState;
            });
          }, 2000);
        })
        .catch(err => {
          console.error('Failed to copy code:', err);
           // Show "Failed to copy." message on error
          setCopySuccessMessages(prev => ({ ...prev, [segmentId]: 'Failed to copy.' }));
           // Hide the message after 2 seconds
           setTimeout(() => {
             setCopySuccessMessages(prev => {
                const newState = { ...prev };
                delete newState[segmentId];
                return newState;
            });
          }, 2000);
        });
    }
  };

  // Determine the animation class for the AI status indicator
  const outerAnimationClass = loading
    ? 'animate-spin-custom' // Spin while loading
    : (listening || (isSpeaking && !isMuted) // Pulse while listening or speaking (and not muted)
        ? 'animate-siri-pulse-outer'
        : 'icon-static' // Static when idle
      );

  const innerAnimationClass = (listening || (isSpeaking && !isMuted))
    ? 'animate-siri-pulse-inner' // Inner pulse while listening or speaking
    : '';


  return (
    // Main container with gradient background and flex layout
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-100 dark:from-gray-800 dark:to-gray-900 text-gray-900 dark:text-gray-100 flex items-center justify-center p-4 transition-colors duration-300 ease-in-out">
      {/* Inject custom CSS */}
      <style>{customStyles}</style>
      {/* Chat window container */}
      <div className="bg-white dark:bg-gray-700 p-8 rounded-xl shadow-lg w-full max-w-2xl border border-gray-200 dark:border-gray-600 transition-colors duration-300 ease-in-out flex flex-col h-[90vh]">
        {/* Header */}
        <div className="flex justify-between items-center mb-2">
          <h1 className="text-3xl font-extrabold text-gray-800 dark:text-gray-100">
            MyAssist
          </h1>
          {/* Dark mode toggle button */}
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-full bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-500 transition-colors duration-200 ease-in-out"
            aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>

        {/* AI Status Indicator */}
        <div className="flex justify-center items-center mb-6">
          <div
            className={`
              relative
              w-16 h-16 rounded-full
              ${isDarkMode ? 'bg-blue-400' : 'bg-blue-600'}
              ${outerAnimationClass}
              transition-colors duration-300 ease-in-out
            `}
            aria-label="AI Assistant Status Indicator (Siri-like Animation)"
          >
            {/* Inner pulsing circles for Siri-like effect */}
            {(listening || (isSpeaking && !isMuted)) && (
                 <>
                    <div className={`siri-inner-circle ${innerAnimationClass}`} style={{ animationDelay: '0s' }}></div>
                    <div className={`siri-inner-circle ${innerAnimationClass}`} style={{ animationDelay: '0.3s' }}></div>
                    <div className={`siri-inner-circle ${innerAnimationClass}`} style={{ animationDelay: '0.6s' }}></div>
                 </>
            )}
          </div>
        </div>

        {/* Listening status message */}
        {listening && (
          <p className="text-center text-blue-600 dark:text-blue-400 mb-4 animate-pulse">Listening... Speak now.</p>
        )}

        {/* Error message display */}
        {error && (
          <p className="text-center text-red-600 dark:text-red-400 mb-4">{error}</p>
        )}

        {/* Conversation Display Area */}
        <div className="flex-1 overflow-y-auto mb-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg shadow-inner transition-colors duration-300 ease-in-out flex flex-col space-y-4">
          {/* Map through conversation messages */}
          {conversation.map((message, messageIndex) => (
            <div
              key={messageIndex}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`
                p-3 rounded-lg max-w-[85%]
                ${message.sender === 'user'
                  ? 'bg-blue-500 text-white' // User message style
                  : (message.sender === 'assistant'
                    ? 'bg-gray-300 dark:bg-gray-600 text-gray-900 dark:text-gray-100' // Assistant message style
                    : 'bg-red-200 text-red-800 dark:bg-red-800 dark:text-red-200' // System/error message style
                  )
                }
              `}>
                {/* Render content based on message type */}
                {message.sender === 'assistant' && message.type === 'code' ? (
                  <>
                    {/* Display explanation text if available */}
                    {message.explanation && <p className="whitespace-pre-wrap leading-relaxed mb-2">{message.explanation}</p>}
                    {/* Map through each code result from the different models */}
                    {message.code_results.map((codeResult, codeIndex) => (
                      // Code editor block container
                      <div key={`code-result-${messageIndex}-${codeIndex}`} className="code-editor-block">
                        {/* Code editor header with model name, status, and copy button */}
                        <div className="code-editor-header">
                            <span className="text-sm font-medium text-gray-800 dark:text-gray-100">
                                {codeResult.model.toUpperCase()} Code ({codeResult.status === 'success' ? 'Generated' : codeResult.status.replace('_', ' ')})
                            </span>
                           {/* Display copy success message */}
                           {copySuccessMessages[`code-copy-${codeResult.model}-${messageIndex}-${codeIndex}`] && (
                               <span className="copy-success-message">
                                   {copySuccessMessages[`code-copy-${codeResult.model}-${messageIndex}-${codeIndex}`]}
                               </span>
                           )}
                           {/* Copy button */}
                           <button
                             onClick={() => handleCopyCode(codeResult.content, codeResult.model, messageIndex, codeIndex)}
                             className="code-editor-copy-button"
                             aria-label={`Copy code from ${codeResult.model}`}
                           >
                             <Copy size={14} /> Copy
                           </button>
                        </div>
                        {/* Code content highlighted by SyntaxHighlighter */}
                         {/* Check if content exists before rendering SyntaxHighlighter */}
                        {codeResult.content && (
                           <SyntaxHighlighter
                            language={codeResult.language || 'javascript'} // Use detected language or default to javascript
                            style={dracula} // Apply Dracula theme
                            wrapLines={true} // Wrap long lines
                            customStyle={{
                                padding: '0.8rem',
                                fontSize: '0.85rem',
                                overflowX: 'auto',
                                backgroundColor: isDarkMode ? '#1a202c' : '#f7fafc', // Match block background
                                marginTop: '0',
                                borderRadius: '0 0 0.5rem 0.5rem' // Rounded corners at the bottom
                            }}
                            className="code-syntax-highlighter" // Apply custom class for potential further styling
                          >
                            {/* Pass only the extracted raw code to the highlighter */}
                            {extractRawCode(codeResult.content)}
                          </SyntaxHighlighter>
                        )}
                      </div>
                    ))}
                  </>
                ) : message.sender === 'assistant' && message.type === 'youtube_summary_notes' ? (
                   // Render YouTube summary and notes
                   <div className="space-y-3">
                       <div className="font-semibold text-lg">YouTube Summary:</div>
                       <p className="whitespace-pre-wrap leading-relaxed">{message.summary}</p>
                       {message.notes && message.notes.trim() && (
                           <>
                               <div className="font-semibold text-lg mt-4">Key Notes:</div>
                               {/* Render notes, assuming they might be bullet points or lines */}
                               <p className="whitespace-pre-wrap leading-relaxed">{message.notes}</p>
                           </>
                       )}
                   </div>
                ) : (
                  // Render regular text message (including system/error messages)
                  <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                )}


                {/* Display Agent Thoughts if available and toggled */}
                {message.agent_thoughts && message.agent_thoughts.length > 0 && (
                    <div className="mt-4 text-sm bg-gray-200 dark:bg-gray-700 p-2 rounded-lg">
                        <button
                            onClick={() => setShowThoughts(!showThoughts)}
                            className="flex items-center text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
                            aria-expanded={showThoughts}
                            aria-controls={`thoughts-${messageIndex}`}
                        >
                            {showThoughts ? <ChevronUp size={16} className="mr-1" /> : <ChevronDown size={16} className="mr-1" />}
                            Agent Thoughts
                        </button>
                        {showThoughts && (
                            <div id={`thoughts-${messageIndex}`} className="mt-2 space-y-1 text-gray-600 dark:text-gray-300 whitespace-pre-wrap text-xs">
                                {message.agent_thoughts.map((thought, thoughtIndex) => (
                                    <p key={`${messageIndex}-thought-${thoughtIndex}`} className="italic border-l-2 border-gray-400 pl-2">
                                        {thought}
                                    </p>
                                ))}
                            </div>
                        )}
                    </div>
                )}
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="flex justify-start">
              <div className="p-3 rounded-lg bg-gray-300 dark:bg-gray-600 text-gray-900 dark:text-gray-100">
                Thinking...
              </div>
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="flex items-center space-x-3 mt-auto">
          <input
            type="text"
            value={query}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            placeholder="Type your query or speak..."
            className="flex-grow p-3 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-600 focus:border-transparent transition duration-200 ease-in-out disabled:opacity-60 disabled:cursor-not-allowed placeholder-gray-500 dark:placeholder-gray-400"
            disabled={loading || listening}
            aria-label="Search input"
          />
          {/* Search button */}
          <button
            onClick={() => handleSearch(query)}
            className={`p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition duration-200 ease-in-out shadow-md ${loading || !query.trim() ? 'opacity-60 cursor-not-allowed' : ''}`}
            disabled={loading || !query.trim()}
            aria-label="Search button"
          >
            <Search size={24} />
          </button>

          {/* Voice input button (only if browser supports it) */}
          {browserSupportsSpeechRecognition && (
            <button
              onClick={toggleVoiceInput}
              className={`p-3 rounded-lg transition duration-200 ease-in-out shadow-md ${isVoiceInputActive ? 'bg-red-600 hover:bg-red-700 text-white' : 'bg-gray-300 dark:bg-gray-600 hover:bg-gray-400 dark:hover:bg-gray-500 text-gray-700 dark:text-gray-200'}`}
              disabled={loading}
              aria-label={isVoiceInputActive ? "Stop voice input" : "Start voice input"}
            >
              <Mic size={24} />
            </button>
          )}
        </div>

         {/* Read Aloud and Mute controls */}
         <div className="flex justify-between items-center mt-4">
            {/* Read Aloud button (only if there's a speakable assistant message) */}
            {conversation.length > 0 && conversation[conversation.length - 1].sender === 'assistant' && (
                <button
                  onClick={() => readMessageContent(conversation[conversation.length - 1])}
                  className={`p-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 transition duration-200 ease-in-out flex items-center shadow-sm ${(!synth || isMuted || isSpeaking || (conversation[conversation.length - 1].type === 'code' && !conversation[conversation.length - 1].explanation.trim() && !conversation[conversation.length - 1].code_results.some(res => res.status === 'success' && res.content.trim())) || (conversation[conversation.length - 1].type === 'youtube_summary_notes' && !conversation[conversation.length - 1].summary.trim() && !conversation[conversation.length - 1].notes.trim())) ? 'opacity-60 cursor-not-allowed' : ''}`}
                  disabled={!synth || isMuted || isSpeaking || (conversation[conversation.length - 1].type === 'code' && !conversation[conversation.length - 1].explanation.trim() && !conversation[conversation.length - 1].code_results.some(res => res.status === 'success' && res.content.trim())) || (conversation[conversation.length - 1].type === 'youtube_summary_notes' && !conversation[conversation.length - 1].summary.trim() && !conversation[conversation.length - 1].notes.trim())}
                  aria-label="Read aloud latest result"
                >
                  <Volume2 size={20} className="mr-2"/> Read Aloud
                </button>
            )}

            {/* Mute toggle switch (only if speech synthesis is available) */}
            {synth && (
              <div className="flex items-center space-x-2">
                <VolumeX size={20} className="text-gray-700 dark:text-gray-300" />
                <label className="toggle-switch">
                  <input type="checkbox" checked={isMuted} onChange={toggleMute} aria-label="Toggle mute" />
                  <span className="slider"></span>
                </label>
                <Volume2 size={20} className="text-gray-700 dark:text-gray-300" />
              </div>
            )}
         </div>

      </div>
    </div>
  );
}

export default App;
