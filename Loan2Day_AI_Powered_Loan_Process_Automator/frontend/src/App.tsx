import React, { useEffect, useState } from 'react';
import { Provider } from 'react-redux';
import { store } from './store';
import { ChatWindow, AudioRecorder, StatusBadge } from './components';
import { useAppDispatch, useAppSelector } from './hooks/redux';
import { sendMessage, sendVoiceMessage, addMessage, setSessionId } from './store/slices/chatSlice';
import { generateSessionId, apiService } from './services/api';
import './index.css';

/**
 * Main App Component for Loan2Day Frontend
 * Integrates chat interface, voice recording, and agent status display
 */
const AppContent: React.FC = () => {
  const dispatch = useAppDispatch();
  const { messages, isLoading, sessionId } = useAppSelector((state) => state.chat);
  const { currentStep, activeAgent, isProcessing } = useAppSelector((state) => state.agent);
  const { preferences } = useAppSelector((state) => state.user);

  const [isConnected, setIsConnected] = useState(false);

  // Initialize session on component mount
  useEffect(() => {
    const initializeSession = async () => {
      try {
        // Check backend health
        await apiService.healthCheck();
        setIsConnected(true);

        // Generate new session ID if not exists
        if (!sessionId) {
          const newSessionId = generateSessionId();
          dispatch(setSessionId(newSessionId));
        }

        // Add welcome message
        dispatch(addMessage({
          text: "Welcome to Loan2Day! I'm here to help you with your loan application. How can I assist you today?",
          sender: 'agent',
          timestamp: new Date(),
          agentType: 'master',
        }));
      } catch (error) {
        console.error('Failed to connect to backend:', error);
        setIsConnected(false);
        
        // Add error message
        dispatch(addMessage({
          text: "I'm having trouble connecting to our services. Please check your internet connection and try again.",
          sender: 'agent',
          timestamp: new Date(),
          agentType: 'system',
        }));
      }
    };

    initializeSession();
  }, [dispatch, sessionId]);

  const handleSendMessage = async (message: string) => {
    if (!sessionId || !isConnected) return;

    // Add user message to chat
    dispatch(addMessage({
      text: message,
      sender: 'user',
      timestamp: new Date(),
    }));

    // Send message to backend
    try {
      await dispatch(sendMessage({
        message,
        sessionId,
        userId: 'user_demo', // TODO: Replace with actual user ID
      })).unwrap();
    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Add error message
      dispatch(addMessage({
        text: "I'm sorry, I couldn't process your message right now. Please try again.",
        sender: 'agent',
        timestamp: new Date(),
        agentType: 'system',
      }));
    }
  };

  const handleVoiceRecorded = async (audioBlob: Blob) => {
    if (!sessionId || !preferences.voiceEnabled || !isConnected) return;

    try {
      // Add placeholder message for voice input
      dispatch(addMessage({
        text: "🎤 Voice message received, processing...",
        sender: 'user',
        timestamp: new Date(),
      }));

      await dispatch(sendVoiceMessage({
        audioBlob,
        sessionId,
      })).unwrap();
    } catch (error) {
      console.error('Failed to process voice message:', error);
      
      // Add error message
      dispatch(addMessage({
        text: "I couldn't process your voice message. Please try typing your message instead.",
        sender: 'agent',
        timestamp: new Date(),
        agentType: 'system',
      }));
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-4 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Loan2Day
          </h1>
          <p className="text-lg text-gray-600">
            AI-Powered Loan Processing Platform
          </p>
          <div className="flex items-center justify-center mt-4 space-x-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <span className="text-sm text-gray-500">
              {isConnected ? 'Connected to Loan2Day Services' : 'Connection Issues'}
            </span>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Chat Interface - Takes up 2 columns on large screens */}
          <div className="lg:col-span-2">
            <ChatWindow
              messages={messages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading || isProcessing}
              currentAgentStatus={`${activeAgent} - ${currentStep}`}
            />
          </div>

          {/* Sidebar with Status and Voice */}
          <div className="space-y-6">
            {/* Agent Status */}
            <StatusBadge
              currentStep={currentStep}
              activeAgent={activeAgent}
              isProcessing={isProcessing || isLoading}
            />

            {/* Voice Recorder */}
            <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 text-center">
                Voice Input
              </h3>
              <div className="flex justify-center">
                <AudioRecorder
                  onAudioRecorded={handleVoiceRecorded}
                  isDisabled={!preferences.voiceEnabled || !isConnected}
                />
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Quick Actions
              </h3>
              <div className="space-y-3">
                <button
                  onClick={() => handleSendMessage("I want to apply for a loan")}
                  disabled={!isConnected}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-200"
                >
                  Start Loan Application
                </button>
                <button
                  onClick={() => handleSendMessage("What documents do I need?")}
                  disabled={!isConnected}
                  className="w-full px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed transition-colors duration-200"
                >
                  Required Documents
                </button>
                <button
                  onClick={() => handleSendMessage("What are your interest rates?")}
                  disabled={!isConnected}
                  className="w-full px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed transition-colors duration-200"
                >
                  Interest Rates
                </button>
              </div>
            </div>

            {/* Language Preferences */}
            <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Language
              </h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Current:</span>
                  <span className="text-sm font-medium text-blue-600 capitalize">
                    {preferences.language === 'tanglish' ? 'Tamil + English' : preferences.language}
                  </span>
                </div>
                <p className="text-xs text-gray-500">
                  Voice interface supports mixed Tamil-English (Tanglish) input
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-sm text-gray-500">
          <p>
            Powered by Agentic AI • Master-Worker Architecture • Zero-Hallucination Mathematics
          </p>
          <p className="mt-1">
            Secure • Fast • Empathetic Loan Processing
          </p>
        </div>
      </div>
    </div>
  );
};

/**
 * Main App Component with Redux Provider
 */
const App: React.FC = () => {
  return (
    <Provider store={store}>
      <AppContent />
    </Provider>
  );
};

export default App;
