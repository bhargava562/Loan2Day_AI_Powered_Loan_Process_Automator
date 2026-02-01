import React, { useState } from 'react';

interface AudioRecorderProps {
  onAudioRecorded?: (audioBlob: Blob) => void;
  isRecording?: boolean;
  isDisabled?: boolean;
  className?: string;
}

/**
 * AudioRecorder component for voice input in Loan2Day
 * Currently a visual placeholder for future Twilio integration
 * Supports Tanglish (Tamil-English) voice processing
 */
export const AudioRecorder: React.FC<AudioRecorderProps> = ({
  onAudioRecorded,
  isRecording = false,
  isDisabled = false,
  className = ''
}) => {
  const [isPressed, setIsPressed] = useState(false);

  const handleMouseDown = () => {
    if (!isDisabled) {
      setIsPressed(true);
      // TODO: Start recording when Twilio integration is implemented
      console.log('Voice recording started (placeholder)');
    }
  };

  const handleMouseUp = () => {
    if (!isDisabled && isPressed) {
      setIsPressed(false);
      // TODO: Stop recording and process audio when Twilio integration is implemented
      console.log('Voice recording stopped (placeholder)');
      
      // Placeholder: Create a mock audio blob for testing
      if (onAudioRecorded) {
        const mockBlob = new Blob(['mock audio data'], { type: 'audio/wav' });
        onAudioRecorded(mockBlob);
      }
    }
  };

  const handleMouseLeave = () => {
    if (isPressed) {
      handleMouseUp();
    }
  };

  return (
    <div className={`flex flex-col items-center space-y-2 ${className}`}>
      <button
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onTouchStart={handleMouseDown}
        onTouchEnd={handleMouseUp}
        disabled={isDisabled}
        className={`
          relative w-16 h-16 rounded-full transition-all duration-200 focus:outline-none focus:ring-4 focus:ring-blue-300
          ${isPressed || isRecording
            ? 'bg-red-500 hover:bg-red-600 scale-110 shadow-lg'
            : 'bg-blue-600 hover:bg-blue-700 shadow-md'
          }
          ${isDisabled
            ? 'bg-gray-400 cursor-not-allowed opacity-50'
            : 'cursor-pointer'
          }
        `}
        aria-label={isPressed || isRecording ? 'Recording... Release to stop' : 'Hold to record voice message'}
        title={isPressed || isRecording ? 'Recording... Release to stop' : 'Hold to record voice message'}
      >
        {/* Microphone Icon */}
        <div className="flex items-center justify-center w-full h-full">
          {isPressed || isRecording ? (
            <svg
              className="w-6 h-6 text-white animate-pulse"
              fill="currentColor"
              viewBox="0 0 20 20"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z"
                clipRule="evenodd"
              />
            </svg>
          ) : (
            <svg
              className="w-6 h-6 text-white"
              fill="currentColor"
              viewBox="0 0 20 20"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                fillRule="evenodd"
                d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z"
                clipRule="evenodd"
              />
            </svg>
          )}
        </div>

        {/* Recording Animation Ring */}
        {(isPressed || isRecording) && (
          <div className="absolute inset-0 rounded-full border-4 border-red-300 animate-ping"></div>
        )}
      </button>

      {/* Status Text */}
      <div className="text-center">
        <p className={`text-sm font-medium ${
          isPressed || isRecording ? 'text-red-600' : 'text-gray-600'
        }`}>
          {isPressed || isRecording ? 'Recording...' : 'Hold to Record'}
        </p>
        <p className="text-xs text-gray-500 mt-1">
          {isDisabled ? 'Voice input disabled' : 'Supports Tamil & English'}
        </p>
      </div>

      {/* Feature Notice */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-md p-2 max-w-xs">
        <div className="flex items-center space-x-2">
          <svg className="w-4 h-4 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <p className="text-xs text-yellow-800">
            Voice feature coming soon with Twilio integration
          </p>
        </div>
      </div>
    </div>
  );
};

export default AudioRecorder;