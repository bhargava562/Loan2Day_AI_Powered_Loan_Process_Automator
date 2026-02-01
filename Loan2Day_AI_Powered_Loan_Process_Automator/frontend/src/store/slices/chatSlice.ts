/**
 * Chat Slice for Redux Store
 * Manages chat messages and conversation state
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { apiService } from '../../services/api';
import type { ChatMessage } from '../../services/api';

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'agent';
  timestamp: Date;
  agentType?: 'master' | 'sales' | 'verification' | 'underwriting' | 'system';
}

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  sessionId: string | null;
}

const initialState: ChatState = {
  messages: [],
  isLoading: false,
  error: null,
  sessionId: null,
};

// Async thunk for sending chat messages
export const sendMessage = createAsyncThunk(
  'chat/sendMessage',
  async (payload: { message: string; sessionId: string; userId?: string }) => {
    const chatMessage: ChatMessage = {
      session_id: payload.sessionId,
      user_input: payload.message,
      user_id: payload.userId,
    };

    const response = await apiService.sendChatMessage(chatMessage);
    return response;
  }
);

// Async thunk for processing voice input
export const sendVoiceMessage = createAsyncThunk(
  'chat/sendVoiceMessage',
  async (payload: { audioBlob: Blob; sessionId: string }) => {
    const response = await apiService.processVoiceInput(payload.audioBlob, payload.sessionId);
    return response;
  }
);

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    addMessage: (state, action: PayloadAction<Omit<Message, 'id'>>) => {
      const message: Message = {
        ...action.payload,
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      };
      state.messages.push(message);
    },
    clearMessages: (state) => {
      state.messages = [];
    },
    setSessionId: (state, action: PayloadAction<string>) => {
      state.sessionId = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Send Message Cases
      .addCase(sendMessage.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(sendMessage.fulfilled, (state, action) => {
        state.isLoading = false;
        
        // Add agent response message
        const agentMessage: Message = {
          id: `msg_${Date.now()}_agent`,
          text: action.payload.response,
          sender: 'agent',
          timestamp: new Date(),
          agentType: action.payload.active_agent as any,
        };
        state.messages.push(agentMessage);
        
        // Update session ID if provided
        if (action.payload.session_id) {
          state.sessionId = action.payload.session_id;
        }
      })
      .addCase(sendMessage.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to send message';
      })
      
      // Send Voice Message Cases
      .addCase(sendVoiceMessage.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(sendVoiceMessage.fulfilled, (state, action) => {
        state.isLoading = false;
        
        // Add agent response message
        const agentMessage: Message = {
          id: `msg_${Date.now()}_voice_agent`,
          text: action.payload.response,
          sender: 'agent',
          timestamp: new Date(),
          agentType: action.payload.active_agent as any,
        };
        state.messages.push(agentMessage);
        
        // Update session ID if provided
        if (action.payload.session_id) {
          state.sessionId = action.payload.session_id;
        }
      })
      .addCase(sendVoiceMessage.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to process voice message';
      });
  },
});

export const { addMessage, clearMessages, setSessionId, clearError } = chatSlice.actions;
export default chatSlice.reducer;