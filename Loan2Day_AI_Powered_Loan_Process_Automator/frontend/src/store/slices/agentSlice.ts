/**
 * Agent Slice for Redux Store
 * Manages agent status and state machine progression
 */

import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { sendMessage, sendVoiceMessage } from './chatSlice';

type AgentStatus = 'GREETING' | 'KYC' | 'NEGOTIATION' | 'SANCTION' | 'PLAN_B' | 'PROCESSING' | 'ERROR' | 'IDLE';
type AgentType = 'master' | 'sales' | 'verification' | 'underwriting' | 'system';

interface AgentState {
  currentStep: AgentStatus;
  activeAgent: AgentType;
  isProcessing: boolean;
  agentState: {
    session_id: string;
    user_id: string;
    current_step: string;
    loan_details: Record<string, any>;
    kyc_status: string;
    fraud_score: number;
    sentiment_history: string[];
    trust_score?: number;
  } | null;
  lastUpdated: Date | null;
}

const initialState: AgentState = {
  currentStep: 'IDLE',
  activeAgent: 'system',
  isProcessing: false,
  agentState: null,
  lastUpdated: null,
};

const agentSlice = createSlice({
  name: 'agent',
  initialState,
  reducers: {
    setCurrentStep: (state, action: PayloadAction<AgentStatus>) => {
      state.currentStep = action.payload;
      state.lastUpdated = new Date();
    },
    setActiveAgent: (state, action: PayloadAction<AgentType>) => {
      state.activeAgent = action.payload;
      state.lastUpdated = new Date();
    },
    setProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
    },
    updateAgentState: (state, action: PayloadAction<AgentState['agentState']>) => {
      state.agentState = action.payload;
      state.lastUpdated = new Date();
      
      // Update current step based on agent state
      if (action.payload?.current_step) {
        state.currentStep = action.payload.current_step as AgentStatus;
      }
    },
    resetAgentState: (state) => {
      state.currentStep = 'IDLE';
      state.activeAgent = 'system';
      state.isProcessing = false;
      state.agentState = null;
      state.lastUpdated = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Handle chat message responses
      .addCase(sendMessage.pending, (state) => {
        state.isProcessing = true;
      })
      .addCase(sendMessage.fulfilled, (state, action) => {
        state.isProcessing = false;
        state.activeAgent = action.payload.active_agent as AgentType;
        state.currentStep = action.payload.current_step as AgentStatus;
        state.agentState = action.payload.agent_state;
        state.lastUpdated = new Date();
      })
      .addCase(sendMessage.rejected, (state) => {
        state.isProcessing = false;
        state.currentStep = 'ERROR';
        state.lastUpdated = new Date();
      })
      
      // Handle voice message responses
      .addCase(sendVoiceMessage.pending, (state) => {
        state.isProcessing = true;
      })
      .addCase(sendVoiceMessage.fulfilled, (state, action) => {
        state.isProcessing = false;
        state.activeAgent = action.payload.active_agent as AgentType;
        state.currentStep = action.payload.current_step as AgentStatus;
        state.agentState = action.payload.agent_state;
        state.lastUpdated = new Date();
      })
      .addCase(sendVoiceMessage.rejected, (state) => {
        state.isProcessing = false;
        state.currentStep = 'ERROR';
        state.lastUpdated = new Date();
      });
  },
});

export const {
  setCurrentStep,
  setActiveAgent,
  setProcessing,
  updateAgentState,
  resetAgentState,
} = agentSlice.actions;

export default agentSlice.reducer;