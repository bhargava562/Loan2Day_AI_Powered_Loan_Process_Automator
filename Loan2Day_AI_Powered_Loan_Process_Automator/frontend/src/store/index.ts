/**
 * Redux Store Configuration for Loan2Day Frontend
 * Manages application state including chat messages, agent status, and user session
 */

import { configureStore } from '@reduxjs/toolkit';
import chatReducer from './slices/chatSlice';
import agentReducer from './slices/agentSlice';
import userReducer from './slices/userSlice';

export const store = configureStore({
  reducer: {
    chat: chatReducer,
    agent: agentReducer,
    user: userReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types for serialization checks
        ignoredActions: ['chat/addMessage'],
        // Ignore these field paths in all actions
        ignoredActionsPaths: ['payload.timestamp'],
        // Ignore these paths in the state
        ignoredPaths: ['chat.messages.timestamp'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;