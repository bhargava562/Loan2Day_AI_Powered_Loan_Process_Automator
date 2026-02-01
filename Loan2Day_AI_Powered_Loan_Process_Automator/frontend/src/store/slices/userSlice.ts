/**
 * User Slice for Redux Store
 * Manages user profile and session information
 */

import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

interface UserProfile {
  user_id: string;
  name: string;
  phone: string;
  email: string;
  income_in_cents: string;
  employment_type: string;
  credit_score?: number;
}

interface UserState {
  profile: UserProfile | null;
  sessionId: string | null;
  isAuthenticated: boolean;
  preferences: {
    language: 'en' | 'ta' | 'tanglish';
    voiceEnabled: boolean;
    notifications: boolean;
  };
}

const initialState: UserState = {
  profile: null,
  sessionId: null,
  isAuthenticated: false,
  preferences: {
    language: 'tanglish',
    voiceEnabled: true,
    notifications: true,
  },
};

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    setUserProfile: (state, action: PayloadAction<UserProfile>) => {
      state.profile = action.payload;
      state.isAuthenticated = true;
    },
    setSessionId: (state, action: PayloadAction<string>) => {
      state.sessionId = action.payload;
    },
    updatePreferences: (state, action: PayloadAction<Partial<UserState['preferences']>>) => {
      state.preferences = { ...state.preferences, ...action.payload };
    },
    clearUserData: (state) => {
      state.profile = null;
      state.sessionId = null;
      state.isAuthenticated = false;
    },
    setLanguage: (state, action: PayloadAction<'en' | 'ta' | 'tanglish'>) => {
      state.preferences.language = action.payload;
    },
    toggleVoice: (state) => {
      state.preferences.voiceEnabled = !state.preferences.voiceEnabled;
    },
    toggleNotifications: (state) => {
      state.preferences.notifications = !state.preferences.notifications;
    },
  },
});

export const {
  setUserProfile,
  setSessionId,
  updatePreferences,
  clearUserData,
  setLanguage,
  toggleVoice,
  toggleNotifications,
} = userSlice.actions;

export default userSlice.reducer;