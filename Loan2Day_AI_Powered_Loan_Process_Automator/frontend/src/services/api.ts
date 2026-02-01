/**
 * API Service for Loan2Day Frontend
 * Handles communication with the FastAPI backend
 * Follows the Master-Worker Agent architecture
 */

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Types for API communication
export interface ChatMessage {
  session_id: string;
  user_input: string;
  user_id?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  current_step: string;
  active_agent: string;
  agent_state: {
    session_id: string;
    user_id: string;
    current_step: string;
    loan_details: Record<string, any>;
    kyc_status: string;
    fraud_score: number;
    sentiment_history: string[];
    trust_score?: number;
  };
}

export interface KYCUploadResponse {
  message: string;
  document_id: string;
  sgs_score: number;
  verification_status: string;
}

export interface PlanBResponse {
  alternative_offers: Array<{
    loan_amount: string;
    interest_rate: string;
    tenure_months: number;
    emi_amount: string;
    eligibility_reason: string;
  }>;
  message: string;
}

export interface DocumentGenerationResponse {
  message: string;
  download_token: string;
  expires_at: string;
}

export interface APIError {
  error_code: string;
  error_message: string;
  error_details: Record<string, any>;
  timestamp: string;
  session_id?: string;
  trace_id: string;
}

/**
 * API Service class for handling all backend communication
 */
class APIService {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Generic fetch wrapper with error handling
   */
  private async fetchWithErrorHandling<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData: APIError = await response.json();
        throw new Error(`API Error: ${errorData.error_message} (${errorData.error_code})`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(`Network error: ${error}`);
    }
  }

  /**
   * Send chat message to the Master Agent
   * POST /v1/chat/message
   */
  async sendChatMessage(message: ChatMessage): Promise<ChatResponse> {
    return this.fetchWithErrorHandling<ChatResponse>('/v1/chat/message', {
      method: 'POST',
      body: JSON.stringify(message),
    });
  }

  /**
   * Upload KYC document for verification
   * POST /v1/upload/kyc
   */
  async uploadKYCDocument(
    sessionId: string,
    file: File,
    documentType: string
  ): Promise<KYCUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);
    formData.append('document_type', documentType);

    const response = await fetch(`${this.baseURL}/v1/upload/kyc`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData: APIError = await response.json();
      throw new Error(`Upload Error: ${errorData.error_message}`);
    }

    return await response.json();
  }

  /**
   * Get Plan B alternative loan offers
   * GET /v1/loan/plan-b
   */
  async getPlanBOffers(sessionId: string): Promise<PlanBResponse> {
    return this.fetchWithErrorHandling<PlanBResponse>(
      `/v1/loan/plan-b?session_id=${sessionId}`,
      { method: 'GET' }
    );
  }

  /**
   * Generate sanction letter document
   * POST /v1/documents/generate-sanction
   */
  async generateSanctionLetter(sessionId: string): Promise<DocumentGenerationResponse> {
    return this.fetchWithErrorHandling<DocumentGenerationResponse>(
      '/v1/documents/generate-sanction',
      {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      }
    );
  }

  /**
   * Download generated document
   * GET /v1/documents/download/{token}
   */
  async downloadDocument(token: string): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/v1/documents/download/${token}`);
    
    if (!response.ok) {
      throw new Error('Failed to download document');
    }

    return await response.blob();
  }

  /**
   * Health check endpoint
   * GET /health
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.fetchWithErrorHandling<{ status: string; timestamp: string }>('/health', {
      method: 'GET',
    });
  }

  /**
   * Voice webhook handler (for future Twilio integration)
   * POST /webhook/voice
   */
  async processVoiceInput(audioBlob: Blob, sessionId: string): Promise<ChatResponse> {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('session_id', sessionId);

    const response = await fetch(`${this.baseURL}/webhook/voice`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData: APIError = await response.json();
      throw new Error(`Voice Processing Error: ${errorData.error_message}`);
    }

    return await response.json();
  }
}

// Export singleton instance
export const apiService = new APIService();

// Export utility functions
export const generateSessionId = (): string => {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

export const formatCurrency = (amountInCents: string | number): string => {
  const amount = typeof amountInCents === 'string' ? parseFloat(amountInCents) : amountInCents;
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 2,
  }).format(amount / 100);
};

export const formatPercentage = (rate: string | number): string => {
  const rateNum = typeof rate === 'string' ? parseFloat(rate) : rate;
  return `${rateNum.toFixed(2)}%`;
};

export default apiService;