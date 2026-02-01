import React from 'react';

type AgentStatus = 'GREETING' | 'KYC' | 'NEGOTIATION' | 'SANCTION' | 'PLAN_B' | 'PROCESSING' | 'ERROR' | 'IDLE';
type AgentType = 'master' | 'sales' | 'verification' | 'underwriting' | 'system';

interface StatusBadgeProps {
  currentStep: AgentStatus;
  activeAgent?: AgentType;
  isProcessing?: boolean;
  className?: string;
}

/**
 * StatusBadge component showing current agent status and processing state
 * Provides visual feedback for the Master-Worker Agent architecture
 */
export const StatusBadge: React.FC<StatusBadgeProps> = ({
  currentStep,
  activeAgent = 'system',
  isProcessing = false,
  className = ''
}) => {
  const getStepInfo = (step: AgentStatus) => {
    switch (step) {
      case 'GREETING':
        return {
          label: 'Welcome',
          description: 'Getting started with your loan application',
          color: 'bg-blue-100 text-blue-800 border-blue-200',
          icon: '👋'
        };
      case 'KYC':
        return {
          label: 'Identity Verification',
          description: 'Verifying your documents and identity',
          color: 'bg-yellow-100 text-yellow-800 border-yellow-200',
          icon: '🔍'
        };
      case 'NEGOTIATION':
        return {
          label: 'Loan Processing',
          description: 'Calculating terms and assessing your application',
          color: 'bg-purple-100 text-purple-800 border-purple-200',
          icon: '💼'
        };
      case 'SANCTION':
        return {
          label: 'Approved!',
          description: 'Your loan has been sanctioned',
          color: 'bg-green-100 text-green-800 border-green-200',
          icon: '✅'
        };
      case 'PLAN_B':
        return {
          label: 'Alternative Options',
          description: 'Exploring alternative loan products for you',
          color: 'bg-orange-100 text-orange-800 border-orange-200',
          icon: '🔄'
        };
      case 'PROCESSING':
        return {
          label: 'Processing',
          description: 'Please wait while we process your request',
          color: 'bg-gray-100 text-gray-800 border-gray-200',
          icon: '⏳'
        };
      case 'ERROR':
        return {
          label: 'Issue Detected',
          description: 'We encountered an issue, please try again',
          color: 'bg-red-100 text-red-800 border-red-200',
          icon: '⚠️'
        };
      case 'IDLE':
      default:
        return {
          label: 'Ready',
          description: 'System ready to assist you',
          color: 'bg-gray-100 text-gray-800 border-gray-200',
          icon: '💬'
        };
    }
  };

  const getAgentInfo = (agent: AgentType) => {
    switch (agent) {
      case 'sales':
        return {
          name: 'Sales Agent',
          description: 'Helping with loan options and terms',
          color: 'text-green-600',
          icon: '💰'
        };
      case 'verification':
        return {
          name: 'Verification Agent',
          description: 'Processing your documents and identity',
          color: 'text-blue-600',
          icon: '🔐'
        };
      case 'underwriting':
        return {
          name: 'Underwriting Agent',
          description: 'Calculating loan terms and risk assessment',
          color: 'text-purple-600',
          icon: '📊'
        };
      case 'master':
        return {
          name: 'Master Agent',
          description: 'Coordinating your loan application',
          color: 'text-gray-600',
          icon: '🎯'
        };
      case 'system':
      default:
        return {
          name: 'System',
          description: 'Ready to assist',
          color: 'text-gray-600',
          icon: '🤖'
        };
    }
  };

  const stepInfo = getStepInfo(currentStep);
  const agentInfo = getAgentInfo(activeAgent);

  return (
    <div className={`bg-white rounded-lg shadow-md border border-gray-200 p-4 ${className}`}>
      {/* Current Step Status */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <span className="text-2xl">{stepInfo.icon}</span>
          <div>
            <h3 className="font-semibold text-gray-900">{stepInfo.label}</h3>
            <p className="text-sm text-gray-600">{stepInfo.description}</p>
          </div>
        </div>
        <div className={`px-3 py-1 rounded-full border text-xs font-medium ${stepInfo.color}`}>
          {stepInfo.label}
        </div>
      </div>

      {/* Active Agent Information */}
      <div className="border-t border-gray-200 pt-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-lg">{agentInfo.icon}</span>
            <div>
              <p className={`text-sm font-medium ${agentInfo.color}`}>
                {agentInfo.name}
              </p>
              <p className="text-xs text-gray-500">{agentInfo.description}</p>
            </div>
          </div>
          
          {/* Processing Indicator */}
          <div className="flex items-center space-x-2">
            {isProcessing ? (
              <>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                <span className="text-xs text-blue-600 font-medium">Active</span>
              </>
            ) : (
              <>
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-xs text-green-600 font-medium">Ready</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Progress Indicator */}
      <div className="mt-4">
        <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
          <span>Application Progress</span>
          <span>{getProgressPercentage(currentStep)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${getProgressPercentage(currentStep)}%` }}
          ></div>
        </div>
      </div>

      {/* Step Indicators */}
      <div className="flex items-center justify-between mt-3 text-xs">
        {['GREETING', 'KYC', 'NEGOTIATION', 'SANCTION'].map((step, index) => (
          <div
            key={step}
            className={`flex items-center space-x-1 ${
              getStepOrder(currentStep) >= index
                ? 'text-blue-600'
                : 'text-gray-400'
            }`}
          >
            <div
              className={`w-2 h-2 rounded-full ${
                getStepOrder(currentStep) >= index
                  ? 'bg-blue-600'
                  : 'bg-gray-300'
              }`}
            ></div>
            <span className="hidden sm:inline">
              {step.charAt(0) + step.slice(1).toLowerCase()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Helper function to get progress percentage based on current step
 */
const getProgressPercentage = (step: AgentStatus): number => {
  switch (step) {
    case 'GREETING':
      return 25;
    case 'KYC':
      return 50;
    case 'NEGOTIATION':
      return 75;
    case 'SANCTION':
      return 100;
    case 'PLAN_B':
      return 60; // Alternative path
    case 'PROCESSING':
      return 40;
    case 'ERROR':
      return 10;
    case 'IDLE':
    default:
      return 0;
  }
};

/**
 * Helper function to get step order for progress indication
 */
const getStepOrder = (step: AgentStatus): number => {
  switch (step) {
    case 'GREETING':
      return 0;
    case 'KYC':
      return 1;
    case 'NEGOTIATION':
    case 'PLAN_B':
      return 2;
    case 'SANCTION':
      return 3;
    default:
      return 0;
  }
};

export default StatusBadge;