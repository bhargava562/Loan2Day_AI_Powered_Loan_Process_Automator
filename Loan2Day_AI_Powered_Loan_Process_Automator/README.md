# Loan2Day - Agentic AI Fintech Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.62-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Loan2Day is an advanced Agentic AI Fintech platform that replaces traditional linear chatbot scripts with a dynamic state machine architecture. The system uses a centralized **Master Agent** to orchestrate specialized **Worker Agents**, providing intelligent loan processing with empathy, security, and mathematical precision.

## 🏗️ Architecture Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        React[React Frontend<br/>TypeScript + Tailwind]
        Voice[Voice Interface<br/>Twilio Integration]
    end
    
    subgraph "API Gateway"
        FastAPI[FastAPI Gateway<br/>< 2s latency]
    end
    
    subgraph "Master-Worker Agent Layer"
        Master[Master Agent<br/>LangGraph Orchestrator]
        Sales[Sales Agent<br/>Sentiment + Plan B]
        Verify[Verification Agent<br/>KYC + SGS + SBEF]
        Under[Underwriting Agent<br/>LQM + EMI Calc]
    end
    
    subgraph "Data Layer"
        Redis[(Redis Cache<br/>AgentState)]
        Postgres[(PostgreSQL<br/>Persistent Data)]
        Kafka[Kafka Streams<br/>Async Events]
    end
    
    subgraph "Security & Math Layer"
        SGS[SGS Module<br/>Deepfake Detection]
        LQM[LQM Module<br/>Decimal Math]
        SBEF[SBEF Algorithm<br/>Conflict Resolution]
    end
    
    React --> FastAPI
    Voice --> FastAPI
    FastAPI --> Master
    Master --> Sales
    Master --> Verify
    Master --> Under
    
    Sales --> Redis
    Verify --> Redis
    Under --> Redis
    
    Verify --> SGS
    Under --> LQM
    Verify --> SBEF
    
    Master --> Postgres
    Master --> Kafka
```

## 🚀 Key Features

### 🧠 Master-Worker Agent Pattern
- **Master Agent**: LangGraph-powered orchestrator managing session state and routing
- **Sales Agent**: Empathetic communication with sentiment analysis and Plan B logic
- **Verification Agent**: KYC processing with SGS security scanning and SBEF conflict resolution
- **Underwriting Agent**: Risk assessment and EMI calculations using LQM mathematics

### 🔒 Zero-Hallucination Security
- **SGS (Spectral-Graph Sentinel)**: Advanced deepfake detection and file security scanning
- **LQM (Logic Quantization Module)**: Decimal-only mathematics preventing float precision errors
- **SBEF Algorithm**: Semantic-Bayesian Evidence Fusion for intelligent data conflict resolution

### 🌍 Multilingual Support
- **Tanglish Processing**: Tamil + English mixed language support
- **Voice Interface**: Twilio integration for telephony on non-smartphone devices
- **Sentiment Analysis**: Real-time emotional state detection for empathetic responses

### 📊 Plan B Logic
- **Rejection Recovery**: Alternative loan products when primary applications are rejected
- **Conversion Optimization**: Maximizes approval rates through intelligent alternatives
- **Empathetic Messaging**: Sentiment-driven communication adaptation

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Backend** | Python | 3.11 LTS | Core application logic |
| **API Framework** | FastAPI | 0.104.1 | High-performance async API |
| **Agent Orchestration** | LangGraph | 0.0.62 | State machine management |
| **Database** | PostgreSQL | Latest | Persistent data storage |
| **Caching** | Redis | Latest | Session state management |
| **Message Queue** | Apache Kafka | Latest | Async communication |
| **Frontend** | React + TypeScript | Latest | User interface |
| **Styling** | Tailwind CSS | Latest | UI styling |
| **Voice Processing** | Twilio | Latest | Telephony integration |

## 📋 Prerequisites

- **Python 3.11 LTS** (Required for compatibility)
- **Node.js 18+** (For frontend development)
- **PostgreSQL 14+** (Database)
- **Redis 6+** (Session caching)
- **Apache Kafka** (Message streaming)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/loan2day.git
cd loan2day

# Create Python 3.11 virtual environment
python3.11 -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Upgrade pip inside venv
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

**Required Environment Variables:**
```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://loan2day:password@localhost/loan2day
REDIS_URL=redis://localhost:6379/0
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Twilio Configuration (Optional for voice features)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token

# Security Configuration
SGS_SECURITY_THRESHOLD=0.85
MAX_FILE_SIZE_MB=10

# Performance Configuration
API_RESPONSE_TIMEOUT_SECONDS=2.0
REDIS_SESSION_TTL_SECONDS=3600
```

### 3. Database Setup

```bash
# Start PostgreSQL and Redis (using Docker)
docker-compose up -d postgres redis kafka

# Run database migrations (when implemented)
# alembic upgrade head
```

### 4. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Install additional packages
npm install react-router-dom @reduxjs/toolkit react-redux

# Start development server
npm run dev
```

### 5. Start the Application

```bash
# In the root directory, start the FastAPI server
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# The API will be available at:
# - Main API: http://localhost:8000
# - Documentation: http://localhost:8000/docs
# - Health Check: http://localhost:8000/health
```

## 📚 API Documentation

### Core Endpoints

#### Chat API
```http
POST /v1/chat/message
Content-Type: application/json

{
  "session_id": "sess_20240301_123456_user123",
  "user_id": "user123",
  "message": "I need a personal loan of ₹5 lakhs",
  "message_type": "text",
  "language": "en"
}
```

**Response:**
```json
{
  "session_id": "sess_20240301_123456_user123",
  "agent_response": "I'd be happy to help you with a ₹5 lakh personal loan. Let me gather some details to find the best options for you.",
  "current_step": "GREETING",
  "next_actions": [
    "Provide your monthly income details",
    "Share your employment information",
    "Upload required KYC documents"
  ],
  "requires_input": true,
  "user_intent": "LOAN_INQUIRY",
  "processing_time_ms": 1250
}
```

#### KYC Upload API
```http
POST /v1/upload/kyc
Content-Type: multipart/form-data

session_id: sess_20240301_123456_user123
user_id: user123
document_type: PAN_CARD
file: [binary file data]
```

**Response:**
```json
{
  "upload_id": "upload_20240301_123456_doc123",
  "session_id": "sess_20240301_123456_user123",
  "document_type": "PAN_CARD",
  "sgs_security_score": {
    "overall_score": 0.92,
    "threat_level": "SAFE",
    "is_safe": true
  },
  "verification_status": "VERIFIED",
  "fraud_score": 0.15,
  "processing_time_ms": 3250
}
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for complete interactive API documentation with:
- Request/response schemas
- Example payloads
- Authentication details
- Error codes and handling

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run property-based tests
pytest tests/property/

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/unit/test_lqm.py -v

# Run Plan B logic tests
pytest tests/unit/test_plan_b_logic.py -v
```

### Test Categories

1. **Unit Tests** (`tests/unit/`):
   - Specific functionality validation
   - Mock external dependencies
   - Edge case handling
   - Plan B logic validation

2. **Property Tests** (`tests/property/`):
   - Universal correctness properties
   - Hypothesis-based randomized testing
   - Mathematical consistency validation
   - 100+ iterations per property

3. **Integration Tests** (`tests/integration/`):
   - End-to-end workflow testing
   - API endpoint validation
   - Database integration

## 🏛️ Architecture Deep Dive

### Master-Worker Agent Pattern

The system implements a sophisticated Master-Worker pattern where:

1. **Master Agent** (LangGraph Orchestrator):
   - Manages session state and user context
   - Routes requests based on intent classification
   - Coordinates between Worker Agents
   - Handles graceful degradation and recovery

2. **Worker Agents**:
   - **Sales Agent**: Handles empathetic communication, sentiment analysis, and Plan B logic
   - **Verification Agent**: Processes KYC documents with SGS security and SBEF conflict resolution
   - **Underwriting Agent**: Performs risk assessment and EMI calculations using LQM mathematics

### State Machine Flow

```mermaid
stateDiagram-v2
    [*] --> GREETING
    GREETING --> KYC : User Intent Detected
    KYC --> NEGOTIATION : KYC Verified
    KYC --> PLAN_B : KYC Failed
    NEGOTIATION --> SANCTION : Loan Approved
    NEGOTIATION --> PLAN_B : Loan Rejected
    PLAN_B --> NEGOTIATION : Alternative Accepted
    PLAN_B --> [*] : User Exits
    SANCTION --> [*] : PDF Generated
```

### Core Modules

#### LQM (Logic Quantization Module)
- **Purpose**: Zero-hallucination mathematics for financial calculations
- **Key Feature**: Strict `decimal.Decimal` enforcement, NO float operations
- **Usage**: All monetary calculations, EMI computations, precision validation

```python
from app.core.lqm import calculate_emi

# Calculate EMI with zero-hallucination math
emi_result = calculate_emi(
    principal=Decimal('500000.00'),    # ₹5 Lakh in cents
    annual_rate=Decimal('12.00'),      # 12% per annum
    tenure_months=36                   # 3 years
)
```

#### SGS (Spectral-Graph Sentinel)
- **Purpose**: Advanced security scanning and deepfake detection
- **Key Feature**: Mandatory security validation for ALL file uploads
- **Usage**: Document security scanning, fraud detection, file integrity validation

```python
from app.core.sgs import scan_topology

# Mandatory security scan for all uploads
security_score = scan_topology(file_data, filename, user_id)
if not security_score.is_safe():
    raise SecurityError("File failed security scan")
```

#### SBEF (Semantic-Bayesian Evidence Fusion)
- **Purpose**: Intelligent resolution of data conflicts between user input and OCR data
- **Key Feature**: Prevents application rejection due to minor data discrepancies
- **Usage**: Conflict resolution, trust score calculation, data validation

## 🔧 Configuration

### Application Settings

The application uses Pydantic Settings for configuration management:

```python
# app/core/config.py
class Settings(BaseSettings):
    # LQM Configuration (Mathematical Precision)
    default_interest_rate: Decimal = Decimal("12.50")
    max_loan_amount_in_cents: Decimal = Decimal("10000000")  # ₹1 Crore
    min_loan_amount_in_cents: Decimal = Decimal("5000000")   # ₹50,000
    
    # SGS Security Configuration
    sgs_security_threshold: float = 0.85
    max_file_size_mb: int = 10
    
    # Performance Configuration
    api_response_timeout_seconds: float = 2.0
    redis_session_ttl_seconds: int = 3600
```

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://loan2day:password@postgres/loan2day
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
      - kafka

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: loan2day
      POSTGRES_USER: loan2day
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
    depends_on:
      - zookeeper
```

## 🚨 Security Considerations

### File Upload Security
- **Mandatory SGS Scanning**: ALL uploads must pass `SGS.scan_topology()`
- **File Type Validation**: Strict MIME type checking
- **Size Limits**: Configurable file size restrictions
- **Deepfake Detection**: Advanced AI-based authenticity verification

### Data Protection
- **No Hardcoded Secrets**: All sensitive data via environment variables
- **Decimal Precision**: Zero-hallucination mathematics for financial data
- **Session Security**: Redis-based session management with TTL
- **Audit Logging**: Comprehensive security event logging

### API Security
- **Input Validation**: Pydantic V2 validation for all inputs
- **Rate Limiting**: Configurable request rate limits
- **CORS Configuration**: Proper cross-origin resource sharing
- **Error Handling**: Structured error responses without information leakage

## 📈 Performance Optimization

### Response Time Targets
- **API Latency**: < 2 seconds for natural conversation flow
- **Session Retrieval**: Sub-millisecond AgentState retrieval from Redis
- **File Processing**: < 5 seconds for complete KYC document processing

### Scalability Features
- **Async Architecture**: FastAPI with async/await throughout
- **Connection Pooling**: Optimized database connection management
- **Caching Strategy**: Redis for session state and frequently accessed data
- **Message Queuing**: Kafka for asynchronous inter-agent communication

## 🤝 Contributing

### Development Workflow

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/your-username/loan2day.git
   cd loan2day
   ```

2. **Create Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Development Setup**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Code Standards**:
   - Follow LQM Standard: Use `decimal.Decimal` for all monetary values
   - Use structured logging (NO `print()` statements)
   - Implement comprehensive error handling
   - Add type hints for all functions
   - Write Google-style docstrings

5. **Testing**:
   ```bash
   # Run tests before committing
   pytest tests/
   
   # Check code coverage
   pytest --cov=app tests/
   ```

6. **Commit and Push**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

### Code Review Guidelines

- **Security First**: Ensure all file uploads pass SGS scanning
- **Mathematical Precision**: Verify decimal.Decimal usage for monetary values
- **Error Handling**: Implement proper exception handling and logging
- **Performance**: Maintain < 2s API response times
- **Testing**: Include unit tests and property tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- **API Documentation**: http://localhost:8000/docs
- **Architecture Specs**: `.kiro/specs/loan2day/`
- **Design Documents**: `.kiro/specs/loan2day/design.md`
- **Requirements**: `.kiro/specs/loan2day/requirements.md`

### Getting Help
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Security**: Report security issues privately to security@loan2day.com

### Health Monitoring
```bash
# Check application health
curl http://localhost:8000/health

# Check specific service health
curl http://localhost:8000/health/redis
curl http://localhost:8000/health/database
```

---

**Built with ❤️ by the Loan2Day Team**

*Empowering financial inclusion through intelligent AI agents and zero-hallucination mathematics.*