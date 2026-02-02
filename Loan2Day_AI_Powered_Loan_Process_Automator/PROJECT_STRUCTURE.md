# 📁 Loan2Day Project Structure

This document outlines the organized directory structure of the Loan2Day Agentic AI Fintech Platform.

## 🏗️ **Root Directory Structure**

```
Loan2Day_AI_Powered_Loan_Process_Automator/
├── 📁 app/                     # Main application code
├── 📁 frontend/               # React frontend application
├── 📁 tests/                  # Comprehensive test suite
├── 📁 .kiro/                  # Project specifications & documentation
├── 📁 docs/                   # Additional documentation
├── 📁 scripts/                # Database and deployment scripts
├── 📁 nginx/                  # Nginx configuration
├── 📁 utils/                  # Development utilities
├── 📁 logs/                   # Runtime logs (gitignored except .gitkeep)
├── 📁 uploads/                # Temporary file uploads (gitignored except .gitkeep)
├── 📁 generated_pdfs/         # Generated PDF documents (gitignored except .gitkeep)
├── 📄 README.md               # Main project documentation
├── 📄 requirements.txt        # Python dependencies
├── 🐳 docker-compose.yml      # Docker services configuration
├── 🐳 Dockerfile             # Production Docker image
├── 🐳 Dockerfile.dev          # Development Docker image
├── 📄 Makefile               # Build and deployment commands
└── 📄 .env.example           # Environment variables template
```

## 📂 **Detailed Directory Breakdown**

### **🐍 `/app/` - Main Application**
```
app/
├── 🤖 agents/                 # Master-Worker Agent implementations
│   ├── master.py             # LangGraph Master Agent orchestrator
│   ├── sales.py              # Sales Agent with sentiment analysis
│   ├── verification.py       # KYC & SGS security agent
│   └── underwriting.py       # EMI calculation & risk assessment
├── 🚪 api/                   # FastAPI routes and endpoints
│   ├── main.py               # FastAPI application entry point
│   └── routes/               # API route handlers
│       ├── chat.py           # Chat interface endpoints
│       ├── upload.py         # KYC document upload
│       ├── plan_b.py         # Plan B recovery endpoints
│       ├── documents.py      # PDF generation endpoints
│       └── voice.py          # Twilio voice integration
├── 🧠 core/                  # Core business logic modules
│   ├── lqm.py                # Logic Quantization Module (decimal math)
│   ├── sgs.py                # Spectral-Graph Sentinel (security)
│   ├── mock_bank.py          # Mock banking API for development
│   ├── config.py             # Application configuration
│   ├── database.py           # Database connection management
│   ├── security.py           # Security utilities
│   ├── error_handling.py     # Error handling framework
│   └── middleware.py         # FastAPI middleware
├── 📊 models/                # Data models
│   ├── pydantic_models.py    # Pydantic models for API validation
│   └── db_models.py          # SQLAlchemy database models
├── 🗄️ repositories/          # Data access layer
│   ├── base_repository.py    # Base repository pattern
│   ├── user_repository.py    # User data operations
│   ├── loan_application_repository.py
│   ├── kyc_document_repository.py
│   └── audit_log_repository.py
└── 🔧 services/              # Business services
    ├── session_service.py    # Session management
    ├── pdf_service.py        # PDF generation service
    ├── voice_service.py      # Voice processing service
    ├── kafka_service.py      # Kafka messaging service
    └── agent_messaging.py    # Inter-agent communication
```

### **⚛️ `/frontend/` - React Application**
```
frontend/
├── 📁 src/                   # Source code
│   ├── components/           # React components
│   ├── hooks/                # Custom React hooks
│   ├── services/             # API service calls
│   ├── store/                # Redux state management
│   ├── App.tsx               # Main application component
│   └── main.tsx              # Application entry point
├── 📁 public/                # Static assets
├── 📄 package.json           # Node.js dependencies
├── 📄 vite.config.ts         # Vite build configuration
├── 📄 tailwind.config.js     # Tailwind CSS configuration
└── 🐳 Dockerfile            # Frontend Docker image
```

### **🧪 `/tests/` - Test Suite**
```
tests/
├── 📁 unit/                  # Unit tests
│   ├── test_lqm.py           # LQM module tests
│   ├── test_health.py        # Health endpoint tests
│   ├── test_plan_b_logic.py  # Plan B logic tests
│   └── test_kafka_service.py # Kafka service tests
├── 📁 property/              # Property-based tests
│   ├── test_lqm_properties.py        # Mathematical correctness
│   ├── test_pdf_properties.py        # PDF generation properties
│   ├── test_sgs_properties.py        # Security properties
│   ├── test_error_handling_properties.py
│   └── test_async_database_properties.py
├── 📁 integration/           # Integration tests
│   └── test_kafka_integration.py
└── 📄 README.md              # Testing documentation
```

### **📋 `/.kiro/specs/` - Project Specifications**
```
.kiro/specs/loan2day/
├── 📄 requirements.md        # Business requirements & acceptance criteria
├── 📄 design.md             # Technical architecture & design patterns
└── 📄 tasks.md              # Implementation roadmap & status
```

### **🔧 `/utils/` - Development Utilities**
```
utils/
├── cleanup.py                # Project cleanup utility
└── [future utilities]        # Additional development tools
```

### **📁 Runtime Directories**
```
logs/                         # Application logs (runtime only)
├── .gitkeep                  # Ensures directory is tracked
└── [*.log files]            # Generated at runtime

uploads/                      # Temporary file uploads (runtime only)
├── .gitkeep                  # Ensures directory is tracked
└── [uploaded files]         # KYC documents (temporary)

generated_pdfs/               # Generated PDF documents (runtime only)
├── .gitkeep                  # Ensures directory is tracked
└── [*.pdf files]            # Sanction letters (temporary)
```

## 🎯 **Key Organizational Principles**

### **1. Separation of Concerns**
- **`/app/`**: Core business logic and API
- **`/frontend/`**: User interface and client-side logic
- **`/tests/`**: All testing code isolated from production
- **`/.kiro/specs/`**: Documentation and specifications

### **2. Clean Architecture**
- **Routes → Services → Repositories**: Clear data flow pattern
- **Master-Worker Agents**: Centralized orchestration with specialized workers
- **Security-First**: SGS scanning mandatory for all file operations

### **3. Development Experience**
- **`/utils/`**: Development tools and utilities
- **Runtime directories**: Proper separation of generated content
- **Docker configuration**: Consistent development and deployment

### **4. Documentation Strategy**
- **README.md**: Main project overview and quick start
- **PROJECT_STRUCTURE.md**: This file - detailed organization
- **`.kiro/specs/`**: Complete technical specifications
- **Code documentation**: Google-style docstrings throughout

## 🧹 **Maintenance**

### **Cleanup Utility**
Use the cleanup utility to maintain a clean development environment:

```bash
# Dry run to see what would be cleaned
python utils/cleanup.py --dry-run

# Actually clean up temporary files
python utils/cleanup.py
```

### **Git Ignore Strategy**
- **Runtime files**: Logs, uploads, generated PDFs are gitignored
- **Dependencies**: node_modules, venv, __pycache__ excluded
- **Specifications**: `.kiro/` directory is intentionally tracked
- **Environment**: `.env` files excluded, `.env.example` tracked

## 🚀 **Getting Started**

1. **Clone the repository**
2. **Follow README.md** for setup instructions
3. **Check `.kiro/specs/`** for detailed requirements and design
4. **Use `utils/cleanup.py`** to maintain clean environment
5. **Follow the directory structure** when adding new features

This organized structure ensures maintainability, scalability, and clear separation of concerns for the Loan2Day platform.