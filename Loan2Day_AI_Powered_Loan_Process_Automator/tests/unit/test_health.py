"""
Unit tests for health check endpoint.

Tests the basic functionality of the FastAPI application
following the LQM Standard for structured testing.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from app.api.main import app


class TestHealthEndpoint:
    """Test suite for health check functionality."""
    
    def setup_method(self):
        """Set up test client for each test method."""
        self.client = TestClient(app)
    
    def test_health_endpoint_returns_200(self):
        """
        Test that health endpoint returns successful status.
        
        Validates: Basic application startup and endpoint availability
        """
        response = self.client.get("/health")
        assert response.status_code == 200
    
    def test_health_endpoint_response_structure(self):
        """
        Test that health endpoint returns proper JSON structure.
        
        Validates: Response format follows HealthCheckResponse model
        """
        response = self.client.get("/health")
        data = response.json()
        
        # Validate required fields exist
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
        assert "services" in data
        
        # Validate field types and values
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert isinstance(data["services"], dict)
        
        # Validate timestamp is recent (within last minute)
        timestamp = datetime.fromisoformat(data["timestamp"])
        now = datetime.now()
        time_diff = (now - timestamp).total_seconds()
        assert time_diff < 60  # Should be very recent
    
    def test_health_endpoint_services_status(self):
        """
        Test that all required services are reported in health check.
        
        Validates: All core system components are monitored
        """
        response = self.client.get("/health")
        services = response.json()["services"]
        
        # Validate all required services are present
        required_services = [
            "database",
            "redis_cache", 
            "mock_bank",
            "sgs_module",
            "lqm_module"
        ]
        
        for service in required_services:
            assert service in services
            assert services[service] == "healthy"  # TODO: Update when real health checks implemented
    
    def test_root_endpoint_returns_welcome_message(self):
        """
        Test that root endpoint returns platform information.
        
        Validates: Basic platform metadata and welcome message
        """
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "architecture" in data
        assert "Loan2Day" in data["message"]
        assert "Master-Worker Agent Pattern" in data["architecture"]