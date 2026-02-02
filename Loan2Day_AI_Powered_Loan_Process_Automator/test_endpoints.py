#!/usr/bin/env python3
"""
Quick end-to-end test script for Loan2Day API endpoints.
"""

from fastapi.testclient import TestClient
from app.api.main import app
import json

def test_endpoints():
    client = TestClient(app)
    
    print("🧪 Testing Loan2Day API Endpoints...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = client.get('/health')
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Health endpoint working")
            services = list(response.json()['services'].keys())
            print(f"   Services: {services}")
        else:
            print(f"   ❌ Health endpoint failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Health endpoint error: {e}")
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = client.get('/')
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Root endpoint working")
        else:
            print(f"   ❌ Root endpoint failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Root endpoint error: {e}")
    
    # Test chat endpoint
    print("\n3. Testing chat endpoint...")
    try:
        response = client.post('/v1/chat/message', json={
            'message': 'I want to apply for a loan',
            'user_id': 'test_user_123',
            'session_id': 'test_session_123'
        })
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Chat endpoint working")
        else:
            print(f"   ❌ Chat endpoint failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Chat endpoint error: {e}")
    
    print("\n🎯 End-to-end API test completed!")

if __name__ == "__main__":
    test_endpoints()