import requests
import json

PROXY_URL = "http://127.0.0.1:8080"

def test_status_page():
    """Tests the status page of the proxy server."""
    print("--- Running test_status_page ---")
    try:
        response = requests.get(PROXY_URL + "/")
        assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
        assert "<html>" in response.text, "Response text does not contain '<html>'"
        assert "Gemini API Proxy Status" in response.text, "Response text does not contain 'Gemini API Proxy Status'"
        print("✅ test_status_page: PASSED")
    except Exception as e:
        print(f"❌ test_status_page: FAILED - {e}")

def test_models_endpoint():
    """Tests the /models endpoint of the proxy server."""
    print("--- Running test_models_endpoint ---")
    try:
        response = requests.get(PROXY_URL + "/models")
        assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
        data = response.json()
        assert "data" in data, "Response JSON does not contain 'data' key"
        print("✅ test_models_endpoint: PASSED")
    except Exception as e:
        print(f"❌ test_models_endpoint: FAILED - {e}")

def test_chat_completions():
    """Tests the /chat/completions endpoint of the proxy server."""
    print("--- Running test_chat_completions ---")
    try:
        payload = {
            "model": "gemini-1.5-pro-latest",
            "messages": [
                {"role": "user", "content": "Hello, what is the capital of France?"}
            ]
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(PROXY_URL + "/chat/completions", data=json.dumps(payload), headers=headers)
        
        assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
        data = response.json()
        assert "choices" in data, "Response JSON does not contain 'choices' key"
        assert "Paris" in data["choices"][0]["message"]["content"], "Response message does not contain 'Paris'"
        print("✅ test_chat_completions: PASSED")
    except Exception as e:
        print(f"❌ test_chat_completions: FAILED - {e}")

if __name__ == "__main__":
    test_status_page()
    test_models_endpoint()
    test_chat_completions()