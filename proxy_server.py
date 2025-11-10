import json
import time
import threading
from flask import Flask, request, jsonify
import requests
import google.generativeai as genai
from datetime import datetime, timedelta
import pytz
import os

# --- Configuration Constants ---
PORT = 8080
API_KEYS_FILE = 'api_keys.json'
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3
TARGET_TIMEZONE = 'America/Los_Angeles'

# --- Global Variables & Locks ---
api_keys = []
current_key_index = 0
key_lock = threading.Lock()
file_lock = threading.Lock()

app = Flask(__name__)

# --- Helper Functions ---

def load_api_keys():
    """
    Loads API keys from the specified JSON file.

    Returns:
        list: A list of API key dictionaries, or an empty list if loading fails.
    """
    try:
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: файл '{API_KEYS_FILE}' не найден.")
    except (ValueError, KeyError, IndexError) as e:
        print(f"Ошибка при чтении или обработке '{API_KEYS_FILE}': {e}")
    return []

def save_api_keys():
    """
    Saves the current state of api_keys to the JSON file.
    """
    with file_lock:
        with open(API_KEYS_FILE, 'w') as f:
            json.dump(api_keys, f, indent=4)

def reset_key_index_daily():
    """
    Resets current_key_index to 0 every day at midnight in the target timezone.
    This function is intended to be run in a background thread.
    """
    pt_timezone = pytz.timezone(TARGET_TIMEZONE)
    while True:
        now_pt = datetime.now(pt_timezone)
        tomorrow_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        sleep_seconds = (tomorrow_pt - now_pt).total_seconds()
        
        print(f"Сброс индекса ключей произойдет через {sleep_seconds:.2f} секунд.")
        time.sleep(sleep_seconds)
        
        global current_key_index
        with key_lock:
            current_key_index = 0
            print("Индекс текущего ключа сброшен на 0.")
        time.sleep(1) # Avoid resetting multiple times in the same second

def print_status_tui():
    """
    Prints a simple Text-based User Interface with the current status.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--- Gemini API Proxy Server ---")
    print(f"Статус: {'Работает' if api_keys else 'Ошибка'}")
    print(f"Порт: {PORT}")
    print(f"Загружено ключей: {len(api_keys)}")
    print(f"Текущий индекс ключа: {current_key_index}")
    print("-----------------------------")
    if api_keys:
        for i, key_info in enumerate(api_keys):
            is_current = "<--" if i == current_key_index else ""
            print(f"  - Ключ {i+1}: Использовано токенов: {key_info.get('token_usage', 0)} {is_current}")
    print("-----------------------------")


# --- Flask Endpoint ---

@app.route('/v1beta/models/gemini-pro:generateContent', methods=['POST'])
def proxy_to_gemini():
    """
    Proxies requests to the Google Gemini API, handling key rotation and retries.
    """
    global current_key_index
    request_data = request.get_json()

    if not api_keys:
        return jsonify({"error": "API ключи не загружены или отсутствуют."}), 500

    last_error_response = None

    while current_key_index < len(api_keys):
        api_key = api_keys[current_key_index]['key']
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        print_status_tui()
        print(f"\nИспользуется ключ #{current_key_index + 1}")

        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(gemini_url, json=request_data)

                # --- 429 Error Handling: Rate limit exceeded ---
                if response.status_code == 429:
                    last_error_response = response
                    print(f"Получен статус 429 (Too Many Requests) для ключа #{current_key_index + 1}. Попытка {attempt + 1}/{MAX_RETRIES}")
                    # Key rotation is handled by breaking the inner loop and advancing the outer while loop
                    with key_lock:
                        current_key_index += 1
                    break # Break from retry loop to switch key

                # --- 503 Error Handling: Service Unavailable ---
                if response.status_code == 503 and attempt < MAX_RETRIES - 1:
                    print(f"Получен статус 503 (Service Unavailable). Повторная попытка через {RETRY_DELAY_SECONDS} сек...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue # Retry with the same key

                response.raise_for_status()

                # --- Success Case ---
                response_data = response.json()
                
                # Update token usage
                request_tokens = model.count_tokens(request_data).total_tokens
                response_tokens = model.count_tokens(response_data).total_tokens
                total_tokens = request_tokens + response_tokens
                
                api_keys[current_key_index]['token_usage'] += total_tokens
                save_api_keys()
                
                print(f"Запрос успешен. Использовано токенов: {total_tokens}")
                print_status_tui()

                return response.content, response.status_code, response.headers.items()

            except requests.exceptions.RequestException as e:
                print(f"Ошибка запроса: {e}. Попытка {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    return jsonify({"error": str(e)}), 500
        else: # This 'else' belongs to the for loop, executed if the loop finishes without 'break'
            continue # Continue to the next key if all retries for the current key failed

        break # Break from the while loop if we switched keys due to a 429 error

    if last_error_response:
        return last_error_response.content, last_error_response.status_code, last_error_response.headers.items()
    
    print("Все API ключи исчерпали свою квоту.")
    return jsonify({"error": "All API keys have reached their quota."}), 429

# --- Main Execution ---

if __name__ == '__main__':
    api_keys = load_api_keys()
    if api_keys:
        reset_thread = threading.Thread(target=reset_key_index_daily, daemon=True)
        reset_thread.start()
        print_status_tui()
        app.run(host='0.0.0.0', port=PORT)
    else:
        print("Сервер не может быть запущен из-за ошибки загрузки ключей.")