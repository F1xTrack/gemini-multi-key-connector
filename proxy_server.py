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
SUPPORTED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# --- Global Variables & Locks ---
api_keys = []
key_lock = threading.Lock()
file_lock = threading.Lock()

app = Flask(__name__)

# --- Helper Functions ---

def load_api_keys():
    """
    Loads API keys from the specified JSON file.
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

def reset_rpd_limits_daily():
    """
    Resets RPD (Requests Per Day) limits for all models on all keys at midnight.
    This function is intended to be run in a background thread.
    """
    pt_timezone = pytz.timezone(TARGET_TIMEZONE)
    while True:
        now_pt = datetime.now(pt_timezone)
        tomorrow_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        sleep_seconds = (tomorrow_pt - now_pt).total_seconds()
        
        print(f"Сброс RPD лимитов произойдет через {sleep_seconds:.2f} секунд.")
        time.sleep(sleep_seconds)
        
        with key_lock:
            for key_info in api_keys:
                if 'usage' in key_info:
                    for model_name in key_info['usage']:
                        key_info['usage'][model_name]['rpd_limit_reached'] = False
                        key_info['usage'][model_name]['request_count'] = 0
            save_api_keys()
            print("RPD лимиты и счетчики запросов сброшены для всех ключей и моделей.")
        
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
    print("-----------------------------")
    if api_keys:
        for i, key_info in enumerate(api_keys):
            print(f"  - Ключ {i+1}:")
            if 'usage' in key_info:
                for model, usage in key_info['usage'].items():
                    limit_status = "ДОСТИГНУТ" if usage.get('rpd_limit_reached', False) else "OK"
                    print(f"    - Модель: {model}")
                    print(f"      Токены: {usage.get('token_count', 0)}")
                    print(f"      Запросы: {usage.get('request_count', 0)}")
                    print(f"      Лимит RPD: {limit_status}")
            else:
                print("    - Данные об использовании отсутствуют.")
    print("-----------------------------")


# --- Flask Endpoint ---

@app.route('/models', methods=['GET'])
def get_models():
    """
    Returns a list of available models in a format compatible with OpenAI clients.
    """
    model_list = []
    for model_name in SUPPORTED_MODELS:
        model_list.append({
            "id": f"{model_name}:generateContent",
            "object": "model",
            "created": 1686935002,
            "owned_by": "google"
        })
    
    return jsonify({
        "object": "list",
        "data": model_list
    })

@app.route('/v1beta/models/<string:model_name>:generateContent', methods=['POST'])
def proxy_to_gemini(model_name):
    """
    Proxies requests to the Google Gemini API, handling key rotation, model-specific limits, and retries.
    """
    request_data = request.get_json()

    if not api_keys:
        return jsonify({"error": "API ключи не загружены или отсутствуют."}), 500

    last_error_response = None

    for key_index, key_info in enumerate(api_keys):
        with key_lock:
            # --- Check RPD Limit before making a request ---
            usage_data = key_info.setdefault('usage', {})
            model_usage = usage_data.setdefault(model_name, {
                'token_count': 0,
                'request_count': 0,
                'rpd_limit_reached': False
            })
            
            if model_usage.get('rpd_limit_reached', False):
                print(f"Ключ #{key_index + 1} уже достиг RPD лимита для модели '{model_name}'. Пропускаем.")
                continue

        api_key = key_info['key']
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        
        print_status_tui()
        print(f"\nИспользуется ключ #{key_index + 1} для модели '{model_name}'")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(gemini_url, json=request_data)

                # --- Other 503 Error Handling: Retryable errors ---
                if response.status_code == 503 and attempt < MAX_RETRIES - 1:
                    print(f"Получен статус {response.status_code}. Повторная попытка через {RETRY_DELAY_SECONDS} сек...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue # Retry with the same key

                response.raise_for_status()

                # --- Success Case (200 OK) ---
                response_data = response.json()
                
                # Configure genai for token counting
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                
                request_tokens = model.count_tokens(request_data).total_tokens
                response_tokens = model.count_tokens(response_data).total_tokens
                total_tokens = request_tokens + response_tokens
                
                with key_lock:
                    api_keys[key_index]['usage'][model_name]['token_count'] += total_tokens
                    api_keys[key_index]['usage'][model_name]['request_count'] += 1
                    save_api_keys()
                
                print(f"Запрос успешен. Использовано токенов: {total_tokens}")
                print_status_tui()

                return response.content, response.status_code, response.headers.items()

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    try:
                        error_json = e.response.json()
                        error_message_str = error_json.get("error", {}).get("message", "")
                        
                        inner_error_json = json.loads(error_message_str)
                        details = inner_error_json.get("error", {}).get("details", [])
                        
                        retry_info = next((d for d in details if d.get("@type") == "type.googleapis.com/google.rpc.RetryInfo"), None)

                        if retry_info and 'retryDelay' in retry_info:
                            delay_str = retry_info['retryDelay'].replace('s', '')
                            delay_seconds = float(delay_str)
                            print(f"Получен статус 429 с retryDelay. Ожидание {delay_seconds} сек...")
                            time.sleep(delay_seconds)
                            continue # Повторная попытка с тем же ключом
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as json_e:
                        print(f"Не удалось распарсить retryDelay из ответа 429: {json_e}")
                        # RPD limit check as a fallback
                        if "quotaMetric" in e.response.text:
                            print(f"Получен статус 429 (RPD Limit) для ключа #{key_index + 1} и модели '{model_name}'.")
                            with key_lock:
                                api_keys[key_index]['usage'][model_name]['rpd_limit_reached'] = True
                                save_api_keys()
                            last_error_response = e.response
                            break # Break from retry loop to switch to the next key
                
                print(f"Ошибка запроса: {e}. Попытка {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    last_error_response = e.response

            except requests.exceptions.RequestException as e:
                print(f"Ошибка запроса: {e}. Попытка {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    last_error_response = jsonify({"error": str(e)}), 500
        
        # If the retry loop finishes without success, we've already broken out to the next key for RPD limit
        # or we continue the outer loop to the next key
        
    # --- All keys have been tried ---
    if last_error_response:
        print("Все API ключи были опробованы, возвращается последняя ошибка.")
        if isinstance(last_error_response, tuple): # Our custom error
             return last_error_response
        return last_error_response.content, last_error_response.status_code, last_error_response.headers.items()
    
    print("Все API ключи исчерпали свою квоту или недоступны.")
    return jsonify({"error": "All available API keys have reached their quota or are unable to process the request."}), 429

# --- Main Execution ---

if __name__ == '__main__':
    api_keys = load_api_keys()
    if api_keys:
        reset_thread = threading.Thread(target=reset_rpd_limits_daily, daemon=True)
        reset_thread.start()
        print_status_tui()
        app.run(host='0.0.0.0', port=PORT)
    else:
        print("Сервер не может быть запущен из-за ошибки загрузки ключей.")