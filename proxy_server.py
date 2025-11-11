import json
import time
import random
import string
import threading
from flask import Flask, request, jsonify, Response
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

# --- Helper Functions: File & Key Management ---

def load_api_keys():
    """
    Loads API keys from the specified JSON file.
    """
    try:
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        app.logger.error(f"Ошибка: файл '{API_KEYS_FILE}' не найден.")
    except (ValueError, KeyError, IndexError) as e:
        app.logger.error(f"Ошибка при чтении или обработке '{API_KEYS_FILE}': {e}")
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
        
        app.logger.info(f"Сброс RPD лимитов произойдет через {sleep_seconds:.2f} секунд.")
        time.sleep(sleep_seconds)
        
        with key_lock:
            for key_info in api_keys:
                if 'usage' in key_info:
                    for model_name in key_info['usage']:
                        key_info['usage'][model_name]['rpd_limit_reached'] = False
                        key_info['usage'][model_name]['request_count'] = 0
            save_api_keys()
            app.logger.info("RPD лимиты и счетчики запросов сброшены для всех ключей и моделей.")
        
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

# --- Helper Functions: Request/Response Conversion ---

def convert_openai_to_gemini_request(openai_request):
    """
    Converts an OpenAI-formatted chat completion request to a Gemini-formatted one.
    """
    messages = openai_request.get('messages', [])
    gemini_contents = []
    system_prompt = None

    for message in messages:
        role = message.get('role')
        content = message.get('content')

        if role == 'system':
            system_prompt = content
            continue

        gemini_role = 'user' if role == 'user' else 'model'

        if system_prompt and gemini_role == 'user':
            if isinstance(content, str):
                content = f"{system_prompt}\n{content}"
            elif isinstance(content, list):
                if content and isinstance(content[0], dict) and content[0].get('type') == 'text':
                    content[0]['text'] = f"{system_prompt}\n{content[0]['text']}"
                else:
                    content.insert(0, {'type': 'text', 'text': system_prompt})
            system_prompt = None

        gemini_contents.append({'role': gemini_role, 'parts': [{'text': content}]})

    return {'contents': gemini_contents}

def convert_gemini_to_openai_response(gemini_response_json, model_name):
    """
    Converts a Gemini response to the OpenAI chat completion format.
    """
    completion_id = 'chatcmpl-' + ''.join(random.choices(string.ascii_letters + string.digits, k=29))
    created_time = int(time.time())
    choices = []

    if gemini_response_json.get('candidates'):
        for candidate in gemini_response_json['candidates']:
            content = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')
            choices.append({
                "index": candidate.get('index', 0),
                "message": {"role": "assistant", "content": content},
                "finish_reason": candidate.get('finishReason', 'stop')
            })

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model_name,
        "choices": choices,
        "usage": usage
    }

# --- Helper Functions: Error Handling ---

def handle_rate_limit_error(error_response, key_index, model_name):
    """
    Parses a 429 error for retry-after info or identifies it as a daily limit error.
    Returns delay in seconds if retryable, or None if it's a hard limit.
    """
    try:
        error_json = error_response.json()
        error_message_str = error_json.get("error", {}).get("message", "")
        
        inner_error_json = json.loads(error_message_str)
        details = inner_error_json.get("error", {}).get("details", [])
        
        retry_info = next((d for d in details if d.get("@type") == "type.googleapis.com/google.rpc.RetryInfo"), None)

        if retry_info and 'retryDelay' in retry_info:
            delay_str = retry_info['retryDelay'].replace('s', '')
            delay_seconds = float(delay_str)
            app.logger.warning(f"Получен статус 429 с retryDelay. Ожидание {delay_seconds} сек...")
            return delay_seconds
            
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as json_e:
        app.logger.error(f"Не удалось распарсить retryDelay из ответа 429: {json_e}")

    if "quotaMetric" in error_response.text:
        app.logger.warning(f"Получен статус 429 (RPD Limit) для ключа #{key_index + 1} и модели '{model_name}'.")
        with key_lock:
            # Убедимся, что 'usage' и 'model_name' существуют перед записью
            usage_data = api_keys[key_index].setdefault('usage', {})
            model_usage = usage_data.setdefault(model_name, {
                'token_count': 0, 'request_count': 0, 'rpd_limit_reached': False
            })
            model_usage['rpd_limit_reached'] = True
            save_api_keys()
    
    return None

# --- Flask Endpoints ---

@app.route('/', methods=['GET'])
def status_page():
    """
    Generates and returns an HTML page with the real-time status of API key usage.
    """
    html = """
    <html>
    <head>
        <title>Gemini API Proxy Status</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2em; background-color: #f4f4f9; color: #333; }
            h1, h2 { color: #444; }
            .key-block { background-color: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 1em; margin-bottom: 1em; }
            .model-info { margin-left: 2em; }
            .rpd-ok { color: green; }
            .rpd-reached { color: red; }
        </style>
    </head>
    <body>
        <h1>Gemini API Proxy Status</h1>
"""
    with key_lock:
        html += f"<p><strong>Total Keys Loaded:</strong> {len(api_keys)}</p>"
        for i, key_info in enumerate(api_keys):
            html += f'<div class="key-block"><h2>Key #{i+1}</h2>'
            if not key_info.get('usage'):
                html += "<p>No usage data yet.</p>"
            else:
                for model_name, usage_data in sorted(key_info['usage'].items()):
                    rpd_status = "Reached" if usage_data.get('rpd_limit_reached', False) else "OK"
                    rpd_class = "rpd-reached" if usage_data.get('rpd_limit_reached', False) else "rpd-ok"
                    html += f"""
                    <div class="model-info">
                        <p><strong>Model:</strong> {model_name}</p>
                        <p>Tokens: {usage_data.get('token_count', 0)}</p>
                        <p>Requests: {usage_data.get('request_count', 0)}</p>
                        <p>RPD Status: <span class="{rpd_class}">{rpd_status}</span></p>
                    </div>
                    """
            html += '</div>'
    html += "</body></html>"
    return html

@app.route('/v1/models', methods=['GET'])
def list_models():
    """
    Provides a list of supported models in the OpenAI format.
    """
    model_data = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google"
        } for model_id in SUPPORTED_MODELS
    ]
    return jsonify({"object": "list", "data": model_data})

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    Handles OpenAI-compatible chat completion requests.
    """
    openai_request = request.json
    model_name = openai_request.get('model')

    gemini_request_data = convert_openai_to_gemini_request(openai_request)
    
    internal_proxy_url = f"http://127.0.0.1:{PORT}/v1beta/models/{model_name}:generateContent"

    try:
        response = requests.post(internal_proxy_url, json=gemini_request_data, timeout=300)
        response.raise_for_status()
        gemini_response_json = response.json()

        openai_response = convert_gemini_to_openai_response(gemini_response_json, model_name)
        return jsonify(openai_response)

    except requests.exceptions.RequestException as e:
        error_message = f"Failed to connect to the internal Gemini proxy: {str(e)}"
        status_code = 500
        if e.response is not None:
            error_message = e.response.text
            status_code = e.response.status_code
        app.logger.error(f"Error in chat_completions: {error_message}")
        return jsonify({"error": error_message}), status_code

@app.route('/v1beta/models/<string:model_name>:generateContent', methods=['POST'])
def proxy_to_gemini(model_name):
    """
    Proxies requests to the Google Gemini API, handling key rotation, limits, and retries.
    """
    request_data = request.get_json()

    if not api_keys:
        app.logger.error("API keys are not loaded or missing.")
        return jsonify({"error": "API ключи не загружены или отсутствуют."}), 500

    last_error_response = None

    for key_index, key_info in enumerate(api_keys):
        with key_lock:
            usage_data = key_info.setdefault('usage', {})
            model_usage = usage_data.setdefault(model_name, {
                'token_count': 0, 'request_count': 0, 'rpd_limit_reached': False
            })
            
            if model_usage.get('rpd_limit_reached', False):
                app.logger.info(f"Ключ #{key_index + 1} уже достиг RPD лимита для модели '{model_name}'. Пропускаем.")
                continue

        api_key_data = key_info['key']
        if isinstance(api_key_data, dict):
            # Обработка старого формата, где ключ находится внутри вложенной структуры
            api_key = api_key_data.get('key', [None, None])[1]
        else:
            # Обработка нового формата, где ключ является простой строкой
            api_key = api_key_data
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        print_status_tui()
        app.logger.info(f"Используется ключ #{key_index + 1} для модели '{model_name}'")

        for attempt in range(MAX_RETRIES):
            try:
                headers = {'x-goog-api-key': api_key}
                response = requests.post(gemini_url, headers=headers, json=request_data)

                if response.status_code == 503 and attempt < MAX_RETRIES - 1:
                    app.logger.warning(f"Получен статус 503. Повторная попытка через {RETRY_DELAY_SECONDS} сек...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue

                response.raise_for_status()

                response_data = response.json()
                
                with key_lock:
                    api_keys[key_index]['usage'][model_name]['request_count'] += 1
                    save_api_keys()
                
                app.logger.info(f"Запрос успешен.")
                print_status_tui()

                # Создаем и возвращаем правильный объект Response
                headers_dict = {k: v for k, v in response.headers.items() if k.lower() not in ['transfer-encoding', 'content-encoding']}
                return Response(response.content, status=response.status_code, headers=headers_dict)

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    delay = handle_rate_limit_error(e.response, key_index, model_name)
                    if delay is not None:
                        time.sleep(delay)
                        continue  # Retry with the same key after delay
                    else:
                        last_error_response = e.response
                        break # RPD limit reached, break to try next key
                
                app.logger.error(f"Ошибка запроса: {e}. Попытка {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    last_error_response = e.response

            except requests.exceptions.RequestException as e:
                app.logger.error(f"Ошибка запроса: {e}. Попытка {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    last_error_response = (jsonify({"error": str(e)}), 500)
        
    if last_error_response:
        app.logger.error("Все API ключи были опробованы, возвращается последняя ошибка.")
        if isinstance(last_error_response, tuple):
             return last_error_response
        # Также оборачиваем ответ с ошибкой в Response
        headers_dict = {k: v for k, v in last_error_response.headers.items() if k.lower() not in ['transfer-encoding', 'content-encoding']}
        return Response(last_error_response.content, status=last_error_response.status_code, headers=headers_dict)
    
    app.logger.error(f"All API keys have reached their daily limit for model {model_name}.")
    return jsonify({
        "error": {
            "code": 503,
            "message": f"Service Unavailable: All available API keys have reached their daily usage limit for the requested model ({model_name}). Please try again later.",
            "status": "SERVICE_UNAVAILABLE"
        }
    }), 503

# --- Main Execution ---

if __name__ == '__main__':
    api_keys = load_api_keys()
    if api_keys:
        reset_thread = threading.Thread(target=reset_rpd_limits_daily, daemon=True)
        reset_thread.start()
        print_status_tui()
        app.run(host='0.0.0.0', port=PORT)
    else:
        app.logger.error("Сервер не может быть запущен из-за ошибки загрузки ключей.")