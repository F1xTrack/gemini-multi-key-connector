import csv
import json
import os

def process_api_keys():
    """
    Reads API keys from a CSV file, creates a JSON file with the keys,
    and then deletes the original CSV file.
    """
    csv_file = 'cart_277693380_1.csv'
    json_file = 'api_keys.json'
    api_keys = []

    try:
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                if row:
                    # Assuming the API key is in the second column (index 1)
                    api_key = row
                    api_keys.append({"key": api_key, "token_usage": 0})

        with open(json_file, 'w') as f:
            json.dump(api_keys, f, indent=4)

        print(f"Successfully created {json_file}")

        os.remove(csv_file)
        print(f"Successfully deleted {csv_file}")

    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_api_keys()