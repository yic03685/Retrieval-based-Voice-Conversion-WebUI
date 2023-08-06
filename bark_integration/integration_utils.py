import json

def get_config_file(config_path):
    try:
        with open(config_path, 'r') as json_file:
            return json.load(json_file)
            # model_path = data.get("model_path")  # Access the "model_path" field
            # return model_path
    except FileNotFoundError:
        print(f"File '{get_config_file}' not found.")
        return None