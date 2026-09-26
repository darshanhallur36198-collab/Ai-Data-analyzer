from backend.modules.chat import chat_with_data

def ask_ai_assistant(file_path: str, query: str, api_key: str = None):
    return chat_with_data(file_path, query, api_key=api_key)
