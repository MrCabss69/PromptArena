# core/llm_client.py

import os
import httpx
import logging
from typing import Optional, Tuple, Dict, Any, List, Set

# --- Configuration ---
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": os.getenv("PROMPT_ARENA_REFERER", "http://localhost:7860"),
    "X-Title": os.getenv("PROMPT_ARENA_TITLE", "PromptArena MVP"),
}

OPENROUTER_MODEL_PREFIXES: Set[str] = {
    "openai/", "google/", "anthropic/", "mistralai/", "meta-llama/",
    "huggingfaceh4/", "microsoft/", "nousresearch/", "gryphe/", "teknium/",
    "togethercomputer/", "koboldai/"
}


# Ollama Config
DEFAULT_OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama API URL
OLLAMA_CHAT_ENDPOINT = "/api/chat"
OLLAMA_HEADERS = {"Content-Type": "application/json"}

# General LLM Defaults
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60.0 
logger = logging.getLogger(__name__)

# --- Helper to determine provider ---
def get_llm_provider(model_name: str) -> str:
    """Determines if the model is likely OpenRouter or Ollama based on naming conventions."""
    if any(model_name.startswith(prefix) for prefix in OPENROUTER_MODEL_PREFIXES):
        return "openrouter"
    logger.debug(f"Model '{model_name}' not matching OpenRouter prefixes, assuming Ollama.")
    return "ollama"

# --- Funcionalidad Principal ---

def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT
) -> Tuple[Optional[str], Optional[str]]:
    """
    Envía un prompt a un modelo LLM a través de OpenRouter o Ollama.

    Determina el proveedor (OpenRouter/Ollama) basado en el nombre del modelo.

    Args:
        prompt: El prompt principal del usuario.
        system_prompt: Prompt opcional del sistema para guiar al modelo.
        model: El identificador del modelo a usar. Si es None, usa el default
               del proveedor inferido o de OpenRouter si no se puede inferir.
        temperature: Parámetro de creatividad/aleatoriedad (0.0 a 2.0).
        max_tokens: Límite máximo de tokens en la respuesta.
        timeout: Tiempo máximo de espera para la respuesta de la API (segundos).

    Returns:
        Una tupla (result, error_message):
        - Si éxito: (contenido_del_mensaje, None)
        - Si error: (None, mensaje_de_error_descriptivo)
    """
    provider = "openrouter"
    if model:
        provider = get_llm_provider(model)
    else:
        model = DEFAULT_OPENROUTER_MODEL
        provider = "openrouter"
        logger.warning(f"No model specified, defaulting to OpenRouter model: {model}")

    # --- Prepare request based on provider ---
    url: str = ""
    headers: Dict[str, str] = {}
    payload: Dict[str, Any] = {}
    effective_model: str = model
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # --- OpenRouter Specifics ---
    if provider == "openrouter":
        api_key = os.getenv(OPENROUTER_API_KEY_ENV)
        if not api_key:
            error_msg = f"La variable de entorno {OPENROUTER_API_KEY_ENV} no está configurada para OpenRouter."
            logger.error(error_msg)
            return None, error_msg

        url = OPENROUTER_URL
        headers = {**OPENROUTER_HEADERS, "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        logger.info(f"Llamando a OpenRouter API: model={effective_model}, temp={temperature}, max_tokens={max_tokens}")

    # --- Ollama Specifics ---
    elif provider == "ollama":
        ollama_base_url = os.getenv(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_BASE_URL)
        url = ollama_base_url.rstrip('/') + OLLAMA_CHAT_ENDPOINT
        headers = OLLAMA_HEADERS
        effective_model = model if model else DEFAULT_OLLAMA_MODEL 
        payload = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "options": {
                 "num_predict": max_tokens
            }
        }
        logger.info(f"Llamando a Ollama API: url={url}, model={effective_model}, temp={temperature}, max_tokens(num_predict)={max_tokens}")

    else:
        # Should not happen if get_llm_provider is exhaustive, but good practice
        error_msg = f"Proveedor LLM desconocido '{provider}' inferido del modelo '{model}'."
        logger.error(error_msg)
        return None, error_msg

    # --- Execute HTTP Request ---
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code >= 400:
                 logger.error(f"Request to {provider} failed. URL: {url}, Status: {response.status_code}, Response Body: {response.text[:500]}...") 
            response.raise_for_status()
        data = response.json()
        logger.info(f"Respuesta recibida de {provider} (status={response.status_code})")

        # --- Process Response (Provider Specific) ---
        content: Optional[str] = None
        parse_error: Optional[str] = None

        if provider == "openrouter":
            # OpenRouter response parsing (existing logic)
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                first_choice = data["choices"][0]
                if "message" in first_choice and isinstance(first_choice["message"], dict) and "content" in first_choice["message"]:
                    if isinstance(first_choice["message"]["content"], str):
                        content = first_choice["message"]["content"]
                    else:
                        parse_error = f"OpenRouter: Contenido del mensaje no es string ({type(first_choice['message']['content'])})."
                else:
                    parse_error = f"OpenRouter: Estructura inesperada en 'choices[0].message'. Recibido: {first_choice.get('message', 'N/A')}"
            else:
                parse_error = f"OpenRouter: Respuesta JSON sin 'choices' o vacía. Recibido: {str(data)[:200]}..."

        elif provider == "ollama":
             if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                 if isinstance(data["message"]["content"], str):
                     content = data["message"]["content"]
                 else:
                     parse_error = f"Ollama: Contenido del mensaje no es string ({type(data['message']['content'])})."
             elif "error" in data:
                 parse_error = f"Ollama API devolvió un error: {data['error']}"
             else:
                 parse_error = f"Ollama: Respuesta JSON sin clave 'message' o 'error' esperada. Recibido: {str(data)[:200]}..."

        # --- Return result or error ---
        if content is not None:
            return content, None
        else:
            error_msg = parse_error if parse_error else "Error desconocido al procesar la respuesta del LLM."
            logger.error(f"Error procesando respuesta de {provider}: {error_msg}")
            return None, error_msg

    # --- Exception Handling ---
    except httpx.HTTPStatusError as e:
        error_body = e.response.text
        error_msg = f"Error HTTP {e.response.status_code} de {provider} API: {error_body[:500]}..."
        logger.error(error_msg, exc_info=logger.isEnabledFor(logging.DEBUG))
        return None, error_msg
    except httpx.TimeoutException as e:
        error_msg = f"Timeout ({timeout}s) esperando respuesta de {provider} API ({url}): {e}"
        logger.error(error_msg)
        return None, error_msg
    except httpx.RequestError as e:
        error_msg = f"Error de conexión/red al llamar a {provider} API ({url}): {e}"
        logger.error(error_msg, exc_info=logger.isEnabledFor(logging.DEBUG))
        return None, error_msg
    except Exception as e:
        error_msg = f"Error inesperado en el cliente LLM ({provider}, {url}): {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

# --- Tests rápidos ---
if __name__ == "__main__":
    # Configure logging for tests if basicConfig wasn't called elsewhere
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("--- Testing LLM Client (OpenRouter & Ollama) ---")

    # --- OpenRouter Tests ---
    print("\n--- OpenRouter Tests ---")
    # Asegúrate de tener OPENROUTER_API_KEY definida
    test_prompt_or = "Escribe una función Python simple."
    print(f"\nTest OR-1: Llamada OpenRouter válida (default: {DEFAULT_OPENROUTER_MODEL})")
    content_or, error_or = call_llm(test_prompt_or, model=DEFAULT_OPENROUTER_MODEL) # Explicitly test default
    if error_or:
        print(f"  -> Error: {error_or}")
    else:
        print(f"  -> Success! OR Received (first 50): {content_or[:50]}...")

    # Test OR Key error
    print("\nTest OR-2: Llamada OpenRouter sin API Key")
    temp_key = os.environ.pop(OPENROUTER_API_KEY_ENV, None)
    content_or_no_key, error_or_no_key = call_llm(test_prompt_or, model=DEFAULT_OPENROUTER_MODEL)
    if error_or_no_key and OPENROUTER_API_KEY_ENV in error_or_no_key:
        print(f"  -> Success! Received expected key error: {error_or_no_key}")
    else:
        print(f"  -> Failed! Expected key error but got: {error_or_no_key} / {content_or_no_key}")
    if temp_key:
        os.environ[OPENROUTER_API_KEY_ENV] = temp_key


    # --- Ollama Tests ---
    print("\n--- Ollama Tests ---")
    # Asegúrate de tener un servidor Ollama corriendo (default en http://localhost:11434)
    test_prompt_ol = "Explica qué es Ollama en una frase."
    ollama_test_model = os.getenv("OLLAMA_TEST_MODEL", DEFAULT_OLLAMA_MODEL)
    print(f"\nTest OL-1: Llamada Ollama válida (model: {ollama_test_model})")
    print(f"           (Asegúrate que Ollama corre en {os.getenv(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_BASE_URL)} y tiene el modelo '{ollama_test_model}')")
    content_ol, error_ol = call_llm(test_prompt_ol, model=ollama_test_model)
    if error_ol:
        print(f"  -> Error: {error_ol}")
        print(f"  -> (Si es error de conexión, verifica que Ollama esté corriendo)")
    else:
        print(f"  -> Success! Ollama Received: {content_ol}")

    # Test Ollama connection error (cambiando la URL)
    print("\nTest OL-2: Llamada Ollama con URL inválida (espera error conexión)")
    original_ollama_url = os.environ.get(OLLAMA_BASE_URL_ENV)
    os.environ[OLLAMA_BASE_URL_ENV] = "http://invalid-host-for-testing:11434"
    content_ol_badurl, error_ol_badurl = call_llm(test_prompt_ol, model=ollama_test_model)
    if error_ol_badurl and ("Error de conexión" in error_ol_badurl or "RequestError" in error_ol_badurl):
         print(f"  -> Success! Received expected connection error: {error_ol_badurl}")
    else:
         print(f"  -> Failed! Expected connection error but got: {error_ol_badurl} / {content_ol_badurl}")
    # Restaurar variable de entorno
    if original_ollama_url is None:
         if OLLAMA_BASE_URL_ENV in os.environ: del os.environ[OLLAMA_BASE_URL_ENV]
    else:
         os.environ[OLLAMA_BASE_URL_ENV] = original_ollama_url

    print("\n--- LLM Client tests finished ---")