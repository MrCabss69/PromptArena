# app/app.py 
import gradio as gr
import logging
import os
from typing import Dict, Tuple, Optional, List
import httpx 

# Importaciones del core actualizadas
from core.problem_bank import get_all_problems, get_problem, Problem
from core.runner import run_comparison, ComparisonResult, hash_prompt
from core.models import EloRating, EvaluationResult 

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ollama Model Fetching & Predefined Models (Paste the code from Step 1 here) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TAGS_ENDPOINT = "/api/tags"

def get_local_ollama_models() -> List[str]:
    models = []
    url = OLLAMA_BASE_URL.rstrip('/') + OLLAMA_TAGS_ENDPOINT
    try:
        logger.info(f"Intentando obtener modelos de Ollama desde: {url}")
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
        if "models" in data and isinstance(data["models"], list):
            models = [m.get("name") for m in data["models"] if m.get("name")]
            logger.info(f"Modelos Ollama encontrados: {models}")
        else:
            logger.warning(f"Respuesta inesperada de Ollama /api/tags: {str(data)[:200]}...")
    except httpx.HTTPStatusError as e:
         logger.error(f"Error HTTP al contactar Ollama ({url}): {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"No se pudo conectar a Ollama en {url}. ¿Está corriendo? Error: {e}")
    except Exception as e:
        logger.error(f"Error inesperado al obtener modelos de Ollama: {e}", exc_info=True)

    if not models:
        logger.warning("No se encontraron modelos Ollama o hubo un error al contactar el servidor.")
        return ["[Error al obtener modelos Ollama]"]
    return sorted(models)

PREDEFINED_OPENROUTER_MODELS = sorted([
    "openai/gpt-4o", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo",
    "anthropic/claude-3-opus", "anthropic/claude-3-sonnet", "anthropic/claude-3-haiku",
    "google/gemini-pro-1.5", "google/gemini-flash-1.5",
    "meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-8b-instruct",
    "mistralai/mistral-large", "mistralai/mistral-7b-instruct",
])

DEFAULT_PROVIDER = "OpenRouter"
DEFAULT_MODEL_MAP = {
    "OpenRouter": PREDEFINED_OPENROUTER_MODELS[0] if PREDEFINED_OPENROUTER_MODELS else "openai/gpt-4o",
    "Ollama": "llama3"
}

# --- Estado en Memoria (Simulación de Base de Datos) ---
PROMPT_RATINGS: Dict[str, EloRating] = {}
DEFAULT_RATING_VALUE = 1200.0

# --- Carga de Problemas (al inicio de la app) ---
try:
    PROBLEMS: list[Problem] = get_all_problems()
    if not PROBLEMS:
        logger.error("No se cargaron problemas. La aplicación puede no funcionar correctamente.")
        PROBLEM_CHOICES = ["¡Error al cargar problemas!"]
        PROBLEM_CHOICES_MAP = {}
    else:
        PROBLEM_CHOICES_MAP = {p.title: p.slug for p in PROBLEMS}
        PROBLEM_CHOICES = list(PROBLEM_CHOICES_MAP.keys())
except Exception as e:
    logger.exception("Error crítico al cargar problemas durante el inicio.")
    PROBLEMS = []
    PROBLEM_CHOICES = ["¡Error crítico al cargar!"]
    PROBLEM_CHOICES_MAP = {}

# --- Lógica del Flujo de la Aplicación ---

def get_current_ratings(prompt_id_a: str, prompt_id_b: str) -> Dict[str, float]:
    """Obtiene los ratings actuales del 'almacén' en memoria."""
    return {
        prompt_id_a: PROMPT_RATINGS.get(prompt_id_a, EloRating(prompt_id=prompt_id_a)).rating,
        prompt_id_b: PROMPT_RATINGS.get(prompt_id_b, EloRating(prompt_id=prompt_id_b)).rating,
    }

def update_ratings_in_memory(result: ComparisonResult):
    """Actualiza los ratings en nuestro 'almacén' en memoria."""
    prompt_id_a = result.prompt_result_a.prompt_id
    prompt_id_b = result.prompt_result_b.prompt_id
    rating_obj_a = PROMPT_RATINGS.get(prompt_id_a, EloRating(prompt_id=prompt_id_a))
    rating_obj_a.rating = result.new_rating_a
    rating_obj_a.matches_played += 1
    PROMPT_RATINGS[prompt_id_a] = rating_obj_a
    rating_obj_b = PROMPT_RATINGS.get(prompt_id_b, EloRating(prompt_id=prompt_id_b))
    rating_obj_b.rating = result.new_rating_b
    rating_obj_b.matches_played += 1
    PROMPT_RATINGS[prompt_id_b] = rating_obj_b
    logger.info(f"Ratings actualizados en memoria. Total prompts trackeados: {len(PROMPT_RATINGS)}")

# ... (format_evaluation - consider the improvement suggested before) ...
def format_evaluation(eval_obj: Optional[EvaluationResult]) -> str:
    """Formatea el resultado de la evaluación para mostrarlo en Markdown."""
    if not eval_obj:
        return "Evaluación no disponible."
    status = "✅ OK" if eval_obj.syntax_ok else "❌ Error Sintaxis/LLM"
    details = ""
    if eval_obj.error:
        if eval_obj.syntax_ok:
            status = "⚠️ Warning (Pyflakes)"
        details = f"\n```\n{eval_obj.error}\n```"
    return f"**Estado:** {status}{details}"


# --- Main Gradio flow function ---
def run_arena_flow(
    provider: str, # <-- ADDED
    model_name: str, # <-- ADDED
    prompt_a_text: str,
    prompt_b_text: str,
    selected_problem_title: str
) -> Tuple[str, str, str]:
    """
    Función principal que Gradio llamará.
    Orquesta la obtención del problema, la ejecución de la comparación y la actualización de ratings.
    """
    # --- Input Validation ---
    if not provider or not model_name:
         return "N/A", "N/A", "⚠️ Por favor, selecciona un proveedor y un modelo LLM."
    if model_name.startswith("[Error"):
        return "N/A", "N/A", f"⚠️ Error con el modelo seleccionado: {model_name}. Verifica la conexión con Ollama o elige otro modelo."
    if not prompt_a_text or not prompt_b_text:
        return "N/A", "N/A", "⚠️ Por favor, introduce ambos prompts."
    if not selected_problem_title or selected_problem_title.startswith("¡Error"):
        return "N/A", "N/A", f"⚠️ Por favor, selecciona un problema válido ({selected_problem_title})."

    try:
        # 1. Obtener el problema seleccionado
        problem_slug = PROBLEM_CHOICES_MAP.get(selected_problem_title)
        if not problem_slug:
             return "N/A", "N/A", f"❌ Error: Título de problema '{selected_problem_title}' no encontrado."
        problem = get_problem(problem_slug)
        if not problem:
            return "N/A", "N/A", f"❌ Error: Problema con slug '{problem_slug}' no encontrado en el banco."


        # 2. Obtener IDs y ratings actuales
        prompt_a_id = hash_prompt(prompt_a_text)
        prompt_b_id = hash_prompt(prompt_b_text)
        current_ratings = get_current_ratings(prompt_a_id, prompt_b_id)

        # 3. Ejecutar la comparación (pass model_name)
        logger.info(f"Ejecutando comparación para problema '{problem.slug}' usando modelo '{model_name}'...")
        result: ComparisonResult = run_comparison(
            prompt_a_text,
            prompt_b_text,
            problem,
            current_ratings,
            model_name=model_name
        )

        # 4. Actualizar ratings en memoria (simulación)
        update_ratings_in_memory(result)

        # 5. Formatear la salida para Gradio
        # ... (format output markdown - use format_evaluation(result.prompt_result_X.evaluation)) ...
        rating_a_change_str = f"{result.rating_change_a:+.1f}" if result.rating_change_a != 0 else ""
        rating_b_change_str = f"{result.rating_change_b:+.1f}" if result.rating_change_b != 0 else ""
        winner_text = { 'A': f"🏆 Prompt A ({result.prompt_result_a.prompt_id})", 'B': f"🏆 Prompt B ({result.prompt_result_b.prompt_id})", 'Draw': "🤝 Empate"}.get(result.winner, "Resultado desconocido")
        output_markdown = f"""
## 🧠 Resultado del Duelo ({problem.title} @ {model_name})

**Ganador:** {winner_text}

| Prompt                | ID ({'Partidas'}) | Rating Actual | Cambio   |
|-----------------------|-------------------|---------------|----------|
| **Prompt A**          | {result.prompt_result_a.prompt_id} ({PROMPT_RATINGS[prompt_a_id].matches_played}) | **{result.new_rating_a:.0f}** | `{rating_a_change_str}` |
| **Prompt B**          | {result.prompt_result_b.prompt_id} ({PROMPT_RATINGS[prompt_b_id].matches_played}) | **{result.new_rating_b:.0f}** | `{rating_b_change_str}` |

---
### ✅ Evaluación Estática Detallada

**Prompt A:**
{format_evaluation(result.prompt_result_a.evaluation)}

**Prompt B:**
{format_evaluation(result.prompt_result_b.evaluation)}
"""
        return (result.prompt_result_a.generated_code, result.prompt_result_b.generated_code, output_markdown.strip())


    except Exception as e:
        logger.exception("Error inesperado durante el flujo de la aplicación Gradio.")
        return "Error", "Error", f"❌ Ocurrió un error inesperado en el servidor: {e}"

# --- Gradio UI Update Function ---
def update_model_dropdown(provider: str) -> gr.Dropdown:
    """Actualiza las opciones del dropdown de modelos basado en el proveedor."""
    if provider == "Ollama":
        models = get_local_ollama_models()
        default_model = DEFAULT_MODEL_MAP["Ollama"]
        # Select default only if it exists, otherwise first in list or error message
        if default_model not in models and models and not models[0].startswith("[Error"):
            default_model = models[0]
        elif models and models[0].startswith("[Error"):
             default_model = models[0] # Show the error message as selected
        return gr.Dropdown(choices=models, value=default_model, label="Modelo Ollama Local", info="Modelos detectados en tu servidor Ollama local.")
    else: # Default to OpenRouter
        models = PREDEFINED_OPENROUTER_MODELS
        default_model = DEFAULT_MODEL_MAP["OpenRouter"]
        return gr.Dropdown(choices=models, value=default_model, label="Modelo OpenRouter", info="Modelos populares disponibles vía OpenRouter API.")


# --- Definición de la Interfaz Gradio ---
with gr.Blocks(title="Prompt Arena", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤺 Prompt Arena — Duelo de Prompts por Código")
    gr.Markdown("Introduce dos prompts, selecciona un proveedor LLM, un modelo y un problema. El sistema generará código, lo evaluará y actualizará ratings Elo.")

    with gr.Row():
        with gr.Column(scale=1):
             provider_radio = gr.Radio(
                 ["OpenRouter", "Ollama"],
                 value=DEFAULT_PROVIDER,
                 label="Proveedor LLM",
                 info="Elige el servicio o backend para generar el código."
             )
             model_dropdown = gr.Dropdown(
                 choices=PREDEFINED_OPENROUTER_MODELS, # Initial choices
                 value=DEFAULT_MODEL_MAP[DEFAULT_PROVIDER], # Initial value
                 label="Modelo LLM", # Initial label
                 info="Elige el modelo específico a usar.",
                 filterable=True # Allow searching models
             )
        with gr.Column(scale=2):
            problem_selector = gr.Dropdown(
                choices=PROBLEM_CHOICES,
                label="Selecciona un Problema de Programación",
                info="Elige el desafío para el que los prompts deben generar una solución.",
                filterable=True
            )

    with gr.Row():
        prompt_a = gr.Textbox(label="Texto del Prompt A", lines=5, placeholder="Ej: 'Usa un bucle for y un set para encontrar duplicados...'")
        prompt_b = gr.Textbox(label="Texto del Prompt B", lines=5, placeholder="Ej: 'Utiliza collections.Counter para contar elementos...'")

    run_button = gr.Button("⚔️ Iniciar Duelo de Prompts", variant="primary")

    with gr.Accordion("Resultados Detallados", open=True):
        result_output = gr.Markdown(value="Esperando resultados...")
        with gr.Row():
            code_output_a = gr.Code(label="Código Generado por Prompt A", language="python", interactive=False)
            code_output_b = gr.Code(label="Código Generado por Prompt B", language="python", interactive=False)

    # --- Event Listeners ---
    # 1. Update model dropdown when provider changes
    provider_radio.change(
        fn=update_model_dropdown,
        inputs=[provider_radio],
        outputs=[model_dropdown]
    )

    # 2. Run comparison when button is clicked
    run_button.click(
        fn=run_arena_flow,
        inputs=[provider_radio, model_dropdown, prompt_a, prompt_b, problem_selector],
        outputs=[code_output_a, code_output_b, result_output],
        api_name="run_duel"
    )

    gr.Markdown("--- \n*Desarrollado como MVP. Los ratings se guardan solo en memoria durante la sesión.*")

# --- Lanzar la aplicación ---
if __name__ == "__main__":
    logger.info("Iniciando la aplicación Gradio Prompt Arena...")
    demo.launch()