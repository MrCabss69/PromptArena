# core/runner.py (CORREGIDO)

import hashlib
import logging
from typing import Dict, Optional, Any

# Importaciones de nuestros módulos y modelos actualizados
from .llm_client import call_llm
from .evaluator import validate_code_static
from .elo import update_elo
from .models import Problem, EvaluationResult, PromptResult, EloRating, ComparisonResult

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constantes ---
DEFAULT_ELO_RATING = 1200.0
LLM_ERROR_CODE = "[LLM_ERROR]"


def hash_prompt(prompt_text: str) -> str:
    """Genera un hash MD5 corto (8 caracteres) para un texto de prompt."""
    return hashlib.md5(prompt_text.encode('utf-8')).hexdigest()[:8]

def _determine_duel_result(score_a: float, score_b: float) -> float:
    """Determina el resultado del duelo para el jugador A basado en los scores."""
    if score_a > score_b: return 1.0
    elif score_b > score_a: return 0.0
    else: return 0.5

def _calculate_score(evaluation: EvaluationResult) -> float:
    """Calcula un score numérico simple basado en la evaluación estática."""
    if not evaluation.syntax_ok: return 0.0
    elif evaluation.error: return 0.5
    else: return 1.0

# --- Lógica Principal de Comparación ---

def run_comparison(
    prompt_a_text: str,
    prompt_b_text: str,
    problem: Problem,
    current_ratings: Dict[str, float],
    model_name: str
) -> ComparisonResult:
    """
    Orquesta la comparación completa entre dos prompts para un problema dado.
    """
    # 1. Definir IDs y obtener ratings iniciales
    prompt_a_id = hash_prompt(prompt_a_text)
    prompt_b_id = hash_prompt(prompt_b_text)
    logger.info(f"Iniciando comparación con modelo '{model_name}': Prompt A ({prompt_a_id}) vs Prompt B ({prompt_b_id}) para problema '{problem.slug}'")

    rating_a = current_ratings.get(prompt_a_id, DEFAULT_ELO_RATING)
    rating_b = current_ratings.get(prompt_b_id, DEFAULT_ELO_RATING)

    # 2. Construcción del Prompt Completo para el LLM
    try:
        test_case_example = problem.test_cases[0]
        input_repr = repr(test_case_example.input)
        output_repr = repr(test_case_example.expected_output)
        example_str = f"\n\nEjemplo:\nInput: {input_repr}\nOutput esperado: {output_repr}"
    except IndexError:
        example_str = ""

    base_prompt = (
        f"Resuelve el siguiente problema de programación en Python:\n\n"
        f"**Problema: {problem.title}**\n"
        f"{problem.description}"
        f"{example_str}\n\n"
        f"**Instrucción específica:**\n{'{prompt_instruction}'}\n\n" # Marcador
        f"Escribe únicamente el código Python completo y funcional."
    )

    prompt_a_full = base_prompt.replace('{prompt_instruction}', prompt_a_text)
    prompt_b_full = base_prompt.replace('{prompt_instruction}', prompt_b_text)

    # 3. Llamada al LLM (UNA SOLA VEZ por prompt, pasando el modelo)
    code_a, error_a = call_llm(prompt_a_full, model=model_name)
    code_b, error_b = call_llm(prompt_b_full, model=model_name)

    # 4. Evaluación Estática (verificando errores LLM PRIMERO)
    if error_a:
        logger.warning(f"Fallo en LLM para Prompt A ({prompt_a_id}): {error_a}") 
        eval_a = EvaluationResult(syntax_ok=False, error=f"LLM Error: {error_a}")
        code_a = LLM_ERROR_CODE
    else:
        # Solo evaluar si no hubo error de LLM
        eval_a = validate_code_static(code_a)

    if error_b:
        logger.warning(f"Fallo en LLM para Prompt B ({prompt_b_id}): {error_b}") 
        eval_b = EvaluationResult(syntax_ok=False, error=f"LLM Error: {error_b}")
        code_b = LLM_ERROR_CODE
    else:
        # Solo evaluar si no hubo error de LLM
        eval_b = validate_code_static(code_b)

    # 5. Crear objetos PromptResult (después de tener code_a/b y eval_a/b)
    prompt_result_a = PromptResult(
        prompt_text=prompt_a_text,
        prompt_id=prompt_a_id,
        generated_code=code_a,
        evaluation=eval_a
    )
    prompt_result_b = PromptResult(
        prompt_text=prompt_b_text,
        prompt_id=prompt_b_id,
        generated_code=code_b,
        evaluation=eval_b
    )

    # 6. Comparación y Actualización ELO
    score_a = _calculate_score(eval_a)
    score_b = _calculate_score(eval_b)
    logger.info(f"Scores calculados: A={score_a}, B={score_b}")

    duel_result_for_a = _determine_duel_result(score_a, score_b)

    new_rating_a, new_rating_b = update_elo(rating_a, rating_b, result_a=duel_result_for_a)
    logger.info(f"Ratings actualizados: A ({prompt_a_id}): {rating_a:.0f} -> {new_rating_a:.0f}, B ({prompt_b_id}): {rating_b:.0f} -> {new_rating_b:.0f}")

    # Determinar ganador textual
    if duel_result_for_a == 1.0: winner = 'A'
    elif duel_result_for_a == 0.0: winner = 'B'
    else: winner = 'Draw'

    # 7. Crear y Devolver Resultado Completo
    comparison_outcome = ComparisonResult(
        prompt_result_a=prompt_result_a,
        prompt_result_b=prompt_result_b,
        problem_slug=problem.slug,
        new_rating_a=new_rating_a,
        new_rating_b=new_rating_b,
        rating_change_a=round(new_rating_a - rating_a, 2),
        rating_change_b=round(new_rating_b - rating_b, 2),
        winner=winner
    )

    return comparison_outcome