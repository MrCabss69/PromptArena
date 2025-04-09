# core/models.py

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# --- Modelos para Problemas y Tests ---
class TestCase(BaseModel):
    """Define un único caso de prueba para un problema de código."""
    input: List[Any] = Field(..., description="Lista de argumentos para la función a probar.")
    expected_output: Any = Field(..., description="El resultado esperado al llamar la función con 'input'.")

    class Config:
        json_schema_extra = {
            "example": {
                "input": [1, 2, 3, 1],
                "expected_output": [1]
            }
        }


class Problem(BaseModel):
    """Representa un problema de programación completo."""
    slug: str = Field(..., description="Identificador único corto para el problema.")
    title: str = Field(..., description="Título legible por humanos del problema.")
    description: str = Field(..., description="Descripción completa del problema a resolver.")
    test_cases: List[TestCase] = Field(..., description="Lista de casos de prueba para validar la solución.")

    class Config:
        json_schema_extra = {
            "example": {
                "slug": "find-duplicates",
                "title": "Encontrar Duplicados",
                "description": "Dada una lista de enteros, devuelve una lista con los elementos que aparecen más de una vez.",
                "test_cases": [
                    {"input": [[1, 2, 3, 1]], "expected_output": [1]},
                    {"input": [[4, 5, 6]], "expected_output": []},
                    {"input": [[7, 7, 7, 7]], "expected_output": [7]}
                ]
            }
        }

# --- Modelos para Evaluación y Resultados de Prompts Individuales ---
class EvaluationResult(BaseModel):
    """Almacena el resultado de la evaluación (estática por ahora) de un fragmento de código."""
    syntax_ok: Optional[bool] = Field(None, description="Indica si el código pasó la validación de sintaxis (AST parsing).")
    error: Optional[str] = Field(None, description="Mensaje de error de sintaxis (AST) o de análisis estático (Pyflakes). Null si no hay errores.")
    # Campos previstos para evaluación funcional (a futuro)
    passed: Optional[int] = Field(None, description="Número de test cases superados (futuro).")
    failed: Optional[int] = Field(None, description="Número de test cases fallidos (futuro).")

    class Config:
        json_schema_extra = {
            "example_syntax_ok": {"syntax_ok": True, "error": None},
            "example_syntax_error": {"syntax_ok": False, "error": "SyntaxError: invalid syntax (<unknown>, line 1)"},
            "example_pyflakes_error": {"syntax_ok": True, "error": "<prompt_code>:1:1 'os' imported but unused"}
        }


class PromptResult(BaseModel):
    """Contiene el resultado completo de ejecutar un prompt para un problema."""
    prompt_text: str = Field(..., description="El texto completo del prompt utilizado.")
    prompt_id: str = Field(..., description="Identificador único del prompt (ej. hash).")
    generated_code: str = Field(..., description="El código generado por el LLM o un código de error.")
    evaluation: EvaluationResult = Field(..., description="El resultado de la evaluación del código generado.")

    class Config:
        # (El ejemplo necesitaría datos válidos de EvaluationResult)
        pass


# --- Modelo para Ratings ---
class EloRating(BaseModel):
    """Almacena el rating Elo asociado a un prompt específico."""
    prompt_id: str = Field(..., description="Identificador único del prompt (ej. hash del texto del prompt).")
    rating: float = Field(default=1200.0, description="El rating Elo actual del prompt.")
    matches_played: int = Field(default=0, description="Número de duelos jugados por este prompt.")

    class Config:
        json_schema_extra = {
            "example": {"prompt_id": "a1b2c3d4", "rating": 1250.5, "matches_played": 15}
        }

class ComparisonResult(BaseModel):
    """Agrupa toda la información relevante resultante de un duelo entre dos prompts."""
    prompt_result_a: PromptResult = Field(..., description="Resultado detallado del Prompt A.")
    prompt_result_b: PromptResult = Field(..., description="Resultado detallado del Prompt B.")
    problem_slug: str = Field(..., description="Slug del problema utilizado en el duelo.")
    new_rating_a: float = Field(..., description="Nuevo rating Elo del Prompt A después del duelo.")
    new_rating_b: float = Field(..., description="Nuevo rating Elo del Prompt B después del duelo.")
    rating_change_a: float = Field(..., description="Cambio neto en el rating del Prompt A.")
    rating_change_b: float = Field(..., description="Cambio neto en el rating del Prompt B.")
    winner: str = Field(..., description="Indica el ganador del duelo ('A', 'B', o 'Draw').")
    error_message: Optional[str] = Field(None, description="Mensaje de error si ocurrió un problema general durante la comparación.")

    class Config:
        # (El ejemplo necesitaría datos válidos de PromptResult y otros campos)
        pass