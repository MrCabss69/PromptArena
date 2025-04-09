# core/evaluator.py

import ast
from pyflakes.api import check
from pyflakes.reporter import Reporter
from io import StringIO
from .models import EvaluationResult 


VIRTUAL_FILENAME = "<prompt_code>"


def validate_code_static(code: str) -> EvaluationResult:
    """
    Realiza una validación estática del código Python proporcionado.

    Utiliza un enfoque en dos pasos:
    1.  **Parsing AST (Abstract Syntax Tree):** Verifica que el código sea
        sintácticamente válido según la gramática de Python. Es el chequeo
        más fundamental.
    2.  **Análisis con Pyflakes:** Si la sintaxis es válida, utiliza Pyflakes
        para detectar errores comunes y código no utilizado (linting ligero).

    Args:
        code: El string de código Python a validar.

    Returns:
        Un objeto EvaluationResult con:
        - syntax_ok: True si el parsing AST fue exitoso, False en caso contrario.
        - error: None si no hubo errores de sintaxis ni de Pyflakes.
                 Contiene el mensaje de error si alguno ocurrió.
                 Prioriza el error de sintaxis si existe.
    """
    result = EvaluationResult()
    
    try:
        ast.parse(code, filename=VIRTUAL_FILENAME)
        result.syntax_ok = True
        # Si llegamos aquí, la sintaxis básica es correcta. Procedemos a Pyflakes.

    except SyntaxError as e:
        # Error de sintaxis detectado por ast.parse
        result.syntax_ok = False
        result.error = f"SyntaxError: {e.msg} (line {e.lineno}, offset {e.offset})"
        # No continuamos con Pyflakes si la sintaxis ya es inválida
        return result

    except Exception as e:
        # Captura otros errores inesperados durante el parsing (menos común)
        result.syntax_ok = False
        result.error = f"Unexpected AST Parsing Error: {type(e).__name__}: {e}"
        return result
    
    pyflakes_stdout = StringIO()
    reporter = Reporter(stdout=pyflakes_stdout,stderr=pyflakes_stdout)

    try:
        # Ejecutamos Pyflakes sobre el código
        error_count = check(code, filename=VIRTUAL_FILENAME, reporter=reporter)

        if error_count > 0:
            result.error = pyflakes_stdout.getvalue().strip()
        else:
            result.error = None

    except Exception as e:
        result.error = f"Unexpected Pyflakes Error: {type(e).__name__}: {e}"
        
    return result

# --- Tests rápidos ---
if __name__ == "__main__":
    print("--- Testing evaluator ---")

    # Caso 1: Código válido
    valid_code = "import os\n\ndef main():\n    print(os.name)\n\nmain()"
    res_valid = validate_code_static(valid_code)
    print(f"\nValid Code:\n{valid_code}\nResult: {res_valid.dict()}")
    assert res_valid.syntax_ok is True
    assert res_valid.error is None

    # Caso 2: Error de sintaxis
    syntax_error_code = "def func(\n    print('hello')"
    res_syntax_error = validate_code_static(syntax_error_code)
    print(f"\nSyntax Error Code:\n{syntax_error_code}\nResult: {res_syntax_error.dict()}")
    assert res_syntax_error.syntax_ok is False
    assert "SyntaxError" in res_syntax_error.error

    # Caso 3: Sintaxis válida, pero error Pyflakes (import no usado)
    pyflakes_error_code = "import sys\n\ndef greet(name):\n    print(f'Hello {name}')\n\ngreet('World')"
    res_pyflakes_error = validate_code_static(pyflakes_error_code)
    print(f"\nPyflakes Error Code:\n{pyflakes_error_code}\nResult: {res_pyflakes_error.dict()}")
    assert res_pyflakes_error.syntax_ok is True
    assert "'sys' imported but unused" in res_pyflakes_error.error

    # Caso 4: Código vacío
    empty_code = ""
    res_empty = validate_code_static(empty_code)
    print(f"\nEmpty Code:\nResult: {res_empty.dict()}")
    assert res_empty.syntax_ok is True # AST parsea OK
    assert res_empty.error is None    # Pyflakes no reporta nada

    # Caso 5: Código con solo comentarios/espacios
    comment_code = "# Esto es un comentario\n   \n"
    res_comment = validate_code_static(comment_code)
    print(f"\nComment Code:\nResult: {res_comment.dict()}")
    assert res_comment.syntax_ok is True
    assert res_comment.error is None

    print("\n--- Evaluator tests passed ---")