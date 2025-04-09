# core/problem_bank.py

import json
import random
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import ValidationError

# Importamos nuestro modelo Pydantic para validar los datos
from .models import Problem, TestCase

# --- Configuración ---
_current_file_dir = Path(__file__).resolve().parent
DEFAULT_PROBLEM_BASE_DIR = _current_file_dir / "data" / "problems"
PROBLEM_BASE_DIR = Path(os.getenv("PROBLEM_BASE_DIR", DEFAULT_PROBLEM_BASE_DIR))

# Nombres de archivo esperados dentro de cada carpeta de problema
METADATA_FILENAME = "metadata.json"
DESCRIPTION_FILENAME = "description.md"
TESTS_FILENAME = "tests.json"

# --- Logging ---
logger = logging.getLogger(__name__)
_problem_cache: Optional[List[Problem]] = None

def _load_single_problem(problem_dir: Path) -> Optional[Problem]:
    """Carga y valida un único problema desde su directorio."""
    problem_slug = problem_dir.name 
    metadata_path = problem_dir / METADATA_FILENAME
    description_path = problem_dir / DESCRIPTION_FILENAME
    tests_path = problem_dir / TESTS_FILENAME

    # Verificar existencia de archivos requeridos
    required_files = [metadata_path, description_path, tests_path]
    if not all(f.exists() and f.is_file() for f in required_files):
        logger.warning(f"Directorio de problema '{problem_slug}' incompleto. Faltan archivos requeridos. Saltando.")
        return None

    try:
        # Cargar metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            if not isinstance(metadata, dict):
                 logger.warning(f"Archivo '{metadata_path}' no contiene un objeto JSON. Saltando problema '{problem_slug}'.")
                 return None
            if metadata.get("slug") != problem_slug:
                 logger.warning(f"Inconsistencia en '{metadata_path}': slug '{metadata.get('slug')}' no coincide con el nombre del directorio '{problem_slug}'. Usando nombre del directorio.")
                 metadata["slug"] = problem_slug

        # Cargar descripción
        with open(description_path, "r", encoding="utf-8") as f:
            description = f.read()

        # Cargar tests
        with open(tests_path, "r", encoding="utf-8") as f:
            test_cases_data = json.load(f)
            if not isinstance(test_cases_data, list):
                logger.warning(f"Archivo '{tests_path}' no contiene una lista JSON de casos de prueba. Saltando problema '{problem_slug}'.")
                return None
        
        problem_data = {
            "slug": problem_slug,
            "title": metadata.get("title", f"Problema {problem_slug}"),
            "description": description,
            "test_cases": test_cases_data
        }

        # Validar con Pydantic
        problem = Problem(**problem_data)
        return problem

    except json.JSONDecodeError as e:
        logger.warning(f"Error decodificando JSON en el directorio '{problem_slug}'. Archivo: {e.doc}. Error: {e}. Saltando.")
        return None
    except ValidationError as e:
        logger.warning(f"Error de validación Pydantic para el problema '{problem_slug}'. Detalles: {e}. Saltando.")
        return None
    except IOError as e:
        logger.warning(f"Error de lectura de archivo en el directorio '{problem_slug}'. Error: {e}. Saltando.")
        return None
    except Exception as e:
        logger.error(f"Error inesperado procesando el problema '{problem_slug}': {e}", exc_info=True)
        return None


def load_problems(force_reload: bool = False) -> List[Problem]:
    """
    Escanea el directorio base, carga y valida todos los problemas encontrados.

    Utiliza un caché simple en memoria.

    Args:
        force_reload: Si es True, ignora el caché y recarga desde el sistema de archivos.

    Returns:
        Una lista de objetos `Problem` validados.
    """
    global _problem_cache
    if _problem_cache is not None and not force_reload:
        logger.debug("Devolviendo problemas desde caché.")
        return _problem_cache

    logger.info(f"Escaneando directorio de problemas: {PROBLEM_BASE_DIR}")
    validated_problems: List[Problem] = []

    if not PROBLEM_BASE_DIR.exists() or not PROBLEM_BASE_DIR.is_dir():
        logger.error(f"El directorio base de problemas '{PROBLEM_BASE_DIR}' no existe o no es un directorio.")
        _problem_cache = []
        return []

    # Iterar sobre los subdirectorios en el directorio base
    for item in PROBLEM_BASE_DIR.iterdir():
        if item.is_dir():
            problem = _load_single_problem(item)
            if problem:
                validated_problems.append(problem)

    logger.info(f"Cargados y validados {len(validated_problems)} problemas de {len(list(PROBLEM_BASE_DIR.glob('*')))} items encontrados.")
    _problem_cache = validated_problems
    return validated_problems


# --- Las funciones get_* no necesitan cambios lógicos ---

def get_problem(slug: str) -> Optional[Problem]:
    """Busca y devuelve un problema por su slug único."""
    problems = load_problems()
    for problem in problems:
        if problem.slug == slug:
            return problem
    logger.warning(f"Problema con slug '{slug}' no encontrado en el banco.")
    return None


def get_random_problem() -> Optional[Problem]:
    """Selecciona y devuelve un problema aleatorio del banco cargado."""
    problems = load_problems()
    if not problems:
        logger.warning("No hay problemas disponibles en el banco para seleccionar uno aleatorio.")
        return None
    return random.choice(problems)


def get_all_problems() -> List[Problem]:
    """Devuelve todos los problemas validados del banco."""
    return load_problems()


# --- Tests rápidos (adaptados a la nueva estructura) ---
if __name__ == "__main__":
    print("--- Testing Problem Bank (Folder Structure) ---")

    # Crear estructura de directorios temporal para probar
    test_base_dir = Path("data_test_problem_bank_folders")
    if test_base_dir.exists(): # Limpiar de ejecuciones anteriores
        import shutil
        shutil.rmtree(test_base_dir)
    test_base_dir.mkdir()

    # Problema 1: Válido
    prob1_slug = "sum-two"
    prob1_dir = test_base_dir / prob1_slug
    prob1_dir.mkdir()
    with open(prob1_dir / METADATA_FILENAME, "w") as f:
        json.dump({"slug": prob1_slug, "title": "Sumar Dos"}, f)
    with open(prob1_dir / DESCRIPTION_FILENAME, "w") as f:
        f.write("Suma dos números.")
    with open(prob1_dir / TESTS_FILENAME, "w") as f:
        json.dump([{"input": [1, 1], "expected_output": 2}], f)

    # Problema 2: Válido
    prob2_slug = "find-max"
    prob2_dir = test_base_dir / prob2_slug
    prob2_dir.mkdir()
    with open(prob2_dir / METADATA_FILENAME, "w") as f:
        json.dump({"slug": prob2_slug, "title": "Máximo"}, f)
    with open(prob2_dir / DESCRIPTION_FILENAME, "w") as f:
        f.write("Encuentra el máximo.")
    with open(prob2_dir / TESTS_FILENAME, "w") as f:
        json.dump([{"input": [[1, 5, 2]], "expected_output": 5}], f)

    # Problema 3: Inválido (falta description.md)
    prob3_slug = "incomplete"
    prob3_dir = test_base_dir / prob3_slug
    prob3_dir.mkdir()
    with open(prob3_dir / METADATA_FILENAME, "w") as f:
        json.dump({"slug": prob3_slug, "title": "Incompleto"}, f)
    with open(prob3_dir / TESTS_FILENAME, "w") as f:
        json.dump([], f)

    # Problema 4: Inválido (tests.json no es una lista)
    prob4_slug = "bad-tests"
    prob4_dir = test_base_dir / prob4_slug
    prob4_dir.mkdir()
    with open(prob4_dir / METADATA_FILENAME, "w") as f:
        json.dump({"slug": prob4_slug, "title": "Tests Malos"}, f)
    with open(prob4_dir / DESCRIPTION_FILENAME, "w") as f:
        f.write("Descripción ok.")
    with open(prob4_dir / TESTS_FILENAME, "w") as f:
        json.dump({"input": [1], "expected_output": 1}, f) # Objeto, no lista

    # Archivo extra (no es directorio, debe ignorarse)
    (test_base_dir / "some_other_file.txt").touch()

    # Forzar uso del directorio temporal
    original_path_env = os.environ.get("PROBLEM_BASE_DIR")
    os.environ["PROBLEM_BASE_DIR"] = str(test_base_dir.resolve())

    print(f"\nUsando directorio temporal: {test_base_dir.resolve()}")

    # Test: Cargar todos
    all_probs = get_all_problems() # Llama a load_problems internamente
    print(f"\nTest 1: Cargar Todos ({len(all_probs)} problemas validados)")
    assert len(all_probs) == 2 # Solo los dos válidos
    assert all(isinstance(p, Problem) for p in all_probs)
    assert {p.slug for p in all_probs} == {prob1_slug, prob2_slug}

    # Test: Obtener por slug existente
    prob_sum = get_problem(prob1_slug)
    print(f"\nTest 2: Obtener '{prob1_slug}': {'Encontrado' if prob_sum else 'No encontrado'}")
    assert prob_sum is not None
    assert prob_sum.title == "Sumar Dos"
    assert isinstance(prob_sum.test_cases[0], TestCase) # Pydantic valida TestCase anidados

    # Test: Obtener por slug inválido/no cargado
    prob_invalid = get_problem(prob3_slug)
    print(f"\nTest 3: Obtener '{prob3_slug}' (incompleto): {'Encontrado' if prob_invalid else 'No encontrado'}")
    assert prob_invalid is None

    # Test: Obtener aleatorio
    random_prob = get_random_problem()
    print(f"\nTest 4: Obtener aleatorio: {random_prob.title if random_prob else 'Ninguno'}")
    assert random_prob is not None
    assert random_prob.slug in [prob1_slug, prob2_slug]

    # Test: Forzar recarga
    print("\nTest 5: Forzar Recarga")
    # Modificar un archivo y recargar para ver si detecta el cambio (o error)
    with open(prob1_dir / METADATA_FILENAME, "w") as f: # Hacerlo inválido
        f.write("not json")
    reloaded_probs = load_problems(force_reload=True)
    print(f"  Recargado ({len(reloaded_probs)} problemas validados ahora)")
    assert len(reloaded_probs) == 1 # Solo prob2 debería cargar ahora
    assert reloaded_probs[0].slug == prob2_slug

    # Restaurar problema 1 para siguiente test
    with open(prob1_dir / METADATA_FILENAME, "w") as f:
        json.dump({"slug": prob1_slug, "title": "Sumar Dos"}, f)


    # Test: Directorio base no existe
    os.environ["PROBLEM_BASE_DIR"] = "non_existent_folder_abc"
    print("\nTest 6: Directorio base no encontrado")
    not_found_probs = load_problems(force_reload=True)
    assert not_found_probs == []

    # Limpiar
    if original_path_env is None:
        del os.environ["PROBLEM_BASE_DIR"]
    else:
        os.environ["PROBLEM_BASE_DIR"] = original_path_env
    import shutil
    shutil.rmtree(test_base_dir)
    print("\n--- Problem Bank (Folder Structure) tests finished ---")