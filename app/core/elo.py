# core/elo.py
import math 

ELO_SCALE_FACTOR = 400.0
ELO_K_FACTOR_DEFAULT = 32 

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calcula la probabilidad esperada de que el jugador A gane contra el jugador B,
    basado en la fórmula estándar de Elo.
    """
    exponent = (rating_b - rating_a) / ELO_SCALE_FACTOR
    return 1.0 / (1.0 + math.pow(10, exponent))


def update_elo(
    rating_a: float,
    rating_b: float,
    result_a: float,
    k: int = ELO_K_FACTOR_DEFAULT
) -> tuple[float, float]:
    """
    Actualiza y devuelve los nuevos ratings Elo para dos jugadores (A y B)
    después de un enfrentamiento directo.

    Args:
        rating_a: Rating actual del jugador A.
        rating_b: Rating actual del jugador B.
        result_a: Resultado desde la perspectiva de A (1.0 victoria, 0.5 empate, 0.0 derrota).
        k: Factor K, controla cuánto cambian los ratings (por defecto 32).

    Returns:
        Una tupla conteniendo (nuevo_rating_a, nuevo_rating_b).
    """
    if result_a not in [0.0, 0.5, 1.0]:
        raise ValueError("El resultado 'result_a' debe ser 0.0, 0.5 o 1.0")

    expected_a = expected_score(rating_a, rating_b)
    change_a = k * (result_a - expected_a)
    change_b = -change_a

    new_rating_a = rating_a + change_a
    new_rating_b = rating_b + change_b

    return round(new_rating_a, 2), round(new_rating_b, 2)

# --- Tests rápidos (opcional, pero buena práctica ponerlos aquí o en un test_elo.py) ---
if __name__ == "__main__":
    r_a, r_b = 1200, 1200
    print(f"Initial Ratings: A={r_a}, B={r_b}")

    # A gana
    new_r_a, new_r_b = update_elo(r_a, r_b, 1.0)
    print(f"A wins: New Ratings: A={new_r_a}, B={new_r_b}") # Esperado: A=1216.0, B=1184.0

    # B gana
    new_r_a, new_r_b = update_elo(r_a, r_b, 0.0)
    print(f"B wins: New Ratings: A={new_r_a}, B={new_r_b}") # Esperado: A=1184.0, B=1216.0

    # Empate
    new_r_a, new_r_b = update_elo(r_a, r_b, 0.5)
    print(f"Draw:   New Ratings: A={new_r_a}, B={new_r_b}") # Esperado: A=1200.0, B=1200.0

    # Ratings diferentes, gana el favorito (A)
    r_a, r_b = 1300, 1100
    print(f"\nInitial Ratings: A={r_a}, B={r_b}")
    exp_a = expected_score(r_a, r_b)
    print(f"Expected score for A: {exp_a:.2f}")
    new_r_a, new_r_b = update_elo(r_a, r_b, 1.0)
    print(f"A wins: New Ratings: A={new_r_a}, B={new_r_b}")

    # Ratings diferentes, gana el underdog (B)
    new_r_a, new_r_b = update_elo(r_a, r_b, 0.0)
    print(f"B wins: New Ratings: A={new_r_a}, B={new_r_b}") # A baja mucho, B sube mucho