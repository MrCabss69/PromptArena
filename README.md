# Prompt Arena 🤺 

Un sistema interactivo para comparar la efectividad de diferentes prompts de ingeniería aplicados a la resolución de problemas de código utilizando modelos de lenguaje grandes (LLMs). Permite enfrentar dos prompts, generar código para un problema específico usando OpenRouter o Ollama, realizar una evaluación estática del código y actualizar un ranking relativo basado en el sistema Elo.

## Características Principales

*   **Comparación de Prompts:** Introduce dos prompts diferentes para resolver el mismo problema.
*   **Selección de Problema:** Elige entre una lista de problemas de programación cargados desde una estructura de directorios.
*   **Selección de LLM:**
    *   Utiliza modelos populares a través de la API de **OpenRouter**.
    *   Conecta con un servidor **Ollama** local para usar modelos como Llama 3, Mistral, etc.
*   **Generación de Código:** El LLM seleccionado genera código Python basado en cada prompt y la descripción del problema.
*   **Evaluación Estática:** El código generado se valida usando `ast.parse` (sintaxis) y `pyflakes` (errores comunes/linting).
*   **Sistema de Ranking Elo:** Los prompts (identificados por su hash) compiten en duelos. Sus ratings Elo se actualizan según el resultado de la evaluación estática.
*   **Interfaz Web:** Interfaz de usuario sencilla e interactiva construida con **Gradio**.
*   **Extensible:** Fácil añadir nuevos problemas creando una nueva carpeta en `app/core/data/problems/`.

## 🛠️ Tech Stack

*   **Lenguaje:** Python 3.9+
*   **UI:** Gradio
*   **Modelos de Datos:** Pydantic
*   **Cliente HTTP:** httpx
*   **Evaluación de Código:** ast, pyflakes
*   **LLM Backends:** OpenRouter API, Ollama API

## ⚙️ Configuración e Instalación

1.  **Clonar el Repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd VCODEBENCH
    ```

2.  **Crear Entorno Virtual:** (Recomendado)
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Instalar Dependencias:**
    *   Primero, asegúrate de tener un archivo `requirements.txt`. Si no existe, créalo desde tu entorno virtual activo donde ya instalaste las librerías:
        ```bash
        pip freeze > requirements.txt
        ```
    *   Luego, instala las dependencias:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Configurar Variables de Entorno:** Necesitas configurar claves de API y opcionalmente URLs. La forma recomendada es crear un archivo `.env` en la raíz (`VCODEBENCH/`) y añadirlo a tu `.gitignore`.

    *   **Crea un archivo `.env`** con el siguiente contenido:
        ```dotenv
        # Obligatorio para usar OpenRouter
        OPENROUTER_API_KEY="tu_clave_api_de_openrouter"

        # Opcional: Si tu servidor Ollama no está en http://localhost:11434
        # OLLAMA_BASE_URL="http://tu_ip_o_host:11434"

        # Opcional: Para identificación en OpenRouter (puedes dejar los defaults)
        # PROMPT_ARENA_REFERER="https://tu_identificador_web.com"
        # PROMPT_ARENA_TITLE="MiPromptArenaCustom"
        ```
    *   **IMPORTANTE:** Asegúrate de que el archivo `.env` esté listado en tu `.gitignore` para no subir tus claves secretas a Git.
    *   **Alternativa:** Puedes exportar las variables directamente en tu terminal antes de ejecutar la aplicación:
        ```bash
        export OPENROUTER_API_KEY="tu_clave_api_de_openrouter"
        # export OLLAMA_BASE_URL="..." # Si es necesario
        ```

5.  **(Opcional) Configurar Ollama:** Si planeas usar Ollama:
    *   Asegúrate de tener [Ollama](https://ollama.com/) instalado y corriendo.
    *   Descarga los modelos que quieras usar, por ejemplo: `ollama pull llama3`. La interfaz intentará listar los modelos disponibles en tu servidor Ollama local.

## ▶️ Ejecución

1.  Asegúrate de que tu entorno virtual esté activado.
2.  Asegúrate de que las variables de entorno estén configuradas (ya sea exportadas o mediante un archivo `.env` si usas una librería como `python-dotenv` - nota: el código actual no carga `.env` automáticamente, tendrías que añadir `load_dotenv()` al inicio de `app.py`).
3.  Navega al directorio raíz del proyecto (`VCODEBENCH/`).
4.  Ejecuta la aplicación Gradio:
    ```bash
    python app/app.py
    ```
5.  Abre tu navegador y ve a la URL local que Gradio indique (normalmente `http://127.0.0.1:7860`).

## 📁 Estructura del Proyecto

```
VCODEBENCH/
├── app/
│   ├── app.py            # Lógica de la interfaz Gradio y flujo principal UI
│   └── core/
│       ├── data/
│       │   └── problems/ # Directorio base para los problemas
│       │       ├── <problem_slug>/ # Carpeta para cada problema
│       │       │   ├── metadata.json     # Título, slug
│       │       │   ├── description.md    # Enunciado del problema
│       │       │   └── tests.json        # Casos de prueba
│       │       └── ...
│       ├── elo.py            # Lógica de cálculo de rating Elo
│       ├── evaluator.py      # Validación estática de código (AST, Pyflakes)
│       ├── llm_client.py     # Cliente para interactuar con OpenRouter y Ollama
│       ├── models.py         # Modelos de datos Pydantic
│       ├── problem_bank.py   # Carga y gestión de problemas desde archivos
│       └── runner.py         # Orquesta la comparación (LLM -> Eval -> Elo)
├── .env.example          # (Recomendado) Ejemplo de variables de entorno
├── .gitignore            # Archivos a ignorar por Git
├── README.md             # Este archivo
└── requirements.txt      # Dependencias de Python
```

## 📝 Añadir Nuevos Problemas

1.  Crea una nueva carpeta dentro de `app/core/data/problems/`. El nombre de la carpeta será el `slug` del problema (ej. `reverse-string`).
2.  Dentro de la nueva carpeta, crea los siguientes archivos:
    *   **`metadata.json`**:
        ```json
        {
          "slug": "reverse-string",
          "title": "Invertir Cadena"
        }
        ```
    *   **`description.md`**: Escribe el enunciado del problema usando Markdown.
    *   **`tests.json`**: Una lista de casos de prueba en formato JSON, donde cada caso es un objeto con `"input"` (una lista de argumentos) y `"expected_output"`:
        ```json
        [
          {"input": ["hello"], "expected_output": "olleh"},
          {"input": [""], "expected_output": ""}
        ]
        ```
3.  La próxima vez que inicies la aplicación, el nuevo problema debería aparecer en el desplegable.

## 🚀 Futuras Mejoras (Ideas)

*   **Evaluación Funcional:** Ejecutar el código generado contra los `tests.json` en un entorno seguro (sandbox/Docker).
*   **Persistencia de Ratings:** Guardar los ratings Elo en una base de datos (SQLite, JSON persistente) en lugar de solo en memoria.
*   **Leaderboard de Prompts:** Mostrar un ranking de los prompts con mejor Elo.
*   **Historial de Duelos:** Guardar y mostrar los resultados de comparaciones anteriores.
*   **Más Métricas de Evaluación:** Añadir complejidad ciclomática, longitud del código, etc.
*   **Interfaz Mejorada:** Más opciones de filtrado, visualizaciones.# PromptArena
# PromptArena
