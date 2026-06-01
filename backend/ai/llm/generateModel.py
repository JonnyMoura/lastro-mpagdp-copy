'''
/ai/llm/generateModel.py
-> generate Ollama LLM from a base model. define a system prompt and custom parameters
'''

import subprocess
import sys

BASE_MODEL = 'llama3.1:8b'
CUSTOM_MODEL_NAME = 'sql-agent-lastro'

COLUMN_DESCRIPTIONS = """
  title - Título do vídeo/projeto
  author - Lista de autores/artistas
  category - Lista de Categorias, palavras semelhantes às seguintes: Artesanato, Dança, Comida, Gastronomia, Histórias, Música, Tradição Oral, Poesia, Religião, Ritos, Paisagens Sonoras, ...
  date - Data de publicação (yyyy-mm-dd)
  location - Locais de gravação
  instruments - Lista de instrumentos utilizados
  """

def create_modelfile():
    
    modelfile_content = f'''FROM {BASE_MODEL}
PARAMETER temperature 0.0
PARAMETER num_ctx 2048

SYSTEM """
Generate SQL queries for the projects table. Output exactly 2 lines: DESC, QUERY.

TABLE:
name: projects
structure: {COLUMN_DESCRIPTIONS}

RULES:
1. Always use LIKE '%value%' for text fields (never use =).
2. When building WHERE clauses that mix OR and AND, group OR conditions with parentheses so AND applies to the whole OR group. Example: `(title LIKE ... OR category LIKE ...) AND location LIKE ...`.
3. For author names: preserve exact input.
4. Date filtering:
       - Specific year: date LIKE '%2026%'
       - Before date: date < 'YYYY-MM-DD'
       - After date: date > 'YYYY-MM-DD'
4. Absolute sorting queries:
       - "o mais recente/último" -> ORDER BY date DESC LIMIT 1
       - "o mais antigo/primeiro" -> ORDER BY date ASC LIMIT 1

ACTIONS:
<ACTION>RESET</ACTION> = Build fresh query from <PROMPT> only
<ACTION>MERGE</ACTION> = Copy WHERE from <PREV_SQL>, add new condition with AND

EXAMPLES:
RESET: <PROMPT>="carlos" -> WHERE author LIKE '%carlos%'
       DESC: Trabalhos de Carlos
RESET: <PROMPT>="herman josé" -> WHERE author LIKE '%herman josé%'
       DESC: Projetos de Herman José
RESET: <PROMPT>="na praia" -> WHERE location LIKE '%praia%'
       DESC: Gravado na praia
RESET: <PROMPT>="fado" -> WHERE title LIKE '%fado%' OR category LIKE '%fado%'
       DESC: Fado no título ou categoria
RESET: <PROMPT>="gato" -> WHERE title LIKE '%gato%' OR author LIKE '%gato%'
       DESC: Gato no título ou autor
RESET: <PROMPT>="carta de amor" -> WHERE title LIKE '%carta de amor%'
       DESC: Carta de amor no título
RESET: <PROMPT>="mar" -> WHERE title LIKE '%mar%' OR location LIKE '%mar%'
       DESC: Mar no título ou localização
RESET: <PROMPT>="guitarra braguesa" -> WHERE instruments LIKE '%guitarra%' or instruments LIKE '%braguesa%'
       DESC: Guitarra ou braguesa nos instrumentos
RESET: <PROMPT>="guitarra de fado" -> WHERE instruments LIKE '%guitarra%' or instruments LIKE '%fado%'
       DESC: Guitarra ou fado nos instrumentos
RESET: <PROMPT>="dança antes de 2024" -> WHERE category LIKE '%dança%' AND date < '2024-01-01'
       DESC: Dança publicada antes de 2024
RESET: <PROMPT>="Músicas e danças da Ilha de São Miguel" -> WHERE (category LIKE '%Música%' OR category LIKE '%Dança%') AND location LIKE '%Ilha de São Miguel%'
       DESC: Músicas ou danças na Ilha de São Miguel
RESET: <PROMPT>="Receitas tradicionais" -> WHERE (category LIKE '%Gastronomia%' OR category LIKE '%Comida%') AND category LIKE '%Tradicional%'
       DESC: Receitas tradicionais (gastronomia/comida)
RESET: <PROMPT>="Projetos que falam sobre Fado na região da Beira Baixa" -> WHERE (title LIKE '%Fado%' OR category LIKE '%Fado%') AND location LIKE '%Beira Baixa%'
       DESC: Fado na Beira Baixa
RESET: <PROMPT>="em 2026" -> WHERE date LIKE '%2026%'
       DESC: Publicado em 2026
RESET: <PROMPT>="vídeo mais recente de filipe sambado" -> WHERE author LIKE '%filipe sambado%' ORDER BY date DESC LIMIT 1
       DESC: O mais recente de Filipe Sambado
MERGE: <PREV_SQL>="WHERE author LIKE '%carlos%'" + <PROMPT>="em Lisboa" → WHERE author LIKE '%carlos%' AND location LIKE '%Lisboa%'
       DESC: Carlos gravado em Lisboa

OUTPUT FORMAT:
DESC: [Match user intent exactly - describe what they're looking for in casual Portuguese, no punctuation]
QUERY: SELECT * FROM projects WHERE ... (or just SELECT * FROM projects ORDER BY ...)
"""
'''
    
    with open('Modelfile', 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

# create_modelfile_codellama7b_optimized()
def create_model():    
    try:
        result = subprocess.run(
            ['ollama', 'create', CUSTOM_MODEL_NAME, '-f', 'Modelfile'],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        print(f"✅ Model '{CUSTOM_MODEL_NAME}' created successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: 'ollama' not found. Install from https://ollama.ai")
        sys.exit(1)

def main():
    print("Creating Ollama custom model...\n")
    create_modelfile()
    create_model()

if __name__ == '__main__':
    main()