'''
/ai/rag/ragQuery.py
-> builds the exploration prompt from retrieved graph context and asks Ollama
'''

import os
import requests
from dotenv import load_dotenv

from ai.rag.retrieval import (
    retrieve_exploration_context,
    retrieve_enhanced_context,
)

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")


def answer_question_with_exploration(
    question, artist_relationships, name_to_artists,
    artist_proj, community_hierarchy, text_token_index, artist_texts,
):
    ctx = retrieve_exploration_context(
        question, artist_relationships, name_to_artists,
        artist_proj, community_hierarchy, text_token_index, artist_texts,
    )

    direct_context = '\n\n'.join(ctx['direct']) if ctx['direct'] else ''
    discovery_context = '\n\n'.join(ctx['discoveries']) if ctx['discoveries'] else ''
    related_context = '\n\n'.join(ctx['related']) if ctx['related'] else ''
    community_context = ctx.get('community_summary', '') or ''

    if not direct_context and not discovery_context:
        return "Nao encontrei informacao relevante no arquivo para responder a essa pergunta."

    prompt = f"""Es um especialista no arquivo da Musica Portuguesa a Gostar Dela Propria (MPAGDP), que documenta nao so musica mas tambem poesia, danca, artesanato, gastronomia, religiao, festas, historias de vida, paisagens sonoras, cultura cigana, medicina popular e muitas outras dimensoes da cultura popular portuguesa. Responde a pergunta do utilizador baseando-te APENAS no contexto fornecido.

A tua resposta DEVE seguir esta estrutura:

## 1. Resposta Direta
Usando o 'Contexto Direto', apresenta os factos mais relevantes sobre artistas, instrumentos, regioes e categorias relacionados com a pergunta. Se nao houver contexto direto, indica-o brevemente.

## 2. Explorar e Descobrir
Esta e a seccao MAIS IMPORTANTE da tua resposta. Deve ser DETALHADA e EXTENSA (varios paragrafos).
Usando o 'Contexto de Descobertas', procura ACTIVAMENTE ligacoes culturais que vao ALEM da musica e dos artistas em si. O arquivo contem projetos sobre muitos temas culturais alem da musica. Quando encontrares referencias a estes temas no contexto, desenvolve-as:
- Religiao, reza, oracoes, bencaos, procissoes, santos e romarias (ha 181+ projetos com referencias religiosas)
- Festas populares, festivais, bailes e celebracoes (437+ projetos festivos)
- Agricultura, vida rural, trabalho no campo (181+ projetos rurais)
- Natal, Janeiras, Reis e celebracoes sazonais (138+ projetos)
- Gastronomia, comida tradicional e receitas (80+ projetos)
- Mar, pesca, pescadores e vida costeira (61+ projetos)
- Medicina popular, curas e crencas (55+ projetos)
- Artesanato, oficios tradicionais, tecelagem e construcao de instrumentos (213+ projetos)
- Poesia, poetas populares, quadras, decimas, rimas e tradicao oral (1700+ projetos)
- Cultura cigana e romani (213+ projetos)
- Historia de vida, memorias, emigracao e transmissao da tradicao (629+ projetos)
- Paisagens sonoras e registos ambientais (133+ projetos)
- Danca, baile e coreografia tradicional (442+ projetos)
- Lendas, contos, historias orais e lenga-lengas
- Morte, luto, encomendacao das almas e rituais funebres
- Carnaval e mascaras tradicionais

Extrai e desenvolve TODAS estas referencias culturais que encontrares no contexto. Para cada uma, explica o que e, onde se situa e porque e interessante. Nao resumas — desenvolve cada ponto com detalhe. Apresenta-as como convites a explorar mais, mostrando como estes temas se cruzam entre si.

## 3. Continuar a Explorar
Sugere 3-4 perguntas de seguimento em portugues que o utilizador possa fazer para continuar a navegar o arquivo. Baseia-as tanto na resposta direta como nas descobertas culturais. Torna-as especificas e intrigantes — pelo menos metade deve ser sobre temas culturais presentes no contexto (religiao, festas, gastronomia, artesanato, vida rural, poesia, cultura cigana, medicina popular, etc.).

## 4. Obras Relacionadas
Usando o 'Contexto Relacionado', menciona brevemente outros artistas ou projetos ligados a este tema que o utilizador possa querer descobrir.

**Regras:**
- Se uma seccao nao tiver contexto, omite-a completamente.
- Escreve em portugues.
- Se engenhoso e desperta curiosidade.
- A seccao 'Explorar e Descobrir' deve ser a mais longa e detalhada de todas.
- NAO inventes informacao que nao esteja no contexto.
- NAO te limites a falar de musica — o arquivo contem poesia, danca, artesanato, gastronomia, religiao, festas, historias de vida, paisagens sonoras, cultura cigana, medicina popular e muito mais. Explora TODAS as dimensoes culturais presentes no contexto.

---
**Contexto Direto:**
{direct_context}

**Contexto de Descobertas:**
{discovery_context}

**Contexto de Comunidade:**
{community_context}

**Contexto Relacionado:**
{related_context}
---

**Pergunta:** {question}

**Resposta:**
"""

    if not OLLAMA_URL:
        return "Error: OLLAMA_URL is not set in the environment."

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': 'llama3.1:8b',
                'prompt': prompt,
                'stream': False,
                'keep_alive': -1,
            },
            timeout=120,
        )
        if response.status_code == 200:
            return response.json().get('response', 'No response content found.')
        return f"Error from Ollama API: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def answer_question_with_rag(question, artist_relationships, name_to_artists,
                             artist_proj, community_hierarchy=None):
    direct_context_blocks, related_context_blocks = retrieve_enhanced_context(
        question, artist_relationships, name_to_artists, artist_proj, community_hierarchy
    )

    if not direct_context_blocks:
        return "I couldn't find any relevant information in the graph to answer that question."

    direct_context = '\n\n'.join(direct_context_blocks)
    related_context = '\n\n'.join(related_context_blocks)

    prompt = f"""You are an expert assistant on Portuguese folk music. Your task is to answer the user's question based *only* on the context provided.

Your answer MUST be structured in two parts:
1. A direct answer to the question.
2. A section called "Related Discoveries" that highlights other artists or projects that are connected to the answer, sparking curiosity.

**Strict Rules:**
- Use ONLY the information from the 'Direct Answer Context' to formulate the direct answer.
- Use ONLY the information from the 'Related Discoveries Context' for the "Related Discoveries" section.
- If the "Related Discoveries Context" is empty, DO NOT include the "Related Discoveries" section.
- Write in a clear, engaging, and informative style.

---
**Direct Answer Context:**
{direct_context}

**Related Discoveries Context:**
{related_context}
---

**Question:** {question}

**Answer:**
"""

    if not OLLAMA_URL:
        return "Error: OLLAMA_URL is not set in the environment."

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': 'llama3.1:8b',
                'prompt': prompt,
                'stream': False,
                'keep_alive': -1,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get('response', 'No response content found.')
        return f"Error from Ollama API: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def answerExplorationQuestion(question):
    """
    Public entry point used by the Flask route: pulls the cached graph state
    built by ai.rag.setup.initRag/rebuildRag and answers the question.
    """
    from ai.rag.setup import getRagState

    state = getRagState()
    return answer_question_with_exploration(
        question,
        state.artist_relationships,
        state.name_to_artists,
        state.artist_proj,
        state.community_hierarchy,
        state.text_token_index,
        state.artist_texts,
    )
