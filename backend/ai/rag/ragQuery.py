'''
/ai/rag/ragQuery.py
-> builds the exploration prompt from retrieved graph context and asks Claude
'''

import anthropic
from dotenv import load_dotenv

from ai.rag.retrieval import retrieve_exploration_context

load_dotenv()
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
MODEL_NAME = 'claude-sonnet-5'


def build_exploration_prompt(direct_context, discovery_context, community_context, question):
    return f"""Es um especialista no arquivo da Musica Portuguesa a Gostar Dela Propria (MPAGDP), que documenta nao so musica mas tambem poesia, danca, artesanato, gastronomia, religiao, festas, historias de vida, paisagens sonoras, cultura cigana, medicina popular e muitas outras dimensoes da cultura popular portuguesa. Responde a pergunta do utilizador baseando-te APENAS no contexto fornecido.

A tua resposta e composta por tres momentos que se seguem em fluxo continuo, apenas em paragrafos de prosa — SEM titulos, SEM cabecalhos markdown (nada de "#", "##"), SEM numeracao de seccoes ("1.", "2.", "3.") a introduzir cada parte. O leitor nao deve ver nenhuma etiqueta a anunciar onde uma parte acaba e a seguinte comeca — apenas a transicao natural do texto.

Primeiro, usando o 'Contexto Direto', apresenta os factos mais relevantes sobre artistas, instrumentos, regioes e categorias relacionados com a pergunta. Sempre que o contexto incluir o link de um video especifico (secao "Videos (titulo -> link)"), refere o titulo exacto e inclui o link no formato [Titulo exacto](link), para que o utilizador possa encontrar esse video. Se nao houver contexto direto, indica-o brevemente.

Depois, o corpo MAIS IMPORTANTE da tua resposta. Deve ser LONGO e MUITO DETALHADO — no minimo 6 a 8 paragrafos densos — escrito em PROSA FLUIDA E CONECTADA — como um ensaio bem escrito ou uma reportagem, nunca como uma lista ou catalogo de topicos.

Usando o 'Contexto de Descobertas', procura ACTIVAMENTE ligacoes culturais que vao ALEM da musica e dos artistas em si. O arquivo documenta muitas dimensoes da cultura popular portuguesa para alem da musica: religiao e devocao popular (rezas, bencaos, procissoes, santos, romarias), festas e celebracoes comunitarias, agricultura e vida rural, o ciclo do Natal e das festas sazonais, gastronomia tradicional, a vida costeira e a pesca, medicina popular, artesanato e oficios tradicionais (incluindo a construcao de instrumentos), poesia e tradicao oral, cultura cigana e romani, historias de vida e emigracao, paisagens sonoras, danca e coreografia, lendas e contos, rituais funebres, e carnaval.

Cobre o maior numero possivel de ligacoes DISTINTAS e DIFERENTES ENTRE SI presentes no contexto — nao te limites a desenvolver apenas 2 ou 3 descobertas em grande detalhe; percorre tantas quantas o contexto permitir. Sempre que existir no 'Contexto de Descobertas' um bloco marcado "(categoria fora da musica: ...)", inclui essa ligacao na tua resposta mesmo que a pergunta original seja sobre um artista, instrumento ou genero musical especifico — o objectivo e mostrares que o arquivo e muito mais do que musica. Da preferencia a citacoes diretas das secoes "Transcricao do video" e "Descricao visual" sempre que existirem: uma frase exacta ouvida ou vista no video torna a descoberta muito mais viva e especifica do que uma referencia generica ao artista. Sempre que mencionares um video especifico cujo link esteja disponivel no contexto, inclui o titulo exacto e o link, sempre no formato [Titulo exacto](link).

Quando encontrares ligacoes a estes temas no contexto, desenvolve-as com detalhe — mas TECE-AS NUMA NARRATIVA CONTINUA, nao numa checklist. Cada descoberta deve conduzir naturalmente a seguinte atraves de uma transicao real (uma geografia partilhada, um artista em comum, um tema que ecoa noutro projeto, um contraste interessante), como se estivesses a contar a historia do arquivo a alguem, ponto por ponto mas em fio continuo. Nao comeces frases ou paragrafos com um topico a negrito seguido de dois pontos (por exemplo "**Religiao e procissoes**:") — em vez disso, integra os nomes de artistas, locais e temas dentro do proprio fluxo da frase. NAO uses bullet points, listas nem titulos a negrito dentro deste corpo — apenas paragrafos de prosa corrida.

Por fim, sem qualquer titulo ou frase de transicao anunciada, sugere 3-4 perguntas de seguimento em portugues que o utilizador possa fazer para continuar a navegar o arquivo, uma por linha. Baseia-as tanto na resposta direta como nas descobertas culturais. Torna-as especificas e intrigantes — pelo menos metade deve ser sobre temas culturais presentes no contexto (religiao, festas, gastronomia, artesanato, vida rural, poesia, cultura cigana, medicina popular, etc.).

**Regras:**
- Se uma das tres partes nao tiver contexto para a sustentar, omite-a completamente — mas nunca substituas a parte omitida por um titulo a dizer que foi omitida.
- Nunca uses cabecalhos markdown ("#", "##") nem numeracao ("1.", "2.", "3.") em lado nenhum da resposta.
- Escreve em portugues.
- Se engenhoso e desperta curiosidade.
- O corpo central (a exploracao de descobertas) deve ser o mais longo e detalhado de toda a resposta (minimo 6-8 paragrafos, cobrindo o maior numero possivel de ligacoes distintas), e DEVE ser escrito em paragrafos de prosa corrida — SEM bullet points, listas numeradas, ou frases a negrito como abertura de paragrafo.
- Sempre que o contexto fornecer o link de um video, cita-o sempre no mesmo formato: titulo exacto + link markdown, [Titulo exacto](link) — nunca uses "->", parenteses a seguir ao titulo, ou um URL nu sem o titulo, para que todos os links da resposta tenham a mesma aparencia.
- NAO inventes informacao que nao esteja no contexto — isto aplica-se tambem a links: NUNCA inventes, adivinhes ou alteres um link. So cites um link se ele aparecer literalmente no contexto fornecido.
- NAO te limites a falar de musica — o arquivo contem poesia, danca, artesanato, gastronomia, religiao, festas, historias de vida, paisagens sonoras, cultura cigana, medicina popular e muito mais. Explora TODAS as dimensoes culturais presentes no contexto, especialmente as marcadas como "categoria fora da musica".

---
**Contexto Direto:**
{direct_context}

**Contexto de Descobertas:**
{discovery_context}

**Contexto de Comunidade:**
{community_context}
---

**Pergunta:** {question}

**Resposta:**
"""


def answer_question_with_exploration(
    question, artist_relationships, name_to_artists,
    community_hierarchy, text_token_index, artist_texts,
    chunk_vectors=None, chunk_meta=None,
):
    ctx = retrieve_exploration_context(
        question, artist_relationships, name_to_artists,
        community_hierarchy, text_token_index, artist_texts,
        chunk_vectors=chunk_vectors, chunk_meta=chunk_meta,
    )

    direct_context = '\n\n'.join(ctx['direct']) if ctx['direct'] else ''
    discovery_context = '\n\n'.join(ctx['discoveries']) if ctx['discoveries'] else ''
    community_context = ctx.get('community_summary', '') or ''

    if not direct_context and not discovery_context:
        return "Nao encontrei informacao relevante no arquivo para responder a essa pergunta."

    prompt = build_exploration_prompt(direct_context, discovery_context, community_context, question)

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=12000,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return next(
            (block.text for block in response.content if block.type == 'text'),
            'No response content found.',
        )
    except anthropic.APIError as e:
        return f"Error from Anthropic API: {e}"
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
        state.community_hierarchy,
        state.text_token_index,
        state.artist_texts,
        chunk_vectors=state.chunk_vectors,
        chunk_meta=state.chunk_meta,
    )
