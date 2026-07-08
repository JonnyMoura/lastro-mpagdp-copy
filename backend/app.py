'''
/app.py
-> main file of the flask app
'''

from flask import Flask, Response, jsonify, request
import json
from dotenv import load_dotenv
load_dotenv()

from dataGen.queries import handleQuery
from dataGen.suggestions import getSuggestions

from database.setup import initDatabase, cleanInteractions
from database.fetchData import fetchCSV
from database.models import Project, Interaction

from ai.rag.setup import initRag
from ai.rag.ragQuery import answerExplorationQuestion

from utilities.scheduler.setup import initScheduler, cleanScheduler
from utilities.cors.setup import initCors
from utilities.ratelimit.setup import initRateLimiter, limiter

app = Flask(__name__)
initDatabase(app)
initRag(app)
initCors(app)
initRateLimiter(app)
initScheduler(app)

# ==================================================
# routes
# ==================================================

@app.route('/projects', methods=['GET'])
def get_projects():
    projects = Project.query.all()
    data = []
    for project in projects:
        try:
            data.append(project.serialize())
        except Exception as e:
            print(f"Error serializing project {project.id}: {e}")
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    return Response(json_str, mimetype='application/json; charset=utf-8')

@app.route('/projects/<int:project_id>', methods=['GET'])
def get_project(project_id):
    return jsonify(Project.query.get_or_404(project_id).serialize())



@app.route('/suggestions/<int:project_id>', methods=['GET'])
def get_suggestions(project_id):
    return jsonify(getSuggestions(Project.query.get_or_404(project_id)))

@app.route('/query', methods=['POST'])
@limiter.limit("20 per minute")
def handle_query():
    return jsonify(handleQuery(request.json))

EXPLORE_ANSWER_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Explore - Lastro</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Geist', 'Segoe UI', system-ui, sans-serif;
        background-color: #080808;
        color: #e0e0e0;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 40px 20px;
    }
    .container { width: 100%; max-width: 640px; }
    h1 { font-size: 20px; margin-bottom: 16px; font-weight: 500; }
    textarea {
        width: 100%; min-height: 90px; padding: 12px; border-radius: 8px;
        border: 1px solid #333; background: #141414; color: #e0e0e0;
        font-size: 15px; resize: vertical; font-family: inherit;
    }
    button {
        margin-top: 12px; padding: 10px 20px; border-radius: 8px; border: none;
        background: #e0e0e0; color: #080808; font-size: 14px; font-weight: 600;
        cursor: pointer;
    }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #answer {
        margin-top: 24px; white-space: pre-wrap; line-height: 1.5;
        border-top: 1px solid #222; padding-top: 20px; display: none;
    }
    #status { margin-top: 12px; color: #888; font-size: 14px; }
</style>
</head>
<body>
<div class="container">
    <h1>Explore o arquivo (RAG + Knowledge Graph)</h1>
    <textarea id="question" placeholder="Ex: Quem toca gaita de foles?"></textarea>
    <br>
    <button id="ask">Perguntar</button>
    <div id="status"></div>
    <div id="answer"></div>
</div>
<script>
    const btn = document.getElementById('ask');
    const questionEl = document.getElementById('question');
    const statusEl = document.getElementById('status');
    const answerEl = document.getElementById('answer');

    async function ask() {
        const question = questionEl.value.trim();
        if (!question) return;
        btn.disabled = true;
        statusEl.textContent = 'A pensar... (pode demorar ate 2 minutos)';
        answerEl.style.display = 'none';

        try {
            const res = await fetch('/explore-answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });
            const data = await res.json();
            answerEl.textContent = data.answer || JSON.stringify(data);
            answerEl.style.display = 'block';
            statusEl.textContent = '';
        } catch (e) {
            statusEl.textContent = 'Erro: ' + e;
        } finally {
            btn.disabled = false;
        }
    }

    btn.addEventListener('click', ask);
    questionEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask();
    });
</script>
</body>
</html>
"""

@app.route('/explore-answer', methods=['GET', 'POST'])
@limiter.limit("20 per minute", methods=["POST"])
def explore_answer():
    if request.method == 'GET':
        return Response(EXPLORE_ANSWER_PAGE, mimetype='text/html')
    question = request.json.get("question", "")
    return jsonify({"answer": answerExplorationQuestion(question)})

@app.route('/fetch-csv')
@limiter.limit("20 per minute")
def fetch_csv():
    return fetchCSV()

@app.route('/user-activity', methods=['GET'])
@limiter.limit("20 per minute")
def get_user_activity():
    data = [interaction.serialize() for interaction in Interaction.query.order_by(Interaction.id.desc()).all()]
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    return Response(json_str, mimetype='application/json; charset=utf-8')

@app.route('/clean-interactions', methods=['POST', 'GET'])
@limiter.limit("20 per minute")
def clean_interactions():
    return jsonify(cleanInteractions())

@app.route('/')
def home():
    return """
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Geist', 'Segoe UI', system-ui, sans-serif;
            background-color: #080808;
            color: #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            font-size: 16px;
        }
    </style>
    <body>Lastro Backend is running!</body>
    """

@app.route('/project-count', methods=['GET'])
def get_project_count():
    count = Project.query.count()
    return jsonify({"project_count": count})

# ==================================================
# main
# ==================================================

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=True)
    finally:
        cleanScheduler()