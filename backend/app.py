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

@app.route('/explore-answer', methods=['POST'])
@limiter.limit("20 per minute")
def explore_answer():
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