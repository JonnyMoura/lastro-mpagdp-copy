'''
/utilities/scheduler/setup.py
-> scheduler setup
'''

from apscheduler.schedulers.background import BackgroundScheduler

from database.fetchData import fetchCSV
from ai.rag.setup import rebuildRag

# ==================================================
# global vars
# ==================================================

scheduler = BackgroundScheduler()

# ==================================================
# methods
# ==================================================

def jobToAppContext(app,jobFunction):
    with app.app_context():
        try:
            result = jobFunction()
            print(f"Scheduled CSV fetch completed.")
        except Exception as e:
            print(f"Error in scheduled CSV fetch: {e}")

def fetchCSVAndRebuildRag(app):
    fetchCSV()
    rebuildRag(app)

# ==================================================
# init and clean methods
# ==================================================

def initScheduler(app):
    # fetch data from CSV, then rebuild the RAG knowledge graph from the refreshed data
    scheduler.add_job(
        func=lambda: jobToAppContext(app, lambda: fetchCSVAndRebuildRag(app)),
        trigger='cron',
        hour=4, minute=0,
        timezone='Europe/Lisbon',
        id='fetchCSV_job',
        name='Fetch CSV Data',
        replace_existing=True
    )

    scheduler.start()

def cleanScheduler():
    if scheduler.running:
        scheduler.shutdown(wait=True)