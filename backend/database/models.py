'''
/database/models.py
-> define all the models of the app db
'''

from sqlalchemy import Date
from datetime import datetime
from database.setup import db

# ==================================================
# Utility functions
# ==================================================

def serializeProjectMinimal(project_dict):
    """
    Convert raw SQL project dict to minimal serialized format.
    Only returns essential fields needed for display in suggestions and search results.
    """
    return {
        "id": project_dict.get("id"),
        "title": project_dict.get("title"),
        "author": project_dict.get("author"),
        "category": project_dict.get("category")
    }

# -----------------------------
# Project model
# -----------------------------

class Project(db.Model):
    __tablename__ = 'projects'

    id = db.Column(db.Integer, primary_key=True)

    link = db.Column(db.String(128), unique=True, nullable=False)
    title = db.Column(db.String(128))
    author = db.Column(db.String(128))
    category = db.Column(db.String(256))
    date = db.Column(Date)

    # technical info
    direction = db.Column(db.String(256))
    sound = db.Column(db.String(256))
    production = db.Column(db.String(256))
    support = db.Column(db.String(256))
    assistance = db.Column(db.String(256))
    research = db.Column(db.String(256))

    # geo info
    location   = db.Column(db.String)
    municipality = db.Column(db.String)
    district   = db.Column(db.String)
    region     = db.Column(db.String)
    # instruments
    instruments = db.Column(db.String(256))

    # other info
    keywords = db.Column(db.String(1024))
    history = db.Column(db.String(2048))
    other_info = db.Column(db.String(2048))
    biographies = db.Column(db.String(2048))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def serialize(self):
        return {
            "id": self.id,
            "link": self.link,
            "title": self.title,
            "author": self.author,
            "category": self.category,
            "date": self.date.isoformat() if self.date else None,

            "direction": self.direction,
            "sound": self.sound,
            "production": self.production,
            "support": self.support,
            "assistance": self.assistance,
            "research": self.research,

            "location": self.location,
            "municipality": self.municipality,
            "district": self.district,
            "region": self.region,

            "instruments": self.instruments,

            "keywords": self.keywords,
            "history": self.history,
            "other_info": self.other_info,
            "biographies": self.biographies,

            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f'<Project {self.id}>'
    
# -----------------------------
# Interaction model
# -----------------------------

class Interaction(db.Model):
    __tablename__ = 'userActivity'

    id = db.Column(db.Integer, primary_key=True)

    userInput = db.Column(db.String(512))
    userPublicIP = db.Column(db.String(32))
    modelOutput = db.Column(db.String(512))
    score = db.Column(db.Integer, default=0) # 0 not rated, -1 bad, 1 good

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def serialize(self):
        return {
            "id": self.id,

            "userInput": self.userInput,
            "userPublicIP": self.userPublicIP,
            "modelOutput": self.modelOutput,
            "score": self.score,

            "created_at": self.created_at.isoformat() if self.created_at else None
        }