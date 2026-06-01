'''
/database/fetchData.py
-> handle of the csv reading and db writing
'''

from io import StringIO
import requests, pandas as pd
from dotenv import load_dotenv
import os, time

from database.models import db, Project
from database.reportBuilder import ReportBuilder
from utilities.vimeo.setup import getVimeoDate

load_dotenv()

# ==================================================
# global vars
# ==================================================

GOOGLE_SHEETS_URL = os.getenv("GOOGLE_SHEETS_URL")

# ==================================================
# auxiliar methods
# ==================================================

def capitalize_portuguese_phrase(phrase):
    """
    Capitalize a Portuguese phrase, handling hyphenated words correctly.
    Each word is capitalized, but for hyphenated words like 'Sacro-profana', 
    only the first part before the hyphen is capitalized.
    """
    if not phrase:
        return phrase
    
    # Split by spaces to handle multi-word phrases
    words = phrase.split(' ')
    capitalized_words = []
    
    for word in words:
        # For hyphenated words, capitalize only the first part
        if '-' in word:
            parts = word.split('-')
            capitalized_word = parts[0].capitalize() + '-' + '-'.join(p.lower() for p in parts[1:])
            capitalized_words.append(capitalized_word)
        else:
            capitalized_words.append(word.capitalize())
    
    return ' '.join(capitalized_words)

def normalizeString(string, capitalize_keywords=False):
    if not isinstance(string, str):
        return ''
    
    string = string.strip()
    if not string:
        return ''
    
    delimiters = ['\n', ' e ', ' & ', ',']
    
    for delimiter in delimiters:
        string = string.replace(delimiter, ', ')
    
    parts = [part.strip() for part in string.split(',') if part.strip()]
    
    # Case-insensitive deduplication, handling hyphens/spaces as equivalent
    seen = {}
    unique_parts = []
    for part in parts:
        # Normalize for comparison: convert hyphens to spaces and lowercase for case-insensitive comparison
        normalized = part.replace('-', ' ').lower()
        if normalized not in seen:
            seen[normalized] = True
            # Apply Portuguese-aware capitalization only if capitalize_keywords is True
            if capitalize_keywords:
                unique_parts.append(capitalize_portuguese_phrase(part))
            else:
                unique_parts.append(part)
    
    return ', '.join(unique_parts)

def concatStrings(strings):
    result = ''

    for s in strings:
        if isinstance(s, str):
            result = result + ", " + s
    
    if result != '': 
        return normalizeString(result.replace('\n', ', ')[2:])
    
    return ''

# ==================================================
# POST and PUT handle
# ==================================================

def insertProject(pid, p, cleanedLink, lineIndex, reporter):
    checkpointCommit = False
    
    date = getVimeoDate(pid)
   
    if date == "RATE_LIMIT_EXCEEDED": 
        checkpointCommit = True
        print("Sleeping for 1 minute to prevent blocking.")
        time.sleep(61)
        date = getVimeoDate(pid)
   
    if isinstance(date, str) and "error" in date:
        reporter.addError(lineIndex, pid, date)
        return checkpointCommit

    newProject = Project(
        id=pid, # use the vimeo id as project row id
        link=cleanedLink,
        title= p['Tema'] if isinstance(p['Tema'], str) else '',
        author=p['Nome'] if isinstance(p['Nome'], str) else '',
        date=date,
        category=p['Categoria'] if isinstance(p['Categoria'], str) else '',
        direction=normalizeString(p['Realizador']),
        sound=normalizeString(p['Som']),
        production = normalizeString(p['Produção']),
        support = normalizeString(p['Apoio']),
        assistance = normalizeString(p['Assistência']),
        research = normalizeString(p['Pesquisa']),
        
        location=p['Local'] if isinstance(p['Local'], str) else '',
        municipality=p['Concelho'] if isinstance(p['Concelho'], str) else '',
        district=p['Distrito/Ilha'] if isinstance(p['Distrito/Ilha'], str) else '',
        region=p['Região'] if isinstance(p['Região'], str) else '',
        instruments = normalizeString(p['Instrumentos']),
        
        keywords = normalizeString(concatStrings([p['Palavras-chave'],p['Conceitos-chave']]), capitalize_keywords=True),
        history = p['História (textos que acompanham vídeos)'] if isinstance(p['História (textos que acompanham vídeos)'], str) else '',
        other_info = p['Outras Informações'] if isinstance(p['Outras Informações'], str) else '',
        biographies = p['Biografias'] if isinstance(p['Biografias'], str) else ''
    )

    db.session.add(newProject)
    reporter.addCreatedProject(lineIndex, pid)
    
    if checkpointCommit:
        db.session.commit()

def updateProject(existingProject, p, lineIndex, cleanedLink, reporter):
    checkpointCommit = False
    
    changes = []
    
    updates = [
        ('title', p['Tema'] if isinstance(p['Tema'], str) else ''),
        ('author', p['Nome'] if isinstance(p['Nome'], str) else ''),
        ('link', cleanedLink),
        ('category', p['Categoria'] if isinstance(p['Categoria'], str) else ''),
        ('direction', normalizeString(p['Realizador'])),
        ('sound', normalizeString(p['Som'])),
        ('production', normalizeString(p['Produção'])),
        ('support', normalizeString(p['Apoio'])),
        ('assistance', normalizeString(p['Assistência'])),
        ('research', normalizeString(p['Pesquisa'])),
        ('location', p['Local'] if isinstance(p['Local'], str) else ''),
        ('municipality', p['Concelho'] if isinstance(p['Concelho'], str) else ''),
        ('district', p['Distrito/Ilha'] if isinstance(p['Distrito/Ilha'], str) else ''),
        ('region', p['Região'] if isinstance(p['Região'], str) else ''),
        ('instruments', normalizeString(p['Instrumentos'])),
        
        ('instruments', normalizeString(p['Instrumentos'])),
        
        ('keywords', normalizeString(concatStrings([p['Palavras-chave'],p['Conceitos-chave']]), capitalize_keywords=True)),
        ('history', p['História (textos que acompanham vídeos)'] if isinstance(p['História (textos que acompanham vídeos)'], str) else ''),
        ('other_info', p['Outras Informações'] if isinstance(p['Outras Informações'], str) else ''),
        ('biographies', p['Biografias'] if isinstance(p['Biografias'], str) else '')
    ]

    # avoid unecessary access to Vimeo API
    if getattr(existingProject, "date") is None:
        date = getVimeoDate(existingProject.id)

        if date == "RATE_LIMIT_EXCEEDED": 
            checkpointCommit = True
            print("Sleeping for 1 minute to prevent blocking.")
            time.sleep(61)
            date = getVimeoDate(existingProject.id)

        if isinstance(date, str) and "error" in date:
            reporter.addError(lineIndex, existingProject.id, date)
            return checkpointCommit
        
        updates.append(("date", date))
    
    for field, new_val in updates:
        #print(field, new_val)
        if getattr(existingProject, field) != new_val:
            setattr(existingProject, field, new_val)
            changes.append(field)
    
    if changes:
        reporter.addUpdatedProject(lineIndex, existingProject.id, changes)
    else:
        reporter.addUnchangedLine(lineIndex)
    
    if checkpointCommit:
        db.session.commit()

# ==================================================
# fetch logic
# ==================================================

def fetchCSV():
    reporter = ReportBuilder()
    visitedIds = {}
    duplicateIds = {} 

    try:
        # fetch CSV data (certificates handle), UTF-8 encoding not to loose chars like 'Ç'
        response = requests.get(GOOGLE_SHEETS_URL)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CSV: {e}")
        return {"error": str(e)}, 500

    df = pd.read_csv(StringIO(response.text))

    reporter.initialize(len(df))

    for lineIndex, p in enumerate(df.iloc, 1):

        print(f"{lineIndex}/{len(df)}")
        lineIndex = lineIndex + 1 # csv header compensation

        cleanedLink = p['Link'].replace(' ', '').replace('\n', '') if isinstance(p['Link'], str) else p['Link']
        if (isinstance(cleanedLink, str) and 'vimeo.com/' in cleanedLink and cleanedLink[-1].isdigit() == False): 
            cleanedLink = cleanedLink[:-1]
       
        # check if link exists and is valid
        if not isinstance(cleanedLink, str) or 'vimeo.com/' not in cleanedLink or not cleanedLink[-1].isdigit(): 
            if pd.isna(cleanedLink) or str(cleanedLink).lower() == 'nan':
                reporter.addNanLine(lineIndex)
                continue
            else:
                reporter.addInvalidLink(lineIndex, cleanedLink)
                continue

        reporter.flushNanBatch()

        pid = int(cleanedLink.split("/")[-1])

        # Check for duplicates
        if pid in visitedIds:
            if pid not in duplicateIds:
                duplicateIds[pid] = [visitedIds[pid]]
            duplicateIds[pid].append(lineIndex)
            
            reporter.addUnchangedLine(lineIndex)
            continue
        
        visitedIds[pid] = lineIndex

        existingProject = Project.query.filter_by(id = pid).first()
        
        if existingProject:
            updateProject(existingProject, p, lineIndex, cleanedLink, reporter)
        else:
            reporter.flushUnchangedBatch()
            insertProject(pid, p, cleanedLink, lineIndex, reporter)
        
        if lineIndex % 100 == 0:
            db.session.commit()
        
    reporter.flushNanBatch()
    reporter.addDuplicateSummary(duplicateIds)
    reporter.addDatabaseSummary(len(Project.query.all()))

    db.session.commit()

    return reporter.finalize()