import json
import os
import time
import pandas as pd
import yt_dlp
import concurrent.futures
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydantic import BaseModel, Field
import sqlite3

# 1. Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("ERROR: Missing or invalid API key! Check your .env file.")

# Initialize the Google client
client = genai.Client(api_key=API_KEY)

# 2. Setup folders and files
folder_downloads = "temp_videos"
os.makedirs(folder_downloads, exist_ok=True)
output_csv = "results.csv"

# 3. Load database from Excel
db_file = r"C:\PRAKTYKI\lastro-mpagdp-copy\backend\database\lastro.db"

prompt_spain = """
Analyze this video featuring the artist {artist} and the track '{title}'. 
Please provide a highly detailed, comprehensive breakdown divided into two main sections:

1. AUDIO TRANSCRIPTION:
Provide a complete and exact text transcription of everything spoken or sung in the video. 

CRITICAL REGIONAL CONTEXT BASED ON DATA:
The metadata for this specific video indicates the following location:
- Concelho/Municipio: {concelho}
- Distrito/Provincia/Região: {distrito}

CRITICAL DEFAULT RULE: First, verify if there are actually any lyrics being sung or spoken in the video. 
If the video is ENTIRELY INSTRUMENTAL (e.g., only instruments are being played, like pandeiretas, bagpipes, or accordions, and nobody is singing words), you MUST NOT transcribe anything. Instead, strictly write exactly: 'Instrumental - no lyrics spoken or sung.'
DO NOT HALLUCINATE OR INVENT LYRICS based on the artist name or title if the audio track contains only instrumental music.

LANGUAGE IDENTIFICATION & TRANSCRIPTION RULES:
You MUST adapt the transcription language strictly according to these rules based on the provided location ({concelho}, {distrito}):
1. If the Region/Distrito involves 'Galiza' or 'Galicia' or 'Corunha' or 'Ourense' or 'Pontevedra' or 'Lugo': 
   - The language is GALICIAN (Galego). It looks similar to Portuguese, but is distinct. 
   - Transcribe strictly in Galician. Keep words like 'mozos' (NOT 'moços'), 'polas portas' (NOT 'pelas portas'), 'hai unha pedra' (NOT 'tem/tei unha pedra'), and all diminutives ending in '-iño'/'-iña'.
2. If the Region/Distrito involves 'León' or 'Zamora' (especially Sanabria like Robledo de Sanabria) or 'Salamanca' or 'Castilla y Leon':
   - The language is ASTUR-LEONESE, SENABRÉS, or a heavy regional DIALECT of Spanish.
   - You MUST preserve all phonetic dialectal traits, such as dropping the 'd' in '-ado' -> '-ao' (e.g., 'cantao', 'pintelao'), vowel closings (like '-u' instead of '-o'), or local archaic words. Do NOT normalize into standard Spanish (es-ES).

STRICT TRANSCRIPTION RULES:
- RULE 1: Listen to the lyrics and cross-reference with the provided location. Transcribe EXACTLY what is sung without correcting, modernizing, or translating.
- RULE 2: Do NOT translate the lyrics into English, standard Portuguese, or standard Spanish. 

2. VISUAL DESCRIPTION:
Describe exactly what is happening in the video in extreme detail. 
CRITICAL: Even though the audio is from Spain, the visual descriptions MUST be written entirely in PORTUGUESE (pt-PT). You must specifically include detailed descriptions of:
- The background, setting, and environment.
- The person/artist (their appearance, posture, facial expressions, and movements).
- The specific instruments being played or visible in the frame.
- The exact clothing and attire of anyone in the video.
- Any text appearing on the screen.

CRITICAL RULES FOR VISUAL DESCRIPTION:
Do not miss any visual details. However, you MUST NOT invent or assume on-screen text. 
Do NOT use the artist name or track title provided in this prompt to hallucinate graphics. 
If there is absolutely no text written directly on the video frame, you MUST explicitly write: 'No text on screen'.
"""

prompt_vimeo = """
Analyze this video featuring the artist {artist} and the track '{title}'. 
Please provide a highly detailed, comprehensive breakdown divided into two main sections:

1. AUDIO TRANSCRIPTION:
Provide a complete and exact text transcription of everything spoken or sung in the video. 

CRITICAL REGIONAL CONTEXT BASED ON DATA:
The metadata for this specific video indicates the following location in Portugal:
- Concelho: {concelho}
- Distrito/Região: {distrito}

CRITICAL DEFAULT RULE: First, verify if there are actually any lyrics being sung or spoken in the video. 
If the video is ENTIRELY INSTRUMENTAL (e.g., only instruments are being played, like pandeiretas, bagpipes, or accordions, and nobody is singing words), you MUST NOT transcribe anything. Instead, strictly write exactly: 'Instrumental - no lyrics spoken or sung.'
DO NOT HALLUCINATE OR INVENT LYRICS based on the artist name or title if the audio track contains only instrumental music.

LANGUAGE IDENTIFICATION RULE (CRITICAL FOR MIRANDÊS):
If the Concelho is 'Miranda do Douro', the audio may be in MIRANDÊS (Mirandese) OR in standard PORTUGUESE (pt-PT).
- Your first task is to listen carefully and determine if the lyrics/speech are in Mirandês or Portuguese.
- If Mirandese is detected (characterized by words like 'lhiteratura', 'tierra', 'falar', distinct phonetics, and roots shared with Astur-Leonese), you MUST transcribe it exactly as it is spoken in MIRANDÊS. Do NOT attempt to translate, normalize, or auto-correct Mirandese words into standard Portuguese.
- If it is standard Portuguese, transcribe it strictly in pt-PT. Do not translate the lyrics.

STRICT TRANSCRIPTION RULES:
- RULE 1: You MUST transcribe the lyrics EXACTLY as they are spoken or sung. No auto-correct, no normalization, and no translation into English or any other language.

2. VISUAL DESCRIPTION:
Describe exactly what is happening in the video in extreme detail. 
CRITICAL: All visual descriptions MUST be written entirely in PORTUGUESE (pt-PT). You must specifically include detailed descriptions of:
- The background, setting, and environment.
- The person/artist (their appearance, posture, facial expressions, and movements).
- The specific instruments being played or visible in the frame.
- The exact clothing and attire of anyone in the video.
- Any text appearing on the screen.

CRITICAL RULES FOR VISUAL DESCRIPTION:
Do not miss any visual details. However, you MUST NOT invent or assume on-screen text. 
Do NOT use the artist name or track title provided in this prompt to hallucinate graphics. 
If there is absolutely no text written directly on the video frame, you MUST explicitly write: 'No text on screen'.
"""

class VideoAnalysis(BaseModel):
    audio_transcription: str = Field(description="Complete audio transcription or 'Instrumental - no lyrics spoken or sung.'")
    visual_description: str = Field(description="Detailed visual description in pt-PT.")

def process_single_video(index, link_vimeo, artist, title, region_context, concelho, distrito):
    prefix = f"[Row {index + 1} | {artist}]"
    local_path = os.path.join(folder_downloads, f"video_{index}.mp4")
    gemini_video = None

    print(f"{prefix} Starting process...")

    ydl_opts = {
        'format': 'best[height<=480][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': local_path,
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {'vimeo': {'player_client': ['web']}},
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        },
        'socket_timeout': 40,
        'retries': 10,
        'fragment_retries': 10,
        'file_access_retries': 5
    }

    try:
        # STEP A: Download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link_vimeo])

        if not os.path.exists(local_path):
            return f"{prefix} Error: File not downloaded."

        # STEP B: Upload
        gemini_video = client.files.upload(file=local_path)

        # STEP C: Wait for processing
        while gemini_video.state.name == "PROCESSING":
            time.sleep(3)
            gemini_video = client.files.get(name=gemini_video.name)

        if gemini_video.state.name == "FAILED":
            return f"{prefix} Error: Cloud processing failed."

        region_str = str(region_context).strip()

        if 'Espanha' in region_str:
            ai_prompt = prompt_spain.format(
                artist=artist,
                title=title,
                concelho=concelho,
                distrito=distrito
            )
        else:
            ai_prompt = prompt_vimeo.format(
                artist=artist,
                title=title,
                concelho=concelho,
                distrito=distrito
            )

        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=[gemini_video, ai_prompt],
            config={
                'response_mime_type': 'application/json',
                'response_schema': VideoAnalysis,
            }
        )

        try:
            analysis_data = json.loads(response.text)
            transcription = analysis_data.get("audio_transcription", "")
            visual_desc = analysis_data.get("visual_description", "")
        except Exception as json_err:
            print(f"{prefix} JSON Parsing failed, saving raw text. Error: {json_err}")
            transcription = response.text
            visual_desc = "ERROR: Could not parse json"

        result_df = pd.DataFrame([{
            'Index': index + 1,
            'Artist': artist,
            'Title': title,
            'Link': link_vimeo,
            'Audio transcription': transcription,
            'Visual description': visual_desc
        }])

        result_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

        print(f"{prefix} SUCCESS! Data saved.")
        return f"{prefix} Done."

    except errors.APIError as e:
        print(f"{prefix} API LIMIT OR ERROR: {e}")
        return f"{prefix} Failed due to API."
    except Exception as e:
        print(f"{prefix} Unexpected Error: {e}")
        return f"{prefix} Failed."

    finally:
        # STEP E: Cleanup
        if gemini_video:
            try:
                client.files.delete(name=gemini_video.name)
            except:
                pass
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except:
                pass


processed_ids = set()
if os.path.exists(output_csv):
    try:
        column_names = ['Index', 'Artist', 'Title', 'Link', 'Audio transcription', 'Visual description']
        df_results = pd.read_csv(output_csv, header=None, names=column_names, encoding='cp1252', encoding_errors='ignore')
        if 'Link' in df_results.columns:
            for l in df_results['Link'].dropna().astype(str):
                digits = ''.join(filter(str.isdigit, l))
                if digits:
                    processed_ids.add(digits)
            print(f" Found {len(processed_ids)} already processed video IDs in '{output_csv}'. They will be skipped.")
    except Exception as e:
        print(f"Could not read existing results.csv: {e}")

conn = sqlite3.connect(db_file)
try:
    df_all = pd.read_sql_query("SELECT * FROM projects", conn)
finally:
    conn.close()

df_clean = df_all.dropna(subset=['link'])
print(f"Total rows to check in DB: {len(df_clean)}")

# --- MULTITHREADING EXECUTOR ---
MAX_CONCURRENT_THREADS = 3


with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_THREADS) as executor:
    futures = []
    currently_submitted_in_batch = 0
    for index, row in df_clean.iterrows():
        current_link = str(row['link']).strip()

        if not current_link or current_link == 'nan':
            continue

        current_id = ''.join(filter(str.isdigit, current_link))

        if current_id in processed_ids:
            print(f"[Row {index + 1}] Already processed. Skipping...")
            continue

        futures.append(
            executor.submit(
                process_single_video,
                index,
                current_link,
                row['author'],
                row['title'],
                row['region'],
                row['municipality'],
                row['district']  
            )
        )
        currently_submitted_in_batch += 1

        if currently_submitted_in_batch == MAX_CONCURRENT_THREADS:
            print(
                f" Sent {MAX_CONCURRENT_THREADS} NEW video to queue. Pausing for 45 seconds to protect TPM/RPM limits...")
            time.sleep(45)
            currently_submitted_in_batch = 0

    if futures:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result and "API LIMIT" in result:
                print(" One of the threads hit strict API Limit. Cooling down execution for 30 seconds...")
                time.sleep(30)
    else:
        print(" All links from Database are already processed!")

print(f" Check '{output_csv}' for your data.")
