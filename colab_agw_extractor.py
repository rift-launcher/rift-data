# ============================================================
# PART 1: INSTALL DEPENDENCIES (No Google Drive Needed)
# ============================================================
import os
print("⚙️ Installing Dependencies...")
os.system('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
os.system('pip install --no-deps xformers trl peft accelerate bitsandbytes')
os.system('pip install requests')
print("✅ Dependencies Installed!")

# ============================================================
# PART 2: THE APPLE GAMING WIKI AI EXTRACTOR
# ============================================================
# 🔥 CHANGE THESE 4 LINES FOR EACH GOOGLE ACCOUNT 🔥
CHUNK_ID = 1              # Account 1=1, Account 2=2, etc.
CHUNK_START = 0           # Where this account starts
CHUNK_END = 500           # Where this account stops
GITHUB_TOKEN = "YOUR_GITHUB_TOKEN_HERE"

import sqlite3
import json
import requests
import torch
import base64
import concurrent.futures
from unsloth import FastLanguageModel

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DRIVE_PATH = "/content/RIFT"
os.makedirs(DRIVE_PATH, exist_ok=True)

REPO_URL = "https://raw.githubusercontent.com/rift-launcher/rift-data/main/agw_reports.json"
BATCH_SIZE = 7
SAVE_EVERY = 10
PROGRESS_FILE = f"{DRIVE_PATH}/agw_progress_part{CHUNK_ID}.json"
DB_PATH = f"{DRIVE_PATH}/agw_knowledge_part{CHUNK_ID}.db"

# ------------------------------------------------------------
# GITHUB AUTO-PULL & AUTO-PUSH FUNCTIONS
# ------------------------------------------------------------
def pull_from_github(filename, local_path):
    print(f"📡 Checking GitHub for previous {filename}...")
    url = f"https://raw.githubusercontent.com/rift-launcher/rift-data/main/{filename}"
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(resp.content)
        print(f"✅ Successfully pulled {filename} from GitHub!")
    else:
        print(f"⚠️ {filename} not found on GitHub (or starting fresh).")

def push_to_github(local_file, github_filename):
    url = f"https://api.github.com/repos/rift-launcher/rift-data/contents/{github_filename}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    sha = requests.get(url, headers=headers).json().get("sha")
    with open(local_file, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    data = {"message": f"🤖 Auto-save AGW part {CHUNK_ID}", "content": content, "branch": "main"}
    if sha: data["sha"] = sha

    resp = requests.put(url, headers=headers, json=data)
    if resp.status_code in [200, 201]:
        print(f"🚀 Successfully pushed {github_filename} to GitHub!")
    else:
        print(f"❌ GitHub Push Failed: {resp.json()}")

# ------------------------------------------------------------
# DOWNLOAD RAW REPORTS
# ------------------------------------------------------------
print(f"📥 Downloading agw_reports.json from GitHub...")
response = requests.get(REPO_URL)
if response.status_code == 200:
    raw_reports = response.json()
else:
    raise Exception(f"Failed to download: HTTP {response.status_code}")

raw_reports = raw_reports[CHUNK_START:CHUNK_END]
print(f"🎯 This worker handles reports {CHUNK_START} to {CHUNK_END} ({len(raw_reports)} reports)")

# ------------------------------------------------------------
# LOAD PROGRESS (AUTO-PULL) & SETUP DB
# ------------------------------------------------------------
pull_from_github(f"agw_progress_part{CHUNK_ID}.json", PROGRESS_FILE)
pull_from_github(f"agw_knowledge_part{CHUNK_ID}.db", DB_PATH)

try:
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
    resume_offset = progress.get("last_processed_offset", 0)
    total_fixes_saved = progress.get("total_fixes_saved", 0)
    print(f"🔄 Resuming from offset {resume_offset} (previously saved {total_fixes_saved} fixes)")
except:
    resume_offset, total_fixes_saved = 0, 0
    print("🆕 Starting fresh for this chunk!")

if resume_offset >= len(raw_reports):
    print("🎉 This chunk is already complete!")
    import sys; sys.exit()

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS agw_compatibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    game_name TEXT NOT NULL, 
    native_status TEXT, native_notes TEXT,
    rosetta_status TEXT, rosetta_notes TEXT,
    crossover_status TEXT, crossover_notes TEXT,
    whisky_status TEXT, whisky_notes TEXT,
    parallels_status TEXT, parallels_notes TEXT,
    porting_kit_status TEXT, porting_kit_notes TEXT
)''')
conn.commit()

# ------------------------------------------------------------
# LOAD MODEL & PROMPT
# ------------------------------------------------------------
print("🧠 Loading Qwen 2.5 14B Instruct (4-bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit", max_seq_length=8192, dtype=None, load_in_4bit=True)
FastLanguageModel.for_inference(model)

PROMPT = """You are a macOS gaming compatibility data extractor.
YOUR ONLY JOB: Read this AppleGamingWiki wikitext and extract the compatibility status and notes for different wrappers/methods (CrossOver, Whisky, Parallels, Native, Rosetta 2, Porting Kit).

Return a JSON object containing the status and notes for each method. 
If a method is not mentioned, leave the string empty.
For status, typical values are "Perfect", "Playable", "Runs", "Unplayable", "na".

{{
  "native_status": "", "native_notes": "",
  "rosetta_status": "", "rosetta_notes": "",
  "crossover_status": "", "crossover_notes": "",
  "whisky_status": "", "whisky_notes": "",
  "parallels_status": "", "parallels_notes": "",
  "porting_kit_status": "", "porting_kit_notes": ""
}}

HARD RULES:
1. Return ONLY raw JSON. No markdown.
2. NEVER invent statuses. Extract ONLY what is explicitly written in the wikitext.
3. Preserve the exact notes (including HTML tags or references).

REPORT:
Game: {game}
Wikitext: "{notes}"
JSON:"""

def parse_ai_response(response_text, report):
    results = []
    try:
        clean = response_text.replace("```json", "").replace("```", "").strip()
        start = clean.find('{')
        end = clean.rfind('}') + 1
        if start == -1: return results
        parsed = json.loads(clean[start:end])
        results.append((parsed, report))
    except: pass
    return results

# ------------------------------------------------------------
# MAIN PROCESSING LOOP
# ------------------------------------------------------------
print(f"\n⚡ Worker {CHUNK_ID} processing {len(raw_reports)} reports...")
batch_count = 0

for i in range(resume_offset, len(raw_reports), BATCH_SIZE):
    batch = raw_reports[i:i+BATCH_SIZE]
    formatted_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": PROMPT.format(game=r.get("title",""), notes=r.get("notes","")[:3000])}], tokenize=False, add_generation_prompt=True) for r in batch]

    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=400, use_cache=True, pad_token_id=tokenizer.eos_token_id, temperature=0.1, do_sample=True)

    decoded_responses = [tokenizer.decode(out[inputs["input_ids"][j].shape[0]:], skip_special_tokens=True).strip() for j, out in enumerate(outputs)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = [executor.submit(parse_ai_response, resp, report) for resp, report in zip(decoded_responses, batch)]
        for future in concurrent.futures.as_completed(futures):
            for fix, report in future.result():
                c.execute('''INSERT INTO agw_compatibility (game_name, native_status, native_notes, rosetta_status, rosetta_notes, crossover_status, crossover_notes, whisky_status, whisky_notes, parallels_status, parallels_notes, porting_kit_status, porting_kit_notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (report.get("title", ""), fix.get("native_status", ""), fix.get("native_notes", ""), fix.get("rosetta_status", ""), fix.get("rosetta_notes", ""), fix.get("crossover_status", ""), fix.get("crossover_notes", ""), fix.get("whisky_status", ""), fix.get("whisky_notes", ""), fix.get("parallels_status", ""), fix.get("parallels_notes", ""), fix.get("porting_kit_status", ""), fix.get("porting_kit_notes", "")))
                total_fixes_saved += 1

    conn.commit()
    batch_count += 1

    # AUTO-SAVE LOCALLY AND PUSH TO GITHUB
    if batch_count % SAVE_EVERY == 0:
        conn.close(); conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"last_processed_offset": i + BATCH_SIZE, "total_fixes_saved": total_fixes_saved, "chunk_id": CHUNK_ID, "chunk_range": f"{CHUNK_START}-{CHUNK_END}"}, f)

        pct = ((i + BATCH_SIZE) / len(raw_reports)) * 100
        print(f"💾 Local Save! {i+BATCH_SIZE}/{len(raw_reports)} ({pct:.1f}%) | Fixes: {total_fixes_saved}")

        push_to_github(DB_PATH, f"agw_knowledge_part{CHUNK_ID}.db")
        push_to_github(PROGRESS_FILE, f"agw_progress_part{CHUNK_ID}.json")

    elif batch_count % 10 == 0:
        pct = ((i + BATCH_SIZE) / len(raw_reports)) * 100
        print(f"⚡ Worker {CHUNK_ID} | {i+BATCH_SIZE}/{len(raw_reports)} ({pct:.1f}%) | Fixes: {total_fixes_saved}")

# FINAL SAVE
conn.commit(); conn.close()
with open(PROGRESS_FILE, "w") as f:
    json.dump({"last_processed_offset": len(raw_reports), "total_fixes_saved": total_fixes_saved, "chunk_id": CHUNK_ID, "status": "COMPLETE"}, f)
push_to_github(DB_PATH, f"agw_knowledge_part{CHUNK_ID}.db")
push_to_github(PROGRESS_FILE, f"agw_progress_part{CHUNK_ID}.json")
print(f"\n🎉 Worker {CHUNK_ID} DONE! Everything pushed to GitHub!")
