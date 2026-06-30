# =============================================================
# RIFT AI EXTRACTOR — Google Colab Notebook
# Paste each section into a separate Colab cell
# =============================================================

# ======================== CELL 1 ========================
# Install Dependencies & Mount Drive

# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes
# !pip install requests
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# import os
# DRIVE_PATH = "/content/drive/MyDrive/RIFT"
# os.makedirs(DRIVE_PATH, exist_ok=True)
# print(f"✅ Google Drive mounted. RIFT folder: {DRIVE_PATH}")


# ======================== CELL 2 ========================
# The Complete 14B Parallel Extraction Pipeline

import sqlite3
import json
import hashlib
import shutil
import requests
import torch
import concurrent.futures
from unsloth import FastLanguageModel

# ============================================================
# CONFIGURATION
# ============================================================
DRIVE_PATH = "/content/drive/MyDrive/RIFT"
REPO_URL = "https://raw.githubusercontent.com/rift-launcher/rift-data/main/raw_reports.json"
BATCH_SIZE = 8          # 8 reports per GPU batch (14B 4-bit fits this on T4)
SAVE_EVERY = 50         # Checkpoint to Drive every 50 batches (400 reports)
PROGRESS_FILE = f"{DRIVE_PATH}/progress.json"
DB_PATH = f"{DRIVE_PATH}/knowledge.db"

# ============================================================
# 1. DOWNLOAD RAW REPORTS
# ============================================================
print("📥 Downloading raw_reports.json from GitHub...")
response = requests.get(REPO_URL)
if response.status_code == 200:
    raw_reports = response.json()
    print(f"✅ Downloaded {len(raw_reports)} reports!")
else:
    raise Exception(f"Failed to download: HTTP {response.status_code}")

# ============================================================
# 2. LOAD PROGRESS (resume from where we left off)
# ============================================================
try:
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
    start_index = progress.get("last_processed_index", 0)
    total_fixes_saved = progress.get("total_fixes_saved", 0)
    print(f"🔄 Resuming from report #{start_index} (previously saved {total_fixes_saved} fixes)")
except (FileNotFoundError, json.JSONDecodeError):
    start_index = 0
    total_fixes_saved = 0
    print("🆕 Starting fresh!")

if start_index >= len(raw_reports):
    print("🎉 All reports have already been processed! Nothing to do.")
else:
    remaining = len(raw_reports) - start_index
    estimated_hours = (remaining / BATCH_SIZE * 3) / 3600  # ~3 sec per batch with parallel
    print(f"📊 Reports remaining: {remaining} | Estimated time: {estimated_hours:.1f} hours")

# ============================================================
# 3. SETUP DATABASE
# ============================================================
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS game_fixes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_name TEXT NOT NULL,
    app_id TEXT,
    proton_version TEXT,
    user_specs TEXT,
    user_os TEXT,
    rating TEXT,
    symptom TEXT DEFAULT '',
    backend TEXT DEFAULT '',
    launch_args TEXT DEFAULT '',
    env_vars TEXT DEFAULT '',
    winetricks TEXT DEFAULT '',
    custom_proton TEXT DEFAULT '',
    kernel_tip TEXT DEFAULT '',
    notes_raw TEXT DEFAULT ''
)''')
conn.commit()
print("📦 Database ready!")

# ============================================================
# 4. LOAD MODEL
# ============================================================
print("🧠 Loading Qwen 2.5 14B Instruct (4-bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("✅ 14B Model loaded!")

# ============================================================
# 5. THE BULLETPROOF PROMPT — INSTANT JUNK DETECTION
# ============================================================
PROMPT = """You are a Linux gaming compatibility data extractor.

YOUR ONLY JOB: Read this ProtonDB report and decide:
A) Does it contain ANY technical fix, workaround, setting change, or configuration tip? → Extract it as JSON.
B) Is it just an opinion, "works fine", rating with no details, or complaint with no solution? → Return EXACTLY: {{}}

DECISION SHORTCUT — Return {{}} immediately if the notes:
- Only say the game "works", "runs", "plays fine", "no issues", "perfect", etc.
- Only contain hardware specs or FPS numbers with no fix
- Only contain complaints like "crashes" or "broken" with NO solution provided
- Are just one or two words with no actionable info

IF there IS a fix, extract into this JSON (leave "" for fields not mentioned):
{{
  "symptom": "the specific problem (e.g. black screen, no audio, crash on save)",
  "backend": "ONLY one of: dxvk, vkd3d, wined3d, d3dmt, or empty string",
  "launch_args": "steam launch options (e.g. -dx11, %command% --skip-launcher, gamemoderun %command%)",
  "env_vars": "environment variables (e.g. PROTON_USE_WINED3D=1, DXVK_ASYNC=1)",
  "winetricks": "winetricks components (e.g. vcrun2019, d3dcompiler_47)",
  "custom_proton": "custom proton build (e.g. GE-Proton8-14, proton-cachyos)",
  "kernel_tip": "kernel or system advice (e.g. use linux-zen, enable fsync, install gamemode)"
}}

HARD RULES:
1. Return ONLY raw JSON. No markdown. No explanation. No text outside the braces.
2. NEVER invent fixes. Extract ONLY what the user explicitly wrote.
3. If multiple distinct fixes exist in one report, return a JSON array of objects.

REPORT:
Game: {game} | Proton: {proton} | Rating: {rating}
Hardware: {specs} | OS: {os}
User Notes: "{notes}"

JSON:"""

# ============================================================
# 6. PARALLEL JSON PARSING (runs on CPU while GPU generates)
# ============================================================
def parse_ai_response(response_text, report):
    """Parse one AI response and return list of (fix_dict, report) tuples to insert."""
    results = []
    try:
        clean = response_text.replace("```json", "").replace("```", "").strip()
        # Find the JSON part (handle cases where model adds extra text)
        start = clean.find('{')
        end = clean.rfind('}') + 1
        if start == -1:
            start = clean.find('[')
            end = clean.rfind(']') + 1
        if start == -1:
            return results
        clean = clean[start:end]
        parsed = json.loads(clean)

        if isinstance(parsed, dict):
            fixes = [parsed]
        elif isinstance(parsed, list):
            fixes = parsed
        else:
            return results

        for fix in fixes:
            # Empty dict = model correctly identified junk report
            if not fix:
                continue
            # Only save if it contains at least ONE actionable field
            has_fix = any([
                fix.get("launch_args", ""),
                fix.get("env_vars", ""),
                fix.get("winetricks", ""),
                fix.get("custom_proton", ""),
                fix.get("kernel_tip", ""),
                fix.get("backend", ""),
                fix.get("symptom", "")
            ])
            if has_fix:
                results.append((fix, report))
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        pass
    return results

# ============================================================
# 7. MAIN PROCESSING LOOP — GPU BATCHED + CPU PARALLEL PARSE
# ============================================================
print(f"\n⚡ Processing reports {start_index} to {len(raw_reports)}...")
batch_count = 0

for i in range(start_index, len(raw_reports), BATCH_SIZE):
    batch = raw_reports[i:i+BATCH_SIZE]

    # Build prompts
    formatted_prompts = []
    for report in batch:
        prompt = PROMPT.format(
            game=report.get("title", "Unknown"),
            proton=report.get("proton_version", "?"),
            rating=report.get("rating", "?"),
            specs=report.get("specs", "?"),
            os=report.get("os", "?"),
            notes=report["notes"][:2000]  # Cap notes to prevent OOM
        )
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)

    # GPU: Generate all responses in one batched forward pass
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,      # Near-deterministic for structured output
            do_sample=True
        )

    # Decode all responses
    decoded_responses = []
    for j, output in enumerate(outputs):
        input_length = inputs["input_ids"][j].shape[0]
        response_text = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
        decoded_responses.append((response_text, batch[j]))

    # CPU: Parse all JSON responses in parallel threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = [executor.submit(parse_ai_response, resp, report) for resp, report in decoded_responses]
        for future in concurrent.futures.as_completed(futures):
            for fix, report in future.result():
                c.execute('''INSERT INTO game_fixes
                    (game_name, app_id, proton_version, user_specs, user_os,
                     rating, symptom, backend, launch_args, env_vars,
                     winetricks, custom_proton, kernel_tip, notes_raw)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (report.get("title", ""),
                     report.get("app_id", ""),
                     report.get("proton_version", ""),
                     report.get("specs", ""),
                     report.get("os", ""),
                     report.get("rating", ""),
                     fix.get("symptom", ""),
                     fix.get("backend", ""),
                     fix.get("launch_args", ""),
                     fix.get("env_vars", ""),
                     fix.get("winetricks", ""),
                     fix.get("custom_proton", ""),
                     fix.get("kernel_tip", ""),
                     report["notes"][:500]))
                total_fixes_saved += 1

    conn.commit()
    batch_count += 1

    # AUTO-SAVE to Google Drive
    if batch_count % SAVE_EVERY == 0:
        progress_data = {
            "last_processed_index": i + BATCH_SIZE,
            "total_fixes_saved": total_fixes_saved,
            "total_reports": len(raw_reports)
        }
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress_data, f)
        pct = ((i + BATCH_SIZE) / len(raw_reports)) * 100
        print(f"💾 Saved! {i+BATCH_SIZE}/{len(raw_reports)} ({pct:.1f}%) | Fixes: {total_fixes_saved}")

    elif batch_count % 10 == 0:
        pct = ((i + BATCH_SIZE) / len(raw_reports)) * 100
        print(f"⚡ {i+BATCH_SIZE}/{len(raw_reports)} ({pct:.1f}%) | Fixes: {total_fixes_saved}")

# ============================================================
# 8. FINAL SAVE
# ============================================================
conn.commit()
conn.close()

progress_data = {
    "last_processed_index": len(raw_reports),
    "total_fixes_saved": total_fixes_saved,
    "total_reports": len(raw_reports)
}
with open(PROGRESS_FILE, "w") as f:
    json.dump(progress_data, f)

print(f"\n🎉 DONE! Saved {total_fixes_saved} fixes to {DB_PATH}")
print(f"📂 knowledge.db is on your Google Drive at: My Drive/RIFT/knowledge.db")
