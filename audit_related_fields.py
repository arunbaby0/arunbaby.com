import os
import re
import yaml

directories = {
    '_dsa': 'related_dsa_day',
    '_ml_system_design': 'related_ml_day',
    '_speech_tech': 'related_speech_day',
    '_ai_agents': 'related_agents_day'
}

# Map directory to the key that SHOULD be missing (the self reference)
self_ref = {
    '_dsa': 'related_dsa_day',
    '_ml_system_design': 'related_ml_day',
    '_speech_tech': 'related_speech_day',
    '_ai_agents': 'related_agents_day'
}

all_keys = set(directories.values())

def parse_front_matter(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return None, "No front matter"
    
    try:
        fm = yaml.safe_load(match.group(1))
        return fm, None
    except Exception as e:
        return None, str(e)

base_dir = "/home/ubuntu/arunbaby.com"

print("Auditing Related Day Fields (Days 1-60)...")
print("-" * 60)

issues_found = []

# Collect all files first to map Day -> File per track
# This is needed if we want to confirm file existence, but for now we just scan what exists.
# We iterate 1..60 and try to find the files.

for day in range(1, 61):
    day_str = f"{day:04d}"
    
    # We need to find the files for this day. Since filename has title, we glob or search.
    # Optimization: pre-scan directories.
    pass

# Pre-scan
files_by_day = {} # day_int -> { dir: filepath }

for dirname in directories.keys():
    dirpath = os.path.join(base_dir, dirname)
    if not os.path.exists(dirpath):
        print(f"Directory not found: {dirpath}")
        continue
        
    for filename in os.listdir(dirpath):
        if not filename.endswith(".md"):
            continue
        
        # Assume valid files start with 4 digits.
        if not filename[:4].isdigit():
            continue
            
        try:
            day_num = int(filename[:4])
        except:
            continue
            
        if 1 <= day_num <= 60:
            if day_num not in files_by_day:
                files_by_day[day_num] = {}
            files_by_day[day_num][dirname] = os.path.join(dirpath, filename)

# Now Audit
for day in range(1, 61):
    if day not in files_by_day:
        # print(f"Day {day}: No files found.")
        continue
        
    for dirname, filepath in files_by_day[day].items():
        fm, error = parse_front_matter(filepath)
        if error:
            issues_found.append(f"[Day {day}] [{dirname}] Parse Error: {error} in {os.path.basename(filepath)}")
            continue
            
        current_keys = set()
        for k in all_keys:
            if k in fm:
                current_keys.add(k)
        
        # Expected keys: All keys EXCEPT check self_ref[dirname]
        expected_keys = all_keys - {self_ref[dirname]}
        
        missing = expected_keys - current_keys
        
        if missing:
            issues_found.append(f"[Day {day}] [{dirname}] Missing {missing} in {os.path.basename(filepath)}")

if not issues_found:
    print("SUCCESS: All posts have correct related_day fields.")
else:
    print(f"Found issues in {len(issues_found)} files:")
    for issue in issues_found:
        print(issue)
