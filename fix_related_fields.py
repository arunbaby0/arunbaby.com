import os
import re

directories = {
    '_dsa': 'related_dsa_day',
    '_ml_system_design': 'related_ml_day',
    '_speech_tech': 'related_speech_day',
    '_ai_agents': 'related_agents_day'
}

# The key that corresponds to the directory (to be excluded)
self_ref = {
    '_dsa': 'related_dsa_day',
    '_ml_system_design': 'related_ml_day',
    '_speech_tech': 'related_speech_day',
    '_ai_agents': 'related_agents_day'
}

base_dir = "/home/ubuntu/arunbaby.com"

def get_related_fields_block(day_num, directory_name):
    fields = []
    for dir_key, field_name in directories.items():
        if field_name != self_ref[directory_name]:
            fields.append(f"{field_name}: {day_num}")
    return "\n".join(fields)

print("Fixing Related Day Fields (Days 1-60)...")

count = 0

for day in range(1, 61):
    for dirname in directories.keys():
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.exists(dirpath):
            continue
            
        # Find the file for this day
        # We look for files starting with {day:04d}
        found_file = None
        for filename in os.listdir(dirpath):
            if filename.startswith(f"{day:04d}") and filename.endswith(".md"):
                found_file = os.path.join(dirpath, filename)
                break
        
        if not found_file:
            continue
            
        # Read content
        with open(found_file, 'r') as f:
            content = f.read()
            
        # Check if already has related fields (simple check)
        # We know from audit they are missing, but let's be safe.
        # We only want to inject if valid "day: X" exists and fields are missing.
        
        # Regex to find 'day: X' in front matter
        # Note: day might be "day: 1" or "day: 01"
        match = re.search(r'^day:\s*(\d+)', content, re.MULTILINE)
        if not match:
            print(f"Skipping {found_file}: 'day:' field not found.")
            continue
            
        day_val = match.group(1) # e.g. "1" or "55"
        
        # Check if fields exist
        # If 'related_dsa_day' matches, we assume it's there.
        # The audit showed they are missing.
        
        # We will insert AFTER the day line.
        insert_text = get_related_fields_block(day_val, dirname)
        
        # Check if insert_text keys are already present
        needs_update = False
        for line in insert_text.split('\n'):
            key = line.split(':')[0]
            if not re.search(f"^{key}:", content, re.MULTILINE):
                needs_update = True
                break
        
        if not needs_update:
            continue
            
        # Perform insertion
        new_content = re.sub(r'^(day:\s*\d+)', f"\\1\n{insert_text}", content, count=1, flags=re.MULTILINE)
        
        with open(found_file, 'w') as f:
            f.write(new_content)
            
        print(f"Fixed {found_file}")
        count += 1

print(f"Total files fixed: {count}")
