import os
import re

directories = [
    '_dsa',
    '_ml_system_design',
    '_speech_tech',
    '_ai_agents'
]

base_dir = "/home/ubuntu/arunbaby.com"
threshold = 3000

print(f"{'File':<60} | {'Words':>5} | {'Day':>3}")
print("-" * 75)

low_word_count_files = []

for dirname in directories:
    dirpath = os.path.join(base_dir, dirname)
    if not os.path.exists(dirpath):
        continue
        
    for filename in sorted(os.listdir(dirpath)):
        if not filename.endswith(".md"):
            continue
            
        filepath = os.path.join(dirpath, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Strip front matter for more accurate content word count
        # Front matter is between the first two ---
        content_body = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        
        # Simple word count
        words = len(content_body.split())
        
        if words < threshold:
            day_match = re.search(r'day:\s*(\d+)', content)
            day = int(day_match.group(1)) if day_match else -1
            
            low_word_count_files.append((filepath, words, day))

# Sort by Day, then Directory
low_word_count_files.sort(key=lambda x: x[2])

for filepath, count, day in low_word_count_files:
    rel_path = os.path.relpath(filepath, base_dir)
    print(f"{rel_path:<60} | {count:>5} | {day:>3}")

print("-" * 75)
print(f"Total files under {threshold} words: {len(low_word_count_files)}")
