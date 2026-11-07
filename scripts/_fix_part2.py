import re

with open('scripts/comprehensive_analysis_part2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add safe_float/safe_int after imports
insert_after = "                      'axes.grid': True, 'grid.alpha': 0.3})"
safe_code = '''

import re as _re
def safe_float(v):
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    m = _re.match(r'tensor\\(([\\d.eE+-]+)\\)', s)
    if m: return float(m.group(1))
    return float(s)

def safe_int(v):
    return int(safe_float(v))
'''

content = content.replace(insert_after, insert_after + safe_code, 1)

# Replace float/int calls on data variables
lines = content.split('\n')
new_lines = []
for line in lines:
    if 'def safe_float' in line or 'def safe_int' in line or 'return float' in line or 'return int' in line:
        new_lines.append(line)
        continue
    line = re.sub(r'\bfloat\(inst\[', 'safe_float(inst[', line)
    line = re.sub(r'\bfloat\(i\[', 'safe_float(i[', line)
    line = re.sub(r'\bfloat\(r\[', 'safe_float(r[', line)
    line = re.sub(r'\bint\(inst\[', 'safe_int(inst[', line)
    line = re.sub(r'\bint\(i\[', 'safe_int(i[', line)
    line = re.sub(r'\bint\(r\[', 'safe_int(r[', line)
    new_lines.append(line)

with open('scripts/comprehensive_analysis_part2.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

print('Part2 fixed successfully')
