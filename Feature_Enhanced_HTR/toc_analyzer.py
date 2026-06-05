import fitz
import json

doc = fitz.open('Thesis 2026.pdf')
page = doc[10] # 11th page (0-indexed)
blocks = page.get_text('dict')['blocks']
out = []
for b in blocks:
    if 'lines' in b:
        for l in b['lines']:
            line_text = ''.join([s['text'] for s in l['spans']]).strip()
            # print y-coord and text to figure out what y matches what 'x'
            out.append(f"Y={l['bbox'][1]:.1f} | X={l['bbox'][0]:.1f} | {line_text}")

with open("toc_analysis.txt", "w") as f:
    f.write("\n".join(out))
print("Analysis written to toc_analysis.txt")
