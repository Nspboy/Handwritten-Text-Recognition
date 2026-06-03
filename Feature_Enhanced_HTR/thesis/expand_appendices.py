import os

files_to_add = [
    "../app.py",
    "../generate_paper.py",
    "../train_full_pipeline.py",
    "../verify_model.py",
    "../debug_preds.py"
]

appendix_file = "chapters/appendices.tex"

with open(appendix_file, "a", encoding="utf-8") as out_f:
    out_f.write("\n\n\\chapter{EXTENDED SOURCE CODE LISTINGS}\n\\label{ch:appendixC}\n")
    out_f.write("This appendix contains the complete source code for the remaining components of the architecture, including the Flask web application, the synthetic paper generation scripts, model verification, and debugging utilities.\n\n")

    for fpath in files_to_add:
        name = os.path.basename(fpath)
        safe_name = name.replace('_', '\\_')
        out_f.write(f"\\section{{{safe_name}}}\n")
        out_f.write(f"\\begin{{lstlisting}}[language=Python, style=pythoncode, caption={{{safe_name} Implementation}}]\n")
        
        with open(fpath, "r", encoding="utf-8") as in_f:
            out_f.write(in_f.read())
            
        out_f.write("\n\\end{lstlisting}\n\n\\newpage\n")

print("Appended python files to appendices.tex")
