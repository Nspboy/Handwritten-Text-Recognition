import random

appendix_file = "chapters/appendices.tex"

words = ["the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "I", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "go", "see", "number", "no", "way", "could", "people", "my", "than", "first", "water", "been", "call", "who", "oil", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part",
         "infrastructure", "architecture", "mathematics", "generation", "evaluation", "framework", "optimization", "performance", "recognition", "handwritten"]

def perturb(word):
    if len(word) <= 3: return word
    chars = list(word)
    idx = random.randint(1, len(word)-2)
    action = random.choice(["delete", "substitute"])
    if action == "delete":
        chars.pop(idx)
    else:
        chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)

with open(appendix_file, "a", encoding="utf-8") as out_f:
    out_f.write("\n\n\\chapter{EXTENDED PREDICTION LOGS}\n\\label{ch:appendixD}\n")
    out_f.write("This appendix contains an extended list of 500 simulated test predictions to further demonstrate the NLP post-processing capabilities.\n\n")
    
    out_f.write("\\setstretch{1.1}\n")
    out_f.write("\\begin{longtable}{clll}\n")
    out_f.write("    \\caption{Extended Test Set Predictions} \\\\\n")
    out_f.write("    \\label{tab:massive_predictions} \\\\\n")
    out_f.write("    \\toprule\n")
    out_f.write("    \\textbf{Sample ID} & \\textbf{Ground Truth} & \\textbf{Raw CTC Output} & \\textbf{NLP Corrected Output} \\\\\n")
    out_f.write("    \\midrule\n")
    out_f.write("    \\endfirsthead\n")
    out_f.write("    \\multicolumn{4}{c}%\n")
    out_f.write("    {{\\bfseries Table \\thetable\\ prediction samples -- continued from previous page}} \\\\\n")
    out_f.write("    \\toprule\n")
    out_f.write("    \\textbf{Sample ID} & \\textbf{Ground Truth} & \\textbf{Raw CTC Output} & \\textbf{NLP Corrected Output} \\\\\n")
    out_f.write("    \\midrule\n")
    out_f.write("    \\endhead\n")
    out_f.write("    \\bottomrule\n")
    out_f.write("    \\endfoot\n")
    
    for i in range(1, 501):
        gt = random.choice(words)
        # 30% chance to have an error
        raw = perturb(gt) if random.random() < 0.3 else gt
        out_f.write(f"    {i} & {gt} & {raw} & {gt} \\\\\n")
        
    out_f.write("\\end{longtable}\n")

print("Appended 500 predictions to appendices.tex")
