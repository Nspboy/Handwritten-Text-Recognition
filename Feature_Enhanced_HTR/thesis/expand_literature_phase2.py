lit_file = "chapters/literature_survey.tex"

content = """
\\section{Exhaustive Dataset History and Corpus Analysis}
The success of deep learning in handwriting recognition is fundamentally tied to the availability of large, annotated datasets. A model's ability to generalize to unseen handwriting styles depends entirely on the variance and distribution of the training corpus. In this section, we provide a deep historical analysis of the primary datasets driving the field.

\\subsection{The IAM Handwriting Database and the LOB Corpus}
The IAM Handwriting Database, compiled by the Research Group on Computer Vision and Artificial Intelligence at the University of Bern, remains the most widely cited benchmark for English offline HTR. The underlying text is sourced from the Lancaster-Oslo/Bergen (LOB) corpus, which itself was compiled in the 1970s as a British English counterpart to the American Brown Corpus.

The LOB corpus contains 500 texts distributed across 15 genres, ensuring a rich linguistic distribution of n-grams. The IAM database creators tasked 657 distinct writers to transcribe portions of the LOB corpus. The writers utilized unconstrained, natural cursive styles. The forms were scanned at 300 DPI, yielding 1,539 pages of scanned text. 

Through semi-automated segmentation algorithms utilizing projection profiles and connected component analysis, the database was split into:
\\begin{itemize}
    \\item 13,353 isolated and labeled text lines.
    \\item 115,320 isolated and labeled words.
\\end{itemize}
The presence of 657 distinct writers makes the IAM database particularly challenging. A model trained on this dataset must learn writer-independent features rather than overfitting to the idiosyncratic loops and slants of a specific individual.

\\subsection{Comparative Dataset Analysis: RIMES, Bentham, and George Washington}
While IAM represents modern British English, other datasets provide necessary historical and linguistic diversity:

\\textbf{RIMES (Reconnaissance et Indexation de donnees Manuscrites et de fac similES):} Created to simulate the processing of incoming French mail, the RIMES dataset contains over 12,000 letters written by 1,300 volunteers. Unlike IAM's literary paragraphs, RIMES focuses on administrative terminology, dates, and postal codes, presenting a completely different Zipfian distribution of vocabulary.

\\textbf{The Bentham Collection:} Sourced from the manuscripts of the renowned English philosopher Jeremy Bentham (1748–1832). This dataset is critical for historical document processing. The paper is heavily degraded with ink bleed-through (show-through), and the text contains archaic ligatures, abbreviations (e.g., \\textit{\\&c} for \\textit{etcetera}), and crossed-out words. Models trained purely on modern IAM fail catastrophically on Bentham due to these severe domain shifts.

\\textbf{The George Washington (GW) Database:} A small but seminal dataset containing 20 pages of letters written by George Washington and his associates in 1755. It contains only 4,894 words but features highly consistent, calligraphic 18th-century English cursive. Due to its small size, GW is typically used to benchmark Transfer Learning and Few-Shot Adaptation techniques, where a network is pre-trained on IAM and fine-tuned on GW.
"""

with open(lit_file, "a", encoding="utf-8") as out_f:
    out_f.write(content)

print("Expanded literature with phase 2 dataset history.")
