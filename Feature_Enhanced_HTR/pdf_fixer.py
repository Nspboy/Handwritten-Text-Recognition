import PyPDF2
import sys

def main():
    input_pdf = 'Thesis 2026.pdf'
    output_pdf = 'Thesis 2026_fixed.pdf'
    
    print("Reading PDF...")
    reader = PyPDF2.PdfReader(input_pdf)
    writer = PyPDF2.PdfWriter()
    
    # 1. Add all pages except trailing blank pages
    non_blank_pages = []
    
    # Check backwards to find where non-blank pages end
    last_non_blank = len(reader.pages) - 1
    for i in range(len(reader.pages) - 1, -1, -1):
        text = reader.pages[i].extract_text()
        if text and text.strip():
            last_non_blank = i
            break
            
    print(f"Total pages: {len(reader.pages)}. Last non-blank page is index {last_non_blank}.")
    
    for i in range(last_non_blank + 1):
        writer.add_page(reader.pages[i])
        
    print(f"Added {last_non_blank + 1} pages to writer.")

    # 2. Add Outlines (Bookmarks)
    # We will map known titles to their 0-indexed page numbers based on our previous scan
    # Certificate: 3, Declaration: 4, Acknowledgements: 5, Abstract: 6
    # TOC: 11, Chap1: 12, Chap2: 17, Chap3: 31, Chap4: 45, Chap5: 58, Refs: 61
    
    # Add outlines to writer
    writer.add_outline_item("CERTIFICATE", 3)
    writer.add_outline_item("DECLARATION", 4)
    writer.add_outline_item("ACKNOWLEDGEMENTS", 5)
    writer.add_outline_item("ABSTRACT", 6)
    writer.add_outline_item("TABLE OF CONTENTS", 11)
    
    chap1 = writer.add_outline_item("CHAPTER 1: INTRODUCTION", 12)
    writer.add_outline_item("1.1 Introduction", 12, parent=chap1)
    writer.add_outline_item("1.2 Problem Statement of the Thesis", 13, parent=chap1) # estimated
    writer.add_outline_item("1.3 Objectives of the Thesis", 14, parent=chap1) # estimated
    writer.add_outline_item("1.4 Organization of the Thesis", 15, parent=chap1) # estimated
    
    chap2 = writer.add_outline_item("CHAPTER 2: LITERATURE SURVEY", 17)
    
    chap3 = writer.add_outline_item("CHAPTER 3: METHODOLOGY", 31)
    
    chap4 = writer.add_outline_item("CHAPTER 4: RESULTS AND DISCUSSION", 45)
    
    chap5 = writer.add_outline_item("CHAPTER 5: CONCLUDING REMARKS", 58)
    
    writer.add_outline_item("REFERENCES", 61)

    print("Writing modified PDF...")
    with open(output_pdf, 'wb') as f:
        writer.write(f)
    print("Done! Saved to Thesis 2026_fixed.pdf.")

if __name__ == "__main__":
    main()
