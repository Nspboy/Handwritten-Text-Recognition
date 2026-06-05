import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib import colors

def create_appendix():
    doc = SimpleDocTemplate(
        "Thesis_Appendix.pdf",
        pagesize=letter,
        rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'AppendixTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=20, alignment=1
    )
    heading_style = ParagraphStyle(
        'AppendixHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=15
    )
    code_style = ParagraphStyle(
        'CodeStyle', parent=styles['Code'], fontSize=8, leading=10, 
        fontName='Courier', textColor=colors.darkblue
    )
    log_style = ParagraphStyle(
        'LogStyle', parent=styles['Code'], fontSize=9, leading=11, 
        fontName='Courier', textColor=colors.black
    )
    
    story = []
    
    # ------------------ APPENDIX A: Source Code ------------------
    story.append(Paragraph("APPENDIX A: SYSTEM SOURCE CODE", title_style))
    story.append(Spacer(1, 20))
    
    py_files = [f for f in os.listdir('.') if f.endswith('.py') and f not in ['pdf_scanner.py', 'fix_pdf.py', 'toc_extractor.py', 'toc_filler.py', 'extend_pdf.py', 'build_bookmarks.py', 'generate_appendix.py']]
    
    for pf in py_files:
        story.append(Paragraph(f"File: {pf}", heading_style))
        try:
            with open(pf, 'r', encoding='utf-8') as f:
                code_text = f.read()
            # Replace characters that might break ReportLab XML parsing
            code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Preformatted(code_text, code_style))
            story.append(PageBreak())
        except Exception as e:
            pass

    # ------------------ APPENDIX B: Pipeline Results ------------------
    story.append(Paragraph("APPENDIX B: DETAILED PIPELINE CONFIGURATION AND RESULTS", title_style))
    story.append(Spacer(1, 20))
    
    if os.path.exists("pipeline_results.json"):
        with open("pipeline_results.json", "r") as f:
            data = f.read()
            data = data.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Preformatted(data, code_style))
    else:
        story.append(Paragraph("No pipeline_results.json found.", styles['Normal']))
        
    doc.build(story)
    print("Appendix PDF generated.")

if __name__ == "__main__":
    create_appendix()
