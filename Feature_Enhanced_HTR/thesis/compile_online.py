import os
import tarfile
import requests
import sys

def make_tarfile(output_filename, source_dir):
    print(f"Creating bzip2 tarball {output_filename} from {source_dir}...")
    with tarfile.open(output_filename, "w:bz2") as tar:
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            # Exclude compile scripts, python scripts, tarballs
            if item in ['compile_thesis.py', 'compile_online.py', 'thesis.tar.gz', 'thesis.tar.bz2', '.git', '__pycache__']:
                continue
            tar.add(item_path, arcname=item)
    print("Tarball created successfully.")

def compile_latex(tarball_path, target_file="main.tex", output_pdf="htr_thesis.pdf"):
    print(f"Sending tarball to LaTeX.Online /data endpoint for compiling target '{target_file}'...")
    
    # Use the active and configured latexonline server
    host = "https://latexonline.cc"
    url = f"{host}/data?target={target_file}"
    
    try:
        with open(tarball_path, 'rb') as f:
            # -F file=@$tarball
            response = requests.post(url, files={'file': f}, timeout=180)
            
        if response.status_code == 200:
            if response.headers.get('content-type') == 'application/pdf' or response.content.startswith(b'%PDF'):
                with open(output_pdf, 'wb') as out_f:
                    out_f.write(response.content)
                print(f"Compilation success! PDF saved to: {output_pdf}")
                return True
            else:
                print("Error: The server did not return a PDF file.")
                print("Server response head:")
                print(response.content[:1000].decode('utf-8', errors='ignore'))
                return False
        else:
            print(f"Error compiling: HTTP {response.status_code}")
            print(response.text[:2000])
            return False
            
    except requests.exceptions.Timeout:
        print("Error: The request timed out. The compilation took too long.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    tarball = os.path.join(script_dir, "thesis.tar.bz2")
    output_pdf = os.path.join(project_dir, "htr_thesis.pdf")
    
    make_tarfile(tarball, script_dir)
    
    success = compile_latex(tarball, "main.tex", output_pdf)
    
    # Clean up local tarball
    if os.path.exists(tarball):
        os.remove(tarball)
        print("Cleaned up temporary tarball.")
        
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()
