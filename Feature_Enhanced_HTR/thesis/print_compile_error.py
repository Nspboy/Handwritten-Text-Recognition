import os
import tarfile
import requests

def get_compile_error():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tarball = os.path.join(script_dir, "test.tar.bz2")
    
    with tarfile.open(tarball, "w:bz2") as tar:
        for item in os.listdir(script_dir):
            if item in ['compile_thesis.py', 'compile_online.py', 'thesis.tar.gz', 'thesis.tar.bz2', 'test.tar.bz2', 'print_compile_error.py', 'test_online_compilers.py', '.git', '__pycache__']:
                continue
            tar.add(os.path.join(script_dir, item), arcname=item)

    url = "https://latexonline.cc/data?target=main.tex"
    try:
        with open(tarball, 'rb') as f:
            r = requests.post(url, files={'file': f}, timeout=60)
        print(f"Status Code: {r.status_code}")
        print("\n--- Response Content (From 4000 to end) ---")
        # Print from 4000 onwards
        print(r.content[4000:].decode('utf-8', errors='ignore'))
    except Exception as e:
        print(f"Failed: {e}")
        
    if os.path.exists(tarball):
        os.remove(tarball)

if __name__ == '__main__':
    get_compile_error()
