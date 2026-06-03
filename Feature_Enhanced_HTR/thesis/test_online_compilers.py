import os
import tarfile
import requests

def test_endpoints():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tarball_bz2 = os.path.join(script_dir, "test.tar.bz2")
    tarball_gz = os.path.join(script_dir, "test.tar.gz")
    
    # Create test.tar.bz2
    with tarfile.open(tarball_bz2, "w:bz2") as tar:
        for item in os.listdir(script_dir):
            if item in ['compile_thesis.py', 'compile_online.py', 'thesis.tar.gz', 'thesis.tar.bz2', 'test.tar.bz2', 'test.tar.gz', '.git', '__pycache__']:
                continue
            tar.add(os.path.join(script_dir, item), arcname=item)
            
    # Create test.tar.gz
    with tarfile.open(tarball_gz, "w:gz") as tar:
        for item in os.listdir(script_dir):
            if item in ['compile_thesis.py', 'compile_online.py', 'thesis.tar.gz', 'thesis.tar.bz2', 'test.tar.bz2', 'test.tar.gz', '.git', '__pycache__']:
                continue
            tar.add(os.path.join(script_dir, item), arcname=item)

    endpoints = [
        ("https://latexonline.cc/data", tarball_bz2, "bz2"),
        ("https://latexonline.cc/data", tarball_gz, "gz"),
        ("https://texlive2020.latexonline.cc/data", tarball_bz2, "bz2"),
        ("https://texlive2020.latexonline.cc/data", tarball_gz, "gz"),
    ]
    
    for url_base, path, fmt in endpoints:
        url = f"{url_base}?target=main.tex"
        print(f"\nTesting endpoint: {url} with {fmt} file...")
        try:
            with open(path, 'rb') as f:
                r = requests.post(url, files={'file': f}, timeout=30)
            print(f"Status Code: {r.status_code}")
            print(f"Content Type: {r.headers.get('content-type')}")
            print(f"Response starts with: {r.content[:100]}")
        except Exception as e:
            print(f"Failed: {e}")
            
    # Clean up
    if os.path.exists(tarball_bz2):
        os.remove(tarball_bz2)
    if os.path.exists(tarball_gz):
        os.remove(tarball_gz)

if __name__ == '__main__':
    test_endpoints()
