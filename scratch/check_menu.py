import glob
import re

for f in glob.glob(".venv/Lib/site-packages/streamlit/static/static/js/*.js"):
    text = open(f, encoding="utf-8", errors="ignore").read()
    for m in re.finditer(r'(getHelp|reportABug|aboutSectionMd|about_section_md|get_help_url)', text):
        idx = m.start()
        print(text[max(0, idx-100):min(len(text), idx+150)])
        print("="*40)
