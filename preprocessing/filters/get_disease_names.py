import os
import re
from bs4 import BeautifulSoup

def clean_text(text):
    return re.sub(r"^→", '', text).strip()

def extract_text_from_column(col, pattern):
    text = col.get_text(strip=True)
    if not text:
        return []

    texts = text.split("、")
    avoid_pattern = r".+と同じ"
    results = set()

    for text in texts:
        if re.match(avoid_pattern, text):
            continue

        match = re.match(pattern, text)
        if match:
            before_parentheses = clean_text(match.group(1))

            if before_parentheses:
                results.add(before_parentheses)
        else:
            results.add(clean_text(text))
    return results

def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table', {'border': '1'})
    rows = table.find_all('tr')
    return rows

def main():
    with open("wiki.html") as f:
        html_content = f.read()

    rows = parse_html(html_content)
    pattern = r"^(.*?)\s*[（(](.+?)[）)]\s*$"
    disease_names = set()

    for i, row in enumerate(rows):
        if i == 0:
            continue # 見出し

        cols = row.find_all('td')
        if len(cols) <= 1:  # 見出し行のスキップ
            continue

        disease_names.update(extract_text_from_column(cols[0], pattern))

        if len(cols) >= 3:
            disease_names.update(extract_text_from_column(cols[2], pattern))
        if len(cols) >= 4:
            disease_names.update(extract_text_from_column(cols[3], pattern))
        if len(cols) >= 6:
            disease_names.update(extract_text_from_column(cols[5], pattern))

    # 結果を出力
    for name in disease_names:
        print(name)

if __name__ == "__main__":
    main()


