"""Script to scrape missing regulatory articles from EUR-Lex and save as formatted .txt files."""

import httpx
from bs4 import BeautifulSoup
import re
import os
import time

CELEX_MAP = {
    "gdpr": "32016R0679",
    "eu_ai_act": "32024R1689"
}

MISSING_ARTICLES = {
    "gdpr": ["9", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "32", "44", "45", "46", "47", "48", "49"],
    "eu_ai_act": ["78", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113"]
}

def get_eurlex_html(celex: str, lang: str = "EN") -> str:
    url = f"https://eur-lex.europa.eu/legal-content/{lang}/TXT/?uri=CELEX:{celex}"
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        r = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        # Check for meta refresh (EUR-Lex 202 Accepted behavior)
        if '<meta http-equiv="refresh"' in r.text.lower():
            soup = BeautifulSoup(r.text, "lxml")
            meta = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
            if meta and meta.get("content"):
                refresh_url = meta["content"].split("url=")[-1].strip("'\"")
                if not refresh_url.startswith("http"):
                    refresh_url = "https://eur-lex.europa.eu" + refresh_url
                print(f"Following meta refresh to {refresh_url}")
                r = client.get(refresh_url, headers={"User-Agent": "Mozilla/5.0"})
                
        return r.text

def extract_article(html: str, article_num: str, lang: str = "EN") -> tuple[str, str]:
    """Returns (title, content)"""
    soup = BeautifulSoup(html, "lxml")
    
    target_pattern = r"^Article\s+" + article_num + r"(?![0-9])" if lang == "EN" else r"^Artikel\s+" + article_num + r"(?![0-9])"
    
    start_node = None
    for tag in soup.find_all(["p", "span", "div", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if re.search(target_pattern, text, re.IGNORECASE):
            start_node = tag
            break
                
    if not start_node:
        return f"[Title Not Found]", f"[Article {article_num} not found in EUR-Lex text]"
        
    curr = start_node
    while curr.name in ['span', 'strong', 'a']:
        if curr.parent:
            curr = curr.parent
        else:
            break
            
    content = []
    
    # Try to extract the title from the same or next few blocks
    title_text = curr.get_text(separator=' ', strip=True)
    if len(title_text) > 15:
        title = title_text.replace("\n", " ")
    else:
        # Title might be in the next node
        next_sib = curr.find_next_sibling()
        if next_sib and len(next_sib.get_text(strip=True)) > 0 and not re.match(r"^\d+\.", next_sib.get_text(strip=True)):
            title = f"{title_text} — {next_sib.get_text(separator=' ', strip=True).replace(chr(10), ' ')}"
        else:
            title = title_text

    content.append(curr.get_text(separator='\n', strip=True))
    
    node = curr.find_next_sibling()
    while node:
        text = node.get_text(strip=True)
        
        # Stop at next article
        if lang == "EN" and re.search(r"^Article\s+\d+(?![0-9])", text, re.IGNORECASE):
            break
        if lang == "DE" and re.search(r"^Artikel\s+\d+(?![0-9])", text, re.IGNORECASE):
            break
            
        if text:
            # Preserve newlines but remove excessive spacing
            node_text = node.get_text(separator='\n', strip=True)
            node_text = re.sub(r'\n{3,}', '\n\n', node_text)
            content.append(node_text)
            
        node = node.find_next_sibling()
        
    return title, "\n\n".join(content)

def main():
    base_dir = os.path.join("data", "regulatory")
    
    for reg, articles in MISSING_ARTICLES.items():
        print(f"=== Fetching {reg.upper()} ===")
        celex = CELEX_MAP[reg]
        
        print("Downloading EN HTML...")
        html_en = get_eurlex_html(celex, "EN")
        print("Downloading DE HTML...")
        html_de = get_eurlex_html(celex, "DE")
        
        out_dir = os.path.join(base_dir, reg)
        os.makedirs(out_dir, exist_ok=True)
        
        for art in articles:
            print(f"Processing Article {art}...")
            title_en, content_en = extract_article(html_en, art, "EN")
            title_de, content_de = extract_article(html_de, art, "DE")
            
            # Format the output string
            file_content = f"""ARTICLE: {art}
REGULATION: {reg}
DOMAIN: article_{art}
TITLE_EN: {title_en}
TITLE_DE: {title_de}

=== EN ===
{content_en}

=== DE ===
{content_de}
"""
            file_path = os.path.join(out_dir, f"article_{art}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            print(f"Saved {file_path}")
            
if __name__ == "__main__":
    main()
