import httpx
from bs4 import BeautifulSoup
import re

url = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679"
r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)

# handle meta refresh
if '<meta http-equiv="refresh"' in r.text.lower():
    soup = BeautifulSoup(r.text, "lxml")
    meta = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
    refresh_url = meta["content"].split("url=")[-1].strip("'\"")
    if not refresh_url.startswith("http"):
        refresh_url = "https://eur-lex.europa.eu" + refresh_url
    r = httpx.get(refresh_url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)

soup = BeautifulSoup(r.text, "lxml")

lang = "EN"
article_num = "15"
target_str = f"Article {article_num}"

start_node = None
for tag in soup.find_all(["p", "span", "div", "h2", "h3"]):
    text = tag.get_text(strip=True)
    # Use search instead of startswith to handle invisible characters
    if re.search(r"^Article\s+" + article_num + r"\b", text, re.IGNORECASE):
        start_node = tag
        break

if start_node:
    curr = start_node
    while curr.name in ['span', 'strong', 'a']:
        curr = curr.parent

    # Use separator='\n\n' to preserve block separation
    content = [curr.get_text(separator='\n', strip=True)]
    node = curr.find_next_sibling()
    while node:
        text = node.get_text(strip=True)
        if re.search(r"^Article\s+\d+\b", text, re.IGNORECASE):
            break
        if text:
            content.append(node.get_text(separator='\n', strip=True))
        node = node.find_next_sibling()
        
    print(f"--- Article {article_num} ---")
    print("\n\n".join(content)[:1000])
else:
    print("Not found")
