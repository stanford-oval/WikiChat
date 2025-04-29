import requests
from bs4 import BeautifulSoup

word_frequency_page = (
    "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/English/Wikipedia_(2016)"
)

word_list = []

html = requests.get(word_frequency_page).text
soup = BeautifulSoup(html, features="lxml")
for header in soup.find_all("h2"):
    if header.text and header.text.startswith("1-1000"):
        # print(header.text)
        for tag in header.parent.next_siblings:
            if tag.name == "p":
                words = tag.find_all("a")
                for w in words:
                    w = w["href"]
                    if "redlink=1" in w:
                        continue
                    if w.startswith("/wiki/"):
                        w = w[len("/wiki/") :]
                    if w.endswith("#English"):
                        w = w[: -len("#English")]
                    w = w.replace("_", " ")
                    word_list.append(w)

print("Extracted %d words from %s" % (len(word_list), word_frequency_page))
with open("./preprocessing/word_list.txt", "w") as f:
    f.write("\n".join(word_list))
