import argparse
import re

import requests
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wikipedia_date", type=str, required=True, help="Enter the date in the format yyyymmdd"
)
args = parser.parse_args()

response = requests.get(
    f"https://dumps.wikimedia.org/other/enterprise_html/runs/{args.wikipedia_date}/"
)

# Check if the request was successful
if response.status_code == 200:
    # Get the HTML content of the webpage
    html_content = response.text
else:
    print("Failed to retrieve the webpage")

# Create a BeautifulSoup object
soup = BeautifulSoup(html_content, "html.parser")

# Regex pattern to match the file names
pattern = re.compile(rf"\w+wiki-NS0-{args.wikipedia_date}-ENTERPRISE-HTML.json.tar.gz")

# Find all <a> tags with href matching the pattern
all_dumps = soup.find("a", href=pattern)
total_size = 0
text = all_dumps.parent.get_text()
for line in text.split("\n"):
    if f"wiki-NS0-{args.wikipedia_date}-ENTERPRISE-HTML.json.tar.gz" in line:
        dump, _, _, size_in_bytes = tuple(line.split())
        dump = dump[
            : dump.find(f"wiki-NS0-{args.wikipedia_date}-ENTERPRISE-HTML.json.tar.gz")
        ]
        size_in_gb = int(size_in_bytes) / 1024 / 1024 / 1024
        total_size += size_in_gb
        print(f"{dump}: {size_in_gb:.2f}")

print(f"Total Wikipedia size in GB: {total_size:.2f}")
