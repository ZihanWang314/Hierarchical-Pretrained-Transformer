import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urlparse, urljoin
import json
import os
import random
import string
from tqdm import tqdm

def rqget(url):
    proxies = [{'http': 'http://121.13.252.58:41564'}]
    proxies = random.choice(proxies)
    return requests.get(url, proxies=proxies)
    
def retrieve_text_from_url(url):
    text = rqget(url).text
    with open("tmp2.html", 'w') as file:
        file.write(bs(text, features='lxml').prettify())
    return text

def parse_document_from_text(url_text, url):
    if url.count('/') != 1:
        return None, []
    else:
        soup = bs(url_text, features='lxml')
        texts = []
        headline = soup.find('h1')
        if headline == None:
            return None, []
        headline = headline.text.strip()
        doc_tag = soup.find('div', {'id': 'guide-contents'})
        if not doc_tag:
            doc_tags = soup.find_all('div', {'class': 'govuk-grid-column-two-thirds'})
            for tag in doc_tags:
                if len(tag.find_all('p')) > 5:
                    doc_tag = tag
                    break
        if doc_tag:
            for tag in doc_tag.find_all():
                if tag.name in ['h1', 'h2', 'h3', 'h4', 'li', 'tr', 'p']:
                    #process to raw text
                    if tag.name == 'tr':
                        text = (tag.name, ' | '.join(tag.text.strip('\n').split('\n')))
                    else:
                        text = (tag.name, ''.join(tag.text.strip('\n').strip().split('\n')[:1]))
                    # process the obtained raw text
                    if text[1] == 'Explore the topic' or text[1] == 'Related content':
                        break
                    text = '<' + text[0] + '>' + text[1] + '</' + text[0] + '>'
                    texts.append(text)
        return headline, texts

def analyze_new_urls_from_text(url_text):
    soup = bs(url_text, features='lxml')
    hrefs = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href is None:
            continue
        href = urljoin(url, href)
        hrefs.append(href)
    return list(set(hrefs))

def generate_random_string():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=30))


if not os.path.exists('data'):
    os.mkdir('data')
if os.path.exists('current_list.json'):
    weburl_list = json.load(open('current_list.json'))
else:
    weburl_list = ['https://www.gov.uk']

if os.path.exists('done_list.json'):
    weburl_done = set(json.load(open('done_list.json')))
else:
    weburl_done = set()

    
for i in tqdm(range(5000), desc = 'valid pages crawlled'):
    if weburl_list:
        url = weburl_list.pop(0)
        while url in weburl_done:
            url = weburl_list.pop(0)
        url_text = retrieve_text_from_url(url)
        qa_headline, qa_content = parse_document_from_text(url_text, url)

        new_urls = analyze_new_urls_from_text(url_text)
        weburl_list += new_urls
        weburl_done.add(url)
        if qa_content != []:
            with open(os.path.join('data', generate_random_string()), 'w') as file:
                json.dump({'title': qa_headline, 'url': url, 'content': qa_content}, file)
        
        if i % 10 == 0:
            print('crawlled', i, 'webpages.')
            print('saving current list to current_list.json')
            json.dump(weburl_list, open('current_list.json', 'w'))
            print('saving crawlled list to done_list.json')
            json.dump(list(weburl_done), open('done_list.json', 'w'))


