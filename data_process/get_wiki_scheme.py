import time

from tqdm import tqdm
import socks
import socket
from SPARQLWrapper import SPARQLWrapper, JSON
from wikidata.client import Client

# 配置 SOCKS 代理
socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 7891)
socket.socket = socks.socksocket

# 设置 SPARQL 端点 URL
endpoint_url = "https://query.wikidata.org/sparql"

# 创建 SPARQLWrapper 对象
sparql = SPARQLWrapper(endpoint_url)
headers = {
    'Accept': 'application/sparql-results+json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
sparql.addCustomHttpHeader("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

import requests

API_ENDPOINT = "https://www.wikidata.org/w/api.php"

def get_wikidata_id(entity):
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': entity
    }

    response = requests.get(API_ENDPOINT, params = params)
    data = response.json()

    if 'search' in data and len(data['search']) > 0:
        return data['search'][0]['title']
    return None

def query_scheme(entity_id):

    query = """SELECT ?item ?itemLabel ?schemeLabel ?schemeDescription WHERE {
      # 指定我们感兴趣的实体（Q42）
      BIND(wd:%s AS ?item)
      
      # 获取实体的名称
      ?item rdfs:label ?itemLabel.
      FILTER(LANG(?itemLabel) = "en")
      
      # 获取实体的类型
      ?item wdt:P31 ?scheme.
      OPTIONAL { ?scheme rdfs:label ?schemeLabel . FILTER(LANG(?schemeLabel) = "en") }  
      OPTIONAL { ?scheme schema:description ?schemeDescription . FILTER(LANG(?schemeDescription) = "en") }  }
    """%(entity_id)
    # query = """SELECT ?scheme ?schemeLabel ?schemeDescription WHERE {
    #       wd:%s wdt:P31 ?scheme .
    #       OPTIONAL { ?scheme rdfs:label ?schemeLabel . FILTER(LANG(?schemeLabel) = "en") }
    #       OPTIONAL { ?scheme schema:description ?schemeDescription . FILTER(LANG(?schemeDescription) = "en") }
    #     }
    # """%(entity_id)
    # 设置查询
    sparql.setQuery(query)

    # 设置返回格式
    sparql.setReturnFormat(JSON)

    # 执行查询并获取结果
    results = sparql.query().convert()

    scheme_labels = []
    scheme_descs = []

    # 处理结果
    for result in results["results"]["bindings"]:
        entity_name = result["itemLabel"]["value"]
        if result.get("schemeLabel", {}).get("No label"):
            return None,None
        scheme_labels = result.get("schemeLabel", {}).get("value", "No label")
        scheme_descs = result.get("schemeDescription", {}).get("value", "No description")

    return scheme_labels, scheme_descs


def query_entity_scheme(entity_id):
    query = """SELECT ?item ?itemLabel ?schemeLabel ?schemeDescription WHERE {
      # 指定我们感兴趣的实体（Q42）
      BIND(wd:%s AS ?item)

      # 获取实体的名称
      ?item rdfs:label ?itemLabel.
      FILTER(LANG(?itemLabel) = "en")

      # 获取实体的类型
      ?item wdt:P31 ?scheme.
      OPTIONAL { ?scheme rdfs:label ?schemeLabel . FILTER(LANG(?schemeLabel) = "en") }  
      OPTIONAL { ?scheme schema:description ?schemeDescription . FILTER(LANG(?schemeDescription) = "en") }  }
    """ % (entity_id)
    # query = """SELECT ?scheme ?schemeLabel ?schemeDescription WHERE {
    #       wd:%s wdt:P31 ?scheme .
    #       OPTIONAL { ?scheme rdfs:label ?schemeLabel . FILTER(LANG(?schemeLabel) = "en") }
    #       OPTIONAL { ?scheme schema:description ?schemeDescription . FILTER(LANG(?schemeDescription) = "en") }
    #     }
    # """%(entity_id)
    # 设置查询
    sparql.setQuery(query)

    # 设置返回格式
    sparql.setReturnFormat(JSON)

    # 执行查询并获取结果
    results = sparql.query().convert()
    retries = 5
    for attempt in range(retries):
        try:
            results = sparql.query().convert()
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(0.2)
            else:
                raise

    scheme_labels = []
    scheme_descs = []
    entity_name = ''

    # 处理结果
    for result in results["results"]["bindings"]:
        entity_name = result["itemLabel"]["value"]
        if result.get("schemeLabel", {}).get("No label"):
            return None, None,None
        scheme_labels = result.get("schemeLabel", {}).get("value", "Unknown")
        scheme_descs = result.get("schemeDescription", {}).get("value", "Unknown")

    return entity_name, scheme_labels, scheme_descs

def query_wiki(entity):
    wikidata_id = get_wikidata_id(entity)

    if not wikidata_id:
        return None,None
    else:
        scheme, scheme_desc = query_scheme(wikidata_id)
        if scheme and scheme_desc:
            return scheme,scheme_desc
        else:
            return None,None


def get_property_info(property_id):
    # SPARQL查询模板
    query = f"""
    SELECT ?property ?propertyLabel ?propertyDescription
    WHERE {{
      BIND(wd:{property_id} AS ?property)
      # 获取属性的标签（名称）
      ?property rdfs:label ?propertyLabel .
      FILTER (LANG(?propertyLabel) = "en")

      # 获取属性的描述
      OPTIONAL {{ ?property schema:description ?propertyDescription . FILTER (LANG(?propertyDescription) = "en") }}
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # 执行查询
    try:
        results = sparql.query().convert()
        # 提取结果
        bindings = results.get('results', {}).get('bindings', [])
        if bindings:
            result = bindings[0]
            property_label = result.get('propertyLabel', {}).get('value', 'No label')
            property_description = result.get('propertyDescription', {}).get('value', 'No description')
            return property_label, property_description
        else:
            return 'No results found', ''
    except Exception as e:
        return f'Query failed: {e}', ''

scheme,scheme_desc = query_wiki('ADF-NALU')

