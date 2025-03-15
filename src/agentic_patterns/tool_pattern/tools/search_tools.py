# src/agentic_patterns/tool_pattern/tools.py

from typing import List
import requests
from bs4 import BeautifulSoup
from ..tool import tool

@tool
def search_health_articles(
    condition: str,
    max_results: int = 5,
    sources: List[str] = ["mayoclinic", "nih", "webmd"]
) -> List[dict]:
    """
    Search for medical articles about chronic disease management from reputable healthcare websites.
    
    Args:
        condition (str): The medical condition to search for (e.g., 'diabetes', 'hypertension')
        max_results (int): Maximum number of articles to return (default: 5)
        sources (List[str]): List of sources to search from ['mayoclinic', 'nih', 'webmd']
    
    Returns:
        List[dict]: A list of articles with their titles and URLs
    """
    articles = []
    
    # Search Mayo Clinic
    if "mayoclinic" in sources:
        try:
            url = f"https://www.mayoclinic.org/search/search-results?q={condition}"
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('a', class_='search-result-link')
                for result in results[:max_results]:
                    articles.append({
                        'title': result.text.strip(),
                        'url': f"https://www.mayoclinic.org{result['href']}",
                        'source': 'Mayo Clinic'
                    })
        except Exception as e:
            print(f"Error searching Mayo Clinic: {e}")

    # Search NIH (PubMed Central)
    if "nih" in sources:
        try:
            url = f"https://www.ncbi.nlm.nih.gov/pmc/?term={condition}+management"
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('div', class_='rslt')
                for result in results[:max_results]:
                    title_elem = result.find('a', class_='title-link')
                    if title_elem:
                        articles.append({
                            'title': title_elem.text.strip(),
                            'url': f"https://www.ncbi.nlm.nih.gov{title_elem['href']}",
                            'source': 'NIH PubMed Central'
                        })
        except Exception as e:
            print(f"Error searching NIH: {e}")

    # Search WebMD
    if "webmd" in sources:
        try:
            url = f"https://www.webmd.com/search/search_results/default.aspx?query={condition}"
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('a', class_='search-result-link')
                for result in results[:max_results]:
                    articles.append({
                        'title': result.text.strip(),
                        'url': result['href'],
                        'source': 'WebMD'
                    })
        except Exception as e:
            print(f"Error searching WebMD: {e}")

    return articles[:max_results]

# Export the tool
tools = [search_health_articles]