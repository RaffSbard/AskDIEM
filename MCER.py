from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MainContentExtractorReader(BaseReader):
    """
    MainContentExtractor web page reader.

    Reads pages from the web.

    Args:
        text_format (str, optional): The format of the text. Defaults to "markdown".
            Requires `MainContentExtractor` package.

    """

    def __init__(self, text_format: str = "markdown") -> None:
        """Initialize with parameters."""
        self.text_format = text_format

    def load_data(self, urls: List[str]) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")

        from MCE import MainContentExtractor
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        from urllib.parse import urlparse
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        import time

        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        documents = []
        page_content = None
        DOMAINS_REQUIRING_JS = {"unisa.coursecatalogue.cineca.it"}

        for url in urls:
            domain = urlparse(url).netloc
            if domain in DOMAINS_REQUIRING_JS:
                try:
                    # Se il dominio richiede JS, usiamo Selenium
                    driver.get(url)
                    wait = WebDriverWait(driver, 10) # Aumentato a 10 secondi per sicurezza
                    wait.until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, "main.app-main-container"))
                    )
                    #time.sleep(10)  # Attesa fissa per assicurarsi che il contenuto sia caricato
                    page_content = driver.page_source
                except TimeoutException:
                    print(f"Timeout durante l'attesa del contenuto dinamico per {url}")
                    page_content = None # Se non carica, non abbiamo contenuto da analizzare
                except Exception as e:
                    print(f"Un altro errore di Selenium è occorso per {url}: {e}")
                    page_content = None
            else:
                # Altrimenti, usiamo il veloce 'requests'
                response = requests.get(url, timeout=10)
                if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                    page_content = response.content

            response = MainContentExtractor.extract(
                page_content, output_format=self.text_format, include_links=True
            )

            documents.append(Document(text=response))

        return documents
