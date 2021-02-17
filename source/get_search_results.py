"""This file is meant to be imported in the workflow notebook as the first step in data collection"""

from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import random
from random import randint
import re
from IPython.display import clear_output

def get_search_results(minresults=40):
    """Collect property urls and types by going through the search result pages of new houses and appartments,
    stopping when having reached the minimum number of results and returning a dictionary of {'url1':True/False, 'url2':True/False, ...}.
    True means house. False means apartment. Without argument only the first page is collected (~60 results)"""

    search_results = {}

    result_count = 0
    # set on which page to start the search
    page_number = 1

    driver = webdriver.Chrome()
    driver.implicitly_wait(10)
    
    # start the progress indicator and timeout logic
    start_time = time.monotonic()
    time_spent = 0

    while result_count < minresults and time_spent < 1800:
        # for each loop, scrape one results page of houses and one of appartments
        # the results are added if they are not there yet
        for houselink in results_page_scrape(pagenr=page_number,kind="house",drv=driver):
            if houselink not in search_results:
                search_results[houselink] = True
        for apartmentlink in results_page_scrape(pagenr=page_number,kind="apartment",drv=driver):
            if apartmentlink not in search_results:
                search_results[apartmentlink] = False
        result_count = len(search_results)
        page_number += 1
        # update progress indicator
        clear_output(wait=True)
        time_spent = time.monotonic() - start_time
        total_time_estimation = 1/(result_count/minresults) * time_spent
        if total_time_estimation > 1800:
            capped_time = 1800
        else:
            capped_time = total_time_estimation
        time_remaining = capped_time - time_spent
        print(f"Finishing in {time_remaining/60:.1f} minutes")
        
    driver.close()
    
    clear_output(wait=True)
    print("Finished")
    return search_results

def results_page_scrape(pagenr,kind,drv):
    '''A subroutine scraping links from 1 specific search result page, links to projects are ignored'''
    # initialise the return
    links = []
    # I slow down the frequency of requests to avoid being identified and therefore ban from the site
    time.sleep(random.uniform(1.0, 2.0))
    url=f'https://www.immoweb.be/en/search/{kind}/for-sale?countries=BE&isALifeAnnuitySale=false&page={pagenr}&orderBy=newest'
    drv.get(url)
    html = drv.page_source
    soup = BeautifulSoup(html,'lxml')
    
    for elem in soup.find_all('a', attrs={"class":"card__title-link"}):
        hyperlink = elem.get('href')
        # include in the return if it is not a -project-
        if "-project-" not in hyperlink:
            # cut the searchID off
            hyperlink = re.match("(.+)\?searchId=.+", hyperlink).group(1)
            links.append(hyperlink)
            
    return links
