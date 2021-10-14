from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
import time

#option = Options()
#option.add_argument('headless')
#option.add_argument('--window-size=1000, 800')

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')

browser = webdriver.Chrome('/root/FaaS-Soong/twitter/chromedriver', chrome_options=chrome_options)
browser.get('http://amp.paasta.koren.kr/map.php?latitude=37.4945&longitude=126.959')
time.sleep(1)
browser.get_screenshot_as_file('./accident_map.png')   
