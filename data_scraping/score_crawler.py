# -*- coding: utf-8 -*-

import datetime
import sys
import os

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

#  Creates dir for data if there is no one.
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

"""It outputs time for the last 3 years, respectively, 
in accordance with the time format on the site.

Dates from the present to the past are created for pressing the buttons back 
to back. There is an element in the list, you can run the code and examine it
"""
dates = []
for i in range(2, 366 * 3):
    dates.append(str(datetime.date.today() - datetime.timedelta(days=i)))

executable_path = "./chromedriver.exe"
# web driver başlatır ve pencerisini büyütüp tam ekran yapar
driver = webdriver.Chrome(executable_path=executable_path)  # open chrome with driver.
driver.set_window_size(1920, 1080)  # Make window size as 1920x1080.
driver.set_window_position(0, 0)  # Adjust windows replacement.
driver.maximize_window()  # Make window full screen.

# Open sites with driver.
url = "https://www.mackolik.com/basketbol/canli-sonuclar"
driver.get(url)

# there is a pop-up window when we first go to the site, it closes it to be able to take action
# the close button of that window is reached via its id and the close command is given
popup_close_id = "close47559333074953319"
popup_close = driver.find_elements(By.ID, popup_close_id)
if len(popup_close) > 0:
    popup_close[0].click()
""" A method that obtains elements as a list rather than as a single element.
we used code because sometimes popup wouldn't open in this case it will pass empty list and
our code would throw an error when we tried to click it, so if the element in the list
If there is a popup, we close the window.
"""

""" The page where we pull the data does not reload, when we press the button, the data 
in the box changes. We do not want to start a new process without changing the data in 
the box, so these functions have been written to wait until the change is completed every 
time the box changes. These are the hold functions of the selenium module.
"""


def pre_waiter(d):
    expected_state = "widget-livescore widget-livescore--variant-full widget-livescore--comp component-loader " \
                     "component-loader--default component-loader--active"
    state = d.find_element(By.CSS_SELECTOR, ".widget-livescore").get_attribute("class")

    return state == expected_state


def waiter(d):
    expected_state = 'widget-livescore widget-livescore--variant-full widget-livescore--comp component-loader ' \
                     'component-loader--default'
    state = d.find_element(By.CSS_SELECTOR, ".widget-livescore").get_attribute("class")

    return state == expected_state


# the names of the different data types as well as the columns where we will store the data
columns = ["date", "start_time", "team1", "team2", "team1_score", "team2_score",
           "first_half_score"]

""" Every match data obtained in 'data_ls' is added in the form of lists, after adding 
a certain amount, we can easily convert it to a pandas dataframe.
"""
data_ls = []

""" There are two nested loops in this part, and the whole process takes place in this part. 
The loop at the top level clicks on the new day on the site and waits for the box to 
load with the waiting functions we have written.
"""
for i in range(len(dates)):
    sys.stdout.write(f"\r Gün: {i}")  # output the day which operating.
    # Find day button and click it.
    day = driver.find_element(By.ID, f"widget-dateslider-day-{dates[i]}")
    day.click()
    # Wait until data box loaded.
    try:
        WebDriverWait(driver, timeout=50).until(pre_waiter)  # Wait loading column.
        WebDriverWait(driver, timeout=50).until(waiter)  # Wait loaded column.
    except:
        print("\r Sayfadan cevap alınamadı devam ediliyor...")
        continue

    # Lists individual match score lines in the box.
    match_rows = driver.find_elements(By.CLASS_NAME, "match-row__match-content")
    date = dates[i]  # Loop current date.
    for row in match_rows:
        # Match starting time.
        start_time = row.find_element(By.CLASS_NAME, "match-row__start-time").text
        # Teams playing the match.
        teams = row.find_elements(By.CLASS_NAME, "match-row__team-name-text")
        team1 = teams[0].text  # First team.
        team2 = teams[1].text  # Second team.
        team1_score = row.find_element(By.CLASS_NAME, "match-row__score-home").text
        team2_score = row.find_element(By.CLASS_NAME, "match-row__score-away").text
        first_half_score = row.find_element(By.CLASS_NAME, "match-row__half-time-score").text

        data_ls.append([date, start_time, team1, team2, team1_score, team2_score, first_half_score])

    """ this code saves data as csv file every 50 days to avoid any problems and speed up data processing, 
    empties the list.
    
     Note that, we are still in the loop.
    """
    if i % 50 == 0:
        df = pd.DataFrame(data=data_ls, columns=columns)
        df.to_csv(f"{data_dir}save{dates[i]}.csv", encoding='utf-8')
        data_ls = []

""" When the whole process is finished, we save the last data as the 
last date will not coincide with every 50 days.
"""
df = pd.DataFrame(data=data_ls, columns=columns)
df.to_csv(f"{data_dir}lastfile.csv", encoding='utf-8')

# Quit and close web driver.
driver.quit()
