import platform
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import math
import csv
import re
import time
import pandas as pd
import numpy as np
import os
import shutil
import datetime
import sys
from linearmodels import PanelOLS
from linearmodels import RandomEffects
import statsmodels.formula.api as smf
import functools
# from PyPDF2 import PdfFileMerger
# import os
# import glob
# import shutil
# from textblob import TextBlob
# from chooser import textclassifierclean as tc
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import PyPDF2
# from pyvirtualdisplay import Display
# from random import randint
# import docx2txt
# from tkinter import *
# from pymongo import MongoClient
# from pprint import pprint
# import numpy as np
# import importlib
# from colour import Color
# from shutil import copyfile
# import random
# import pandas as pd
# from census import Census
# from us import states
# import us
# import pandas as pd



def reduce_concat(x, sep=""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)






class wonder:

    def mcd(self,
            MCD_ICD_10_CODE_1,
            MCD_ICD_10_CODE_2,
            MCD_CAUSE_CODE_1,
            MCD_CAUSE_CODE_2,
            RUN_NAME,
            AGEG,
            by_variables
            ):


        MCD_ICD_10_CODE_1 = ["X93","X94","X95","X72","X73","X74","W32","W33","W34","Y22","Y23","Y24","Y35.0","Y36.4"]
        MCD_ICD_10_CODE_2 = None

        chrome_options = webdriver.ChromeOptions()

        download_dir = self.download_dir

        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
            })

        d = webdriver.Chrome(chrome_options=chrome_options)
        d.implicitly_wait(30)

        # base_url = "https://wonder.cdc.gov/ucd-icd10.html"
        base_url = "https://wonder.cdc.gov/mcd-icd10.html"

        d.get(base_url)
        accept_button = d.find_elements_by_class_name("button2")[1]
        d.execute_script("arguments[0].click();", accept_button)
        # d.execute_script("sendTab();")
        # accept_button.click()


        ct = 1
        for var in by_variables:
            if var == 'state':
                ## By State
                d.find_element_by_xpath("//select[@name='B_{}']/optgroup[@label='Location']/option[text()='State']".format(ct)).click()
            elif var == 'year':
                ## By Year
                d.find_element_by_xpath("//select[@name='B_{}']/optgroup[@label='Year and Month']/option[text()='Year']".format(ct)).click()
            elif var == 'month':
                ## By Year
                d.find_element_by_xpath("//select[@name='B_{}']/optgroup[@label='Year and Month']/option[text()='Month']".format(ct)).click()
            elif var == "age":
                ## By Age Groups
                d.find_element_by_xpath("//select[@name='B_{}']/optgroup[@label='Demographics']/option[text()='Age Groups']".format(ct)).click()
            elif var == "race":
                # ## By Race
                d.find_element_by_xpath("//select[@name='B_{}']/optgroup[@label='Demographics']/option[text()='Race']".format(ct)).click()
            elif var == "hispanic":
                ## By Hispanic
                d.find_element_by_xpath("//select[@name='B_{}']/optgroup[@label='Demographics']/option[text()='Hispanic Origin']".format(ct)).click()
            elif var == "gender" or var == "sex":
                ## By Gender
                d.find_element_by_xpath("//select[@name='B_{}']/optgroup[@label='Demographics']/option[text()='Gender']".format(ct)).click()
            elif var == '':
                ct = ct - 1

            ct = ct + 1


        ### Measures

        ## Crude rate 95%CI
        d.find_element_by_css_selector("input#CM_31").click()

        ## Crude rate SE
        d.find_element_by_css_selector("input#CM_32").click()



        if 'age' not in by_variables and (AGEG is None or len(AGEG) != 1):
            print("Getting age adjusted numbers")
            ## Age Adjusted Rates
            d.find_element_by_css_selector("input#CO_aar_enable").click()

            ## AAR CI
            d.find_element_by_css_selector("input#CO_aar_CI").click()

            ## AAR SE
            d.find_element_by_css_selector("input#CO_aar_SE").click()

        ## Run name
        d.find_element_by_css_selector("input#TO_title").send_keys(RUN_NAME)

        def double_click(element):
            actionChains = ActionChains(d)
            actionChains.double_click(element).perform()



        # Filters

        # Agebox
        if self.AGEG is not None:
            d.find_element_by_xpath("//select[@name='V_D77.V5']").send_keys(Keys.CONTROL, 'a')
            double_click(d.find_element_by_xpath("//select[@name='V_D77.V5']/option[@value='{}']".format(self.AGEG)))

        # Gender
        if self.GENDERG is not None:
            d.find_element_by_xpath("//select[@name='V_D77.V7']").send_keys(Keys.CONTROL, 'a')
            double_click(d.find_element_by_xpath("//select[@name='V_D77.V7']/option[@value='{}']".format(self.GENDERG)))

            # V_D77.V7

        if self.HISPANICG is not None:
            d.find_element_by_xpath("//select[@name='V_D77.V17']").send_keys(Keys.CONTROL, 'a')
            double_click(d.find_element_by_xpath("//select[@name='V_D77.V17']/option[@value='{}']".format(self.HISPANICG)))

            # V_D77.V17


        if self.RACEG is not None:
            d.find_element_by_xpath("//select[@name='V_D77.V8']").send_keys(Keys.CONTROL, 'a')
            double_click(d.find_element_by_xpath("//select[@name='V_D77.V8']/option[@value='{}']".format(self.RACEG)))

            # V_D77.V8



        if MCD_ICD_10_CODE_1 is not None:

            obj = d.find_element_by_css_selector("div.finder-picks").find_element_by_xpath("//input[@id='RO_mcdD77.V13']")
            d.execute_script("arguments[0].click();", obj)

            d.find_element_by_id("codes-D77.V13").send_keys(Keys.CONTROL, 'a')

            obj = d.find_element_by_xpath("//input[@name='finder-action-D77.V13-Open Fully']")
            d.execute_script("arguments[0].click();", obj)

            double_click(d.find_element_by_id("codes-D77.V13").find_element_by_xpath("//option[@value='{}']".format(MCD_ICD_10_CODE_1[0])))

            d.find_element_by_id("codes-D77.V13").find_element_by_xpath("//option[@value='{}']".format("*All*")).click()
            d.find_element_by_id("codes-D77.V13").find_element_by_xpath("//option[@value='{}']".format(MCD_ICD_10_CODE_1[0])).click()
            obj = d.find_element_by_id("codes-D77.V13").find_element_by_xpath("//option[@value='{}']".format(MCD_ICD_10_CODE_1[0]))
            d.execute_script("arguments[0].click();", obj)

            if len(MCD_ICD_10_CODE_1) > 1:
                for ct in range(1, len(MCD_ICD_10_CODE_1)):
                    # d.find_element_by_id("codes-D77.V13").find_element_by_xpath("//option[@value='{}']".format(MCD_ICD_10_CODE_1[ct])).click()
                    obj = d.find_element_by_id("codes-D77.V13").find_element_by_xpath("//option[@value='{}']".format(MCD_ICD_10_CODE_1[ct]))
                    d.execute_script("arguments[0].click();", obj)


            d.execute_script("add('D77.V13', 'and2')")


            if MCD_ICD_10_CODE_2 is not None:


                double_click(d.find_element_by_id("codes-D77.V13").find_element_by_xpath("//option[@value='{}']".format(MCD_ICD_10_CODE_2[0])))

                if len(MCD_ICD_10_CODE_2) > 1:
                    for ct in range(1, len(MCD_ICD_10_CODE_2)):
                        # d.find_element_by_id("codes-D77.V13").find_element_by_xpath("//option[@value='{}']".format(MCD_ICD_10_CODE_2[ct])).click()
                        obj = d.find_element_by_id("codes-D77.V13").find_element_by_xpath("//option[@value='{}']".format(MCD_ICD_10_CODE_2[ct]))
                        d.execute_script("arguments[0].click();", obj)


                d.execute_script("add('D77.V13_AND', 'and2')")



        elif MCD_CAUSE_CODE_1 is not None:
            ############ OPTION 2
            ## Choose 113 Cause List
            d.find_element_by_css_selector("div.finder-picks").find_element_by_xpath("//input[@id='RO_mcdD77.V15']").click()


            first_button = d.find_elements_by_xpath("//select[@id='codes-D77.V15']/option")[0]
            first_button.click()

            cause_button = d.find_element_by_xpath("//select[@id='codes-D77.V15']/option[@value='{}']".format(MCD_CAUSE_CODE_1[0]))
            cause_button.click()


            if len(MCD_CAUSE_CODE_1) > 1:
                for ct in range(1, len(MCD_CAUSE_CODE_1)):
                    d.find_element_by_xpath("//select[@id='codes-D77.V15']/option[@value='{}']".format(MCD_CAUSE_CODE_1[ct])).click()

                    # d.find_element_by_css_selector("div.req-form-select-wide-div").find_element_by_xpath("//option[@value='{}']".format(CAUSE_LIST_CODE[ct])).click()

            d.execute_script("add('D77.V15', 'and2 nohier nocodes')")

            for code in MCD_CAUSE_CODE_1:
                d.find_element_by_xpath("//select[@id='codes-D77.V15']/option[@value='{}']".format(code)).click()



            if MCD_CAUSE_CODE_2 is not None:

                ## Choose cause
                # double_click(d.find_element_by_id("codes-D77.V15").find_element_by_xpath("//option[@value='{}']".format(MCD_CAUSE_CODE_2[0])))
                cause_button = d.find_element_by_xpath("//select[@id='codes-D77.V15']/option[@value='{}']".format(MCD_CAUSE_CODE_2[0]))
                cause_button.click()
                # double_click(d.find_element_by_css_selector("div.req-form-select-wide-div").find_element_by_xpath("//option[@value='{}']".format(CAUSE_LIST_CODE[0]))) # suicide

                if len(MCD_CAUSE_CODE_2) > 1:
                    for ct in range(1, len(MCD_CAUSE_CODE_2)):
                        d.find_element_by_xpath("//select[@id='codes-D77.V15']/option[@value='{}']".format(MCD_CAUSE_CODE_2[ct])).click()

                        # d.find_element_by_css_selector("div.req-form-select-wide-div").find_element_by_xpath("//option[@value='{}']".format(CAUSE_LIST_CODE[ct])).click()

                d.execute_script("add('D77.V15_AND', 'and2 nohier nocodes')")



        ### SUBMIT

        # input("Hit enter...")
        # d.find_element_by_css_selector("input#export-option").click()

        export = d.find_element_by_css_selector("input#export-option")
        d.execute_script("arguments[0].click();", export)

        # d.find_element_by_xpath("//div[@class='footer-buttons']/input[@value='Send']").click()
        send = d.find_element_by_xpath("//div[@class='footer-buttons']/input[@value='Send']")
        try:
            d.execute_script("arguments[0].click();", send)
        except TimeoutException:
            time.sleep(360)
        except:
            time.sleep(360)

        not_done = True
        while not_done:
            time.sleep(2)
            if os.path.exists("{}/{}.txt".format(download_dir, RUN_NAME)):
                not_done = False



        time.sleep(3)
        old_filename = max([download_dir + "/" + f for f in os.listdir(download_dir)], key=os.path.getctime)
        filename = download_dir + "/" + "attach.csv"
        try:
            shutil.copy(os.path.join(old_filename), filename)
            os.remove(old_filename)
        except shutil.SameFileError:
            pass
        except IOError:
            time.sleep(5)
            shutil.copy(os.path.join(old_filename), filename)
            os.remove(old_filename)

        ## READ IN CSV
        def read_wonder_csv(filename):
            initial = pd.read_csv(filename, delimiter = "\t")
            final =   pd.read_csv(filename, delimiter = "\t", skipfooter = len(initial.index) - initial[initial[initial.columns[0]]=='---'].index[0])
            return final

        self.df = read_wonder_csv(filename)
        self.df.to_csv("{}/{}_pull.csv".format(download_dir, RUN_NAME))
        os.remove(filename)

        d.quit()




    def start_plots(self, nrow=4, ncol=1, figsize=(6, 14)):
        import matplotlib
        import matplotlib.pyplot as plt
        plt.style.use('classic')
        matplotlib.rcParams['axes.formatter.useoffset'] = False
        matplotlib.rcParams.update({'font.size': 8})

        self.fig, self.ax = plt.subplots(nrows=nrow, ncols=ncol, figsize = figsize)
        plt.subplots_adjust(left=0.5, bottom=0.5)
        self.plot_counter = 0



    def state_plot_trend(self, title, value, begin_year, end_year):
        import plotly.graph_objects as go
        df = self.df


        for index, row in df.iterrows():

            year = df.loc[index, "Year"]
            if year not in [begin_year, end_year]:
                df = df.drop(index)
                continue


            if value == "Age Adjusted Rate":

                aarmean = df.loc[index, "Age Adjusted Rate"]
                aarlo95 = df.loc[index, "Age Adjusted Rate Lower 95% Confidence Interval"]
                aarhi95 = df.loc[index, "Age Adjusted Rate Upper 95% Confidence Interval"]


                if aarmean == "Unreliable" and aarlo95 != "Unreliable":
                    df.loc[index, "Age Adjusted Rate"] = (float(aarlo95) + float(aarhi95)) / 2

                elif aarmean == "Unreliable" and aarlo95 == "Unreliable":
                    df.loc[index, "Age Adjusted Rate"] = np.nan

                else:
                    pass


            elif value == "Crude Rate":

                crmean = df.loc[index, "Crude Rate"]
                crlo95 = df.loc[index, "Crude Rate Lower 95% Confidence Interval"]
                crhi95 = df.loc[index, "Crude Rate Upper 95% Confidence Interval"]



                if crmean == "Unreliable" and crlo95 != "Unreliable":
                    df.loc[index, "Crude Rate"] = (float(crlo95) + float(crhi95)) / 2

                elif crmean == "Unreliable" and crlo95 == "Unreliable":
                    df.loc[index, "Crude Rate"] = np.nan

                else:
                    pass

        df = df[df['Year'] != "All"]
        df['Year'] = ["y{}".format(int(x)) for x in df['Year']]
        df = pd.DataFrame(df.pivot(index='State', columns='Year', values=value).to_records())

        # print(df.columns)
        # print(df.head())

        end_year_column = "y{}".format(int(end_year))
        begin_year_column = "y{}".format(int(begin_year))

        for index, row in df.iterrows():
            state = df.loc[index, "State"]
            end_year_value = float(df.loc[index, end_year_column])
            begin_year_value = float(df.loc[index, begin_year_column])
            pct_diff = ((end_year_value - begin_year_value) / begin_year_value) * 100
            print("{}: {}, {} (pct diff {})".format(state, end_year_value, begin_year_value, pct_diff))
            df.loc[index, "pct_diff"] = pct_diff


        df = df[df["State"].notnull()]
        print(df.tail())

        abb_mapping = us.states.mapping('name', 'abbr')

        self.statefig = go.Figure(data=go.Choropleth(
            locations=[abb_mapping.get(x, None) for x in df['State']], # Spatial coordinates
            z = df["pct_diff"].astype(float), # Data to be color-coded
            locationmode = 'USA-states', # set of locations match entries in `locations`
            colorscale = 'Reds',
            colorbar_title = value,
        ))

        self.statefig.update_layout(
            title_text = title,
            geo_scope='usa', # limite map scope to USA
        )

        self.statefig.show()




    def state_plot(self, title, value, year = None):
        import plotly.graph_objects as go

        df = self.df


        for index, row in df.iterrows():


            if value == "Age Adjusted Rate":

                aarmean = df.loc[index, "Age Adjusted Rate"]
                aarlo95 = df.loc[index, "Age Adjusted Rate Lower 95% Confidence Interval"]
                aarhi95 = df.loc[index, "Age Adjusted Rate Upper 95% Confidence Interval"]


                if aarmean == "Unreliable" and aarlo95 != "Unreliable":
                    df.loc[index, "Age Adjusted Rate"] = (float(aarlo95) + float(aarhi95)) / 2

                elif aarmean == "Unreliable" and aarlo95 == "Unreliable":
                    df.loc[index, "Age Adjusted Rate"] = np.nan

                else:
                    pass


            elif value == "Crude Rate":

                crmean = df.loc[index, "Crude Rate"]
                crlo95 = df.loc[index, "Crude Rate Lower 95% Confidence Interval"]
                crhi95 = df.loc[index, "Crude Rate Upper 95% Confidence Interval"]

                if crmean == "Unreliable" and crlo95 != "Unreliable":
                    df.loc[index, "Crude Rate"] = (float(crlo95) + float(crhi95)) / 2

                elif crmean == "Unreliable" and crlo95 == "Unreliable":
                    df.loc[index, "Crude Rate"] = np.nan

                else:
                    pass

        df = df[df['Year'] != "All"]

        if year is None:
            df = df[df['Year'] == df['Year'].max()]
        else:
            df = df[df['Year'] == year]


        abb_mapping = us.states.mapping('name', 'abbr')

        self.statefig = go.Figure(data=go.Choropleth(
            locations=[abb_mapping.get(x, None) for x in df['State']], # Spatial coordinates
            z = df[value].astype(float), # Data to be color-coded
            locationmode = 'USA-states', # set of locations match entries in `locations`
            colorscale = 'Reds',
            colorbar_title = "Percent change in\n{}".format(value),
        ))

        self.statefig.update_layout(
            title_text = title,
            geo_scope='usa', # limite map scope to USA
        )

        self.statefig.show()



    def race_trends(self, title="", xtitle="", ytitle="", value='Deaths', byvar='Race'):

        import matplotlib.pyplot as plt

        df = self.df

        for index, row in df.iterrows():

            if value == "Age Adjusted Rate":

                aarmean = df.loc[index, "Age Adjusted Rate"]
                aarlo95 = df.loc[index, "Age Adjusted Rate Lower 95% Confidence Interval"]
                aarhi95 = df.loc[index, "Age Adjusted Rate Upper 95% Confidence Interval"]

                if aarmean == "Unreliable" and aarlo95 != "Unreliable":
                    df.loc[index, "Age Adjusted Rate"] = (float(aarlo95) + float(aarhi95)) / 2
                elif aarmean == "Unreliable" and aarlo95 == "Unreliable":
                    df.loc[index, "Age Adjusted Rate"] = np.nan
                else:
                    pass


            elif value == "Crude Rate":

                crmean = df.loc[index, "Crude Rate"]
                crlo95 = df.loc[index, "Crude Rate Lower 95% Confidence Interval"]
                crhi95 = df.loc[index, "Crude Rate Upper 95% Confidence Interval"]

                if crmean == "Unreliable" and crlo95 != "Unreliable":
                    df.loc[index, "Crude Rate"] = (float(crlo95) + float(crhi95)) / 2

                elif crmean == "Unreliable" and crlo95 == "Unreliable":
                    df.loc[index, "Crude Rate"] = np.nan

                else:
                    pass


            if byvar=="Ten-Year Age Groups":
                age = df.loc[index, "Ten-Year Age Groups"]

                if pd.isnull(age):
                    df = df.drop(index)


            if byvar=="Race":

                race = df.loc[index, 'Race']
                hispanic = df.loc[index, 'Hispanic Origin']

                if pd.isnull(race):
                    df.loc[index, 'Race'] = "Combined"

                elif "Hispanic or Latino" in df['Hispanic Origin'].values.tolist():

                    if race == "White":

                        if hispanic == "Hispanic or Latino":
                            df.loc[index, 'Race'] = "Hispanic White"

                        elif hispanic == "Not Hispanic or Latino":
                            df.loc[index, 'Race'] = "Non-Hispanic White"

                    else:
                        if pd.isnull(hispanic):
                            df.loc[index, 'Race'] = race
                        else:
                            df = df.drop(index)
                            continue


                newrace = df.loc[index, 'Race']
                if newrace not in ['Hispanic White', 'Non-Hispanic White', 'Black or African American']:
                    df = df.drop(index)
                    continue

        df = df[df[byvar].notnull()]
        df = df[df.Year.notnull()]
        byvar_value = df[byvar].unique().tolist()

        if value == "Deaths":

            plot_df = df[df.Year!=""].pivot_table(index='Year', columns=byvar,
                                                        values = value,
                                                        aggfunc='sum',
                                                        margins = False,
                                                        fill_value = 0
                                                        )

        elif value in ['Age Adjusted Rate', 'Crude Rate']:

            df = df[df[value] != "Not Applicable"]
            df[value] = [float(x) for x in df[value]]

            plot_df = df[df.Year!=""].pivot_table(index='Year', columns=byvar,
                                                        values = value,
                                                        aggfunc='first',
                                                        margins = False,
                                                        fill_value = 0
                                                        )


        y_upper_lim = plot_df.max().max()*1.1
        plot_df['Year'] = plot_df.index.values
        plot_df = plot_df[plot_df['Year'] != "All"]
        self.plot_df = plot_df

        for val in byvar_value:
            try:
                self.ax[self.plot_counter].plot(plot_df.Year, plot_df[val], label=val)
            except:
                pass

        self.ax[self.plot_counter].legend(loc='best', frameon = True)
        self.ax[self.plot_counter].axis([1998, 2018, 0, y_upper_lim])
        self.ax[self.plot_counter].set_xlabel(xtitle)
        self.ax[self.plot_counter].set_ylabel(ytitle)
        self.ax[self.plot_counter].set_title(title)

        self.fig.tight_layout()
        self.plot_counter = self.plot_counter + 1


    def mcd_or_load(
            self,
            MCD_ICD_10_CODE_1 = None,
            MCD_ICD_10_CODE_2 = None,
            MCD_CAUSE_CODE_1 = None,
            MCD_CAUSE_CODE_2 = None,
            RUN_NAME = "",
            by_variables = None,
            existing_file = False,
            AGEG = None
            ):
        if not existing_file:

            self.mcd(
                    MCD_ICD_10_CODE_1 = MCD_ICD_10_CODE_1,
                    MCD_ICD_10_CODE_2 = MCD_ICD_10_CODE_2,
                    MCD_CAUSE_CODE_1 = None,
                    MCD_CAUSE_CODE_2 = None,
                    RUN_NAME = RUN_NAME,
                    by_variables = by_variables,
                    AGEG = AGEG
                    )

        else:
            self.df = pd.read_csv("{}/{}_pull.csv".format(self.download_dir, RUN_NAME))


    def __init__(self, MCD_ICD_10_CODE_1=None, MCD_ICD_10_CODE_2=None,
                MCD_CAUSE_CODE_1 = None, MCD_CAUSE_CODE_2 = None,
                RUN_NAME="", AGEG=None, GENDERG = None, RACEG = None,
                HISPANICG = None, by_variables="", existing_file = False,
                download_dir = True, just_go=False):


        if not isinstance(download_dir, str):
            print("{} is not a string".format(download_dir))
            if platform.system()=="Windows":
                print("Platform is Windows")
                download_dir = "C:\\Users\\tcapu\\Google Drive\\modules\\pull-wonder\\downloads"
            else:
                print("Platform is NOT Windows")
                download_dir = "/media/sf_Google_Drive/modules/pull-wonder/downloads"

        print("Download dir is {}".format(download_dir))



        self.MCD_ICD_10_CODE_1 = MCD_ICD_10_CODE_1
        self.MCD_ICD_10_CODE_2 = MCD_ICD_10_CODE_2
        self.MCD_CAUSE_CODE_1 = MCD_ICD_10_CODE_1
        self.MCD_CAUSE_CODE_2 = MCD_ICD_10_CODE_2
        self.RUN_NAME = RUN_NAME
        self.by_variables = by_variables
        self.download_dir = download_dir
        self.existing_file = existing_file
        self.AGEG=AGEG
        self.GENDERG =GENDERG
        self.RACEG = RACEG
        self.HISPANICG =HISPANICG

        if not just_go:

            self.mcd_or_load(
                    MCD_ICD_10_CODE_1 = self.MCD_ICD_10_CODE_1,
                    MCD_ICD_10_CODE_2 = self.MCD_ICD_10_CODE_2,
                    MCD_CAUSE_CODE_1 = self.MCD_CAUSE_CODE_1,
                    MCD_CAUSE_CODE_2 = self.MCD_CAUSE_CODE_2,
                    RUN_NAME = self.RUN_NAME,
                    by_variables = self.by_variables,
                    existing_file = self.existing_file,
                    AGEG = self.AGEG

                    )
