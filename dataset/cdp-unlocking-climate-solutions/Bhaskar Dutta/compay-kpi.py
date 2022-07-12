# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sn

def company_kpi():
    df_company_climate_change_2020 = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Climate Change/2020_Full_Climate_Change_Dataset.csv")
    selected_question_company = {'Do you engage in activities that could either directly or indirectly influence public policy on climate-related issues through any of the following?': 0.5,
                              'Do you engage with your value chain on climate-related issues?': 0.5,
                              'Have you identified any climate-related opportunities with the potential to have a substantive financial or strategic impact on your business?': 0.15,
                              'Have you identified any inherent climate-related risks with the potential to have a substantive financial or strategic impact on your business?': 0.15,
                              'Did you have an emissions target that was active in the reporting year?': 0.2,
                              'Did you have emissions reduction initiatives that were active within the reporting year? Note that this can include those in the planning and/or implementation phases.': 0.15,
                              'Do you classify any of your existing goods and/or services as low-carbon products or do they enable a third party to avoid GHG emissions?': 0.15,
                              'Does your organization break down its Scope 1 emissions by greenhouse gas type?': 0.2,
                              'Do you engage with your value chain on water-related issues?': 0.1,
                              'Do you evaluate and classify the tailings dams under your control according to the consequences of their failure to human health and ecosystems?': 0.2,
                              'Does your organization undertake a water-related risk assessment?': 0.3,
                              'Have you identified any inherent water-related risks with the potential to have a substantive financial or strategic impact on your business?': 0.3,
                              'Have you identified any water-related opportunities with the potential to have a substantive financial or strategic impact on your business?': 0.1}
    selected_response_company = ['Yes',
                         'Yes, water-related risks are assessed',
                         'Yes, we have identified opportunities, and some/all are being realized',
                         'Yes, our customers or other value chain partners; Yes, our suppliers',
                         'Yes, only in our value chain beyond our direct operations',
                         'Yes, only within our direct operations',
                         'Yes, our suppliers',
                         'Yes, both in direct operations and the rest of our value chain',
                         'Yes, we have identified opportunities but are unable to realize them',
                         'Yes, our customers or other value chain partners',
                         'Yes, we evaluate the consequences of tailings dam failure',
                         'Other, please specify: We utilize various methods for classification of our dams, ANCOLD, CDA and Nevada specific.',
                         "Yes, tailings dams have been classified as 'hazardous' or 'highly hazardous' (or equivalent)",
                         'Canadian Dam Association (CDA)',
                         'Australian National Committee on Large Dams (ANCOLD)',
                         "None of our tailings dams have been classified as 'hazardous' or 'highly hazardous' (or equivalent)",
                         'Company-specific guidelines',
                         "Australian National Committee on Large Dams (ANCOLD); Canadian Dam Association (CDA); Company-specific guidelines; South Africa (SANS) 10286;  Other, please specify: DENR DS, DEQ MT, NDR UT, DSOD CA, DWR NV, OSE NM",
                         "Direct engagement with policy makers",
    "Yes, other partners in the value chain; Yes, our suppliers",
    "Intensity target",
    "Trade associations",
    "Yes, other partners in the value chain",
    "Yes, our customers",
    "Yes, we have identified opportunities but are unable to realize them",
    "Absolute target",
    "Yes, our customers; Yes, our suppliers",
    "Both absolute and intensity targets",
    "Yes, other partners in the value chain; Yes, our customers; Yes, our suppliers",
    "Yes, our suppliers",
    "Yes, other partners in the value chain; Yes, our customers",
    "Direct engagement with policy makers; Funding research organizations; Trade associations",
    "Direct engagement with policy makers; Other; Trade associations",
    "Direct engagement with policy makers; Trade associations",
    "Direct engagement with policy makers; Funding research organizations; Other; Trade associations",
    "Other; Trade associations",
    "Direct engagement with policy makers; Other",
    "Other",
    "Yes, other partners in the value chain; Yes, our customers; Yes, our investee companies; Yes, our suppliers",
    "Yes, other partners in the value chain; Yes, our customers; Yes, our investee companies",
    "Funding research organizations; Other; Trade associations",
    "Funding research organizations; Trade associations",
    "Yes, our investee companies; Yes, our suppliers",
    "Yes, our customers; Yes, our investee companies; Yes, our suppliers",
    "Direct engagement with policy makers; Funding research organizations; Other",
    "No; Other; Trade associations",
    "Funding research organizations",
    "Yes, our investee companies",
    "No; Trade associations",
    "Direct engagement with policy makers; Funding research organizations",
    "Funding research organizations; Other",
    "Yes, other partners in the value chain; Yes, our investee companies",
    "Yes, our customers; Yes, our investee companies"]

    df_company_climate_change_2020_2 = df_company_climate_change_2020[df_company_climate_change_2020['question_unique_reference'].apply(lambda x: True if x in selected_question_company.keys() else False)]
    df_company_climate_change_2020_3 = df_company_climate_change_2020_2[df_company_climate_change_2020_2['response_value'].apply(lambda x: True if x in selected_response_company else False)]

    df_company_cc_summary = pd.pivot_table(df_company_climate_change_2020_3,
                                    index=['organization'], columns=['question_unique_reference'],
                   values='question_number', aggfunc='count')

    for i in df_company_cc_summary.columns:
        df_company_cc_summary[i] = df_company_cc_summary[i].apply(lambda x: 1 if x>=1 else 0)

    df_company_cc_summary['Engagement in Climate Change'] = df_company_cc_summary[list(selected_question_company.keys())[0]]*list(selected_question_company.values())[0]+df_company_cc_summary[list(selected_question_company.keys())[1]]*list(selected_question_company.values())[1]
    df_company_cc_summary['Target and Performance - Climate Change'] = df_company_cc_summary[list(selected_question_company.keys())[2]]*list(selected_question_company.values())[2] + df_company_cc_summary[list(selected_question_company.keys())[3]]*list(selected_question_company.values())[3] + df_company_cc_summary[list(selected_question_company.keys())[4]]*list(selected_question_company.values())[4] + df_company_cc_summary[list(selected_question_company.keys())[5]]*list(selected_question_company.values())[5] + df_company_cc_summary[list(selected_question_company.keys())[6]]*list(selected_question_company.values())[6] + df_company_cc_summary[list(selected_question_company.keys())[7]]*list(selected_question_company.values())[7]

    df_company_ws_2020 = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Water Security/2020_Full_Water_Security_Dataset.csv")

    df_company_ws_2020_2 = df_company_ws_2020[df_company_ws_2020['question_unique_reference'].apply(lambda x: True if x in selected_question_company.keys() else False)]
    df_company_ws_2020_3 = df_company_ws_2020_2[df_company_ws_2020_2['response_value'].apply(lambda x: True if x in selected_response_company else False)]

    df_company_ws_summary = pd.pivot_table(df_company_ws_2020_3,
                                    index=['organization'], columns=['question_unique_reference'],
                   values='question_number', aggfunc='count')

    for i in df_company_ws_summary.columns:
        df_company_ws_summary[i] = df_company_ws_summary[i].apply(lambda x: 1 if x>=1 else 0)

    df_company_ws_summary['Water Security'] = df_company_ws_summary[list(selected_question_company.keys())[8]]*list(selected_question_company.values())[8] + df_company_ws_summary[list(selected_question_company.keys())[9]]*list(selected_question_company.values())[9] + df_company_ws_summary[list(selected_question_company.keys())[10]]*list(selected_question_company.values())[10] + df_company_ws_summary[list(selected_question_company.keys())[11]]*list(selected_question_company.values())[11] + df_company_ws_summary[list(selected_question_company.keys())[12]]*list(selected_question_company.values())[12] 

    all_company = pd.DataFrame(index=np.append(np.unique(df_company_climate_change_2020['organization']), np.unique(df_company_ws_2020['organization'])))

    KPI_2 = ['Engagement in Climate Change',
                  'Target and Performance - Climate Change',
                  'Water Security']

    company_KPI = all_company.merge(df_company_cc_summary[list(set(df_company_cc_summary.columns).intersection(set(KPI_2)))], how='left', left_index=True, right_index=True)
    company_KPI = company_KPI.merge(df_company_ws_summary[list(set(df_company_ws_summary.columns).intersection(set(KPI_2)))], how='left', left_index=True, right_index=True)

    print(sn.heatmap(company_KPI.loc[['American Tower Corp.','Cadence Design Systems, Inc.', 'IQVIA', 'JPMorgan Chase & Co.',
'UnitedHealth Group Inc']]))
