# %% [code]
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

# %% [code]
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sn

# %% [code]
def city_kpi (city_name=None):
    df_city_2020 = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2020_Full_Cities_Dataset.csv")
    selected_question_city = {'Does your city council, or similar authority, have a published plan that addresses climate change adaptation?': 0.4,
                              'Does your city have a strategy, or other policy document, in place for how to measure and reduce consumption-based GHG emissions in your city?': 0.3,
                              'Does your city have a city-wide emissions inventory to report?': 0.3,
                              'Does your city have a consumption-based inventory to measure emissions from consumption of goods and services by your residents?': 0.3,
                              'Since your last submission, have you needed to recalculate any past city-wide GHG emission inventories previously reported to CDP?': 0.1,
                              'Based on the climate hazards identified as "high risk" in your city, have you identified climate exposure scenarios?': 0.15,
                              'Have you compiled information related to climate risk, vulnerabilities, and adaptive capacities into a baseline synthesis report?': 0.15,
                              'Have you identified the most vulnerable geographic areas in your city?': 0.15,
                              'Is your city facing risks to public health or health systems associated with climate change?': 0.55,
                              'Does your city have an update/revision process for the climate risk and vulnerability assessment?': 0.5,
                              'Has a climate change risk and vulnerability assessment been undertaken for your city?': 0.5,
                              'Does your city have a climate change mitigation or energy access plan for reducing city-wide GHG emissions?': 0.4,
                              'Does your city-wide emissions reduction target(s) account for the use of transferable emissions units?': 0.4,
                              'Is your city-wide emissions reduction target(s) conditional on the success of an externality or component of policy outside of your control?': 0.1,
                              'Does your city incorporate sustainability goals and targets (e.g. GHG reductions) into the master planning for the city?': 0.4,
                              'Has the GHG emissions data you are currently reporting been externally verified or audited in part or in whole?': 0.2,
                              'Does your city report to the national Measurement, Reporting and Verification (MRV) system (if in place)?': 0.1,
                              'Does your city collaborate in partnership with businesses in your city on sustainability projects?': 1,
                              'Does your city have a publicly available Water Resource Management\u202fstrategy?': 0.5,
                              'Are you aware of any substantive current or future risks to your city’s water security?': 0.5,
                              'Does your city have a renewable energy or electricity target?': 0.5,
                              'Does your city have a target to increase energy efficiency?': 0.5,
                              'Do you incentivise fresh fruit/vegetables vendor locations?': 0.1,
                              'Do you subsidise fresh fruits and vegetables?': 0.1,
                              'Do you tax/ban higher carbon foods (meat, dairy, ultra-processed)?': 0.4,
                              'Do you use regulatory mechanisms that limit advertising of higher carbon foods (meat, dairy, ultra-processed)?': 0.4,
                              'Bans or restrictions on single use or non-recyclable materials': 0.35,
                              'Mandatory waste segregation': 0.35,
                              'Sanitary landfill with leachate capture and landfill gas management system': 0.1,
                              'Target(s) on reducing food waste to disposal (landfill and incineration)': 0.1,
                              'Volume based waste collection fees/incentives': 0.1}
    selected_response = ['Yes',
                         'For heat, yes. See the Neighborhood heat vulnerability study here: https://www.denvergov.org/content/denvergov/en/environmental-health/health-equity/health-equity-data/heat-vulnerability.html.',
                         'Yes - Flood modelling has been completed to see which geographic locations within the municipality are most at risk. Heat mapping and open space analysis has been completed to identify where hotspots exist.',
                        "1) By 2030, all new buildings will be built to produce near-zero greenhouse gas (GHG) emissions.2) By 2050, all existing buildings will have been retrofitted to achieve net zero emissions.A set of performance targets for different building typologies will be part of the Existing Buildings Emissions Strategy. The strategy is scheduled to be presented to City Council in the first half of 2021",
"1) By 2030, all new buildings will be built to produce near-zero greenhouse gas (GHG) emissions.2) By 2050, all existing buildings will have been retrofitted to achieve net zero emissions.A set of performance targets for different building typologies will be part of the Existing Buildings Emissions Strategy. The strategy is scheduled to be presented to City Council in the first half of 2021",
"45% Reduction by 2030 over 2008 baseline, and net-zero carbon by 2050. https://www.seattle.gov/Documents/Departments/Environment/ClimateChange/2013_CAP_20130612.pdf",
"Baseline analysis on climate risk, vulnerabilities and adaptive capacities is provided in the City's Climate Action Plan, Chapter 5.",
"20% reduction  in heat and 10% reduction of electricity consumption in 2025 compared to 2010",
"Carbon neutrality target applies across all sectors; the City adopted a new target to reduce emissions by 60% by 2030 from a 2005 baseline in the 2019 Climate Action Plan Update: https://www.boston.gov/sites/default/files/imce-uploads/2019-10/city_of_boston_2019_climate_action_plan_update_2.pdf",
"10% Reduction in Energy Use by 2030 over 2008 baseline. https://www.seattle.gov/Documents/Departments/Environment/ClimateChange/2013_CAP_20130612.pdf",
"By 2030, all new buildings will be built to produce near-zero greenhouse gas (GHG) emissions.",
"40% reduction  in energy consumption in 2025 compared to 2010",
"A political decision to divest from fossil energy was taken in 2016.",
"All new buildings will be net zero carbon by 2030; and 100% ofbuildings will be net zero carbon by 2050 (More information available at plan.lamayor.org)",
"- Implementation of Anti-Junk Food and Sugary Drinks Ordinance of 2017 where different stores and canteens in primary and secondary schools within 100 meters are mandated to sell nutritious food.- Continuous implementation of Urban Farming Project to cont",
"39% reduction in emissions by 2030 over a 2008 baseline, net-carbon neutral by 2050.",
"1) By 2030, all new buildings will be built to produce near-zero greenhouse gas (GHG) emissions.2) By 2050, all existing buildings will have been retrofitted to achieve net zero emissions.A set of performance targets for different building typologies will be part of the Existing Buildings Emissions Strategy. The strategy is scheduled to be presented to City Council in the first half of 2021",
"- https://www.melbourne.vic.gov.au/SiteCollectionDocuments/emissions-reduction-plan.pdf",
"A corporate climate adaptation plan will include M&E in the process of development, with risk and action registers.",
"Cardiff Council food Strategy (for Council properties only). The aim of the policy is that Cardiff Council wants everyone in Cardiff to have access to affordable good food, and to understand where their food comes from.",
"A large portion of the Denver Employee Retirement Plan is invested in fossil fuels: https://www.derp.org/",
"20% reduction  in heat and electricity consumption in 2025 compared to 2010",
"32% Reduction by 2030 over 2008 baseline, and net-zero carbon by 2050. https://www.seattle.gov/Documents/Departments/Environment/ClimateChange/2013_CAP_20130612.pdf",
"All new buildings will be net zero carbon by 2030; and 100% ofbuildings will be net zero carbon by 2050 (More information available at plan.lamayor.org)",
"20% Reduction in Energy Use by 2030 over 2008 baseline. https://www.seattle.gov/Documents/Departments/Environment/ClimateChange/2013_CAP_20130612.pdf",
"1) By 2030, all new buildings will be built to produce near-zero greenhouse gas (GHG) emissions.2) By 2050, all existing buildings will have been retrofitted to achieve net zero emissions.A set of performance targets for different building typologies will be part of the Existing Buildings Emissions Strategy. The strategy is scheduled to be presented to City Council in the first half of 2021",
"A programmatic monitoring and evaluation system of climate change adaptation plans, programs, projects, and activities are encapsulated in budget preparation documents for each fiscal cycle. This pertains to tagging projects as either climate mitigation or adaptation actions under the Climate Change Expenditure Tagging as indicated under existing Philippine laws and regulations. Each budgetary form indicates major final outputs and key performance indicators which will measure the degree of success of the planned initiative.Quezon City's Local Climate Change Action Plan (LCCAP) 2017-2027 is also set up with the following systems:â€¢ collecting and recording the information;â€¢ analysing the information; andâ€¢ using information to inform decision makersChapter 7 of the Local Climate Change Action Plan (LCCAP) 2017-2027 deals with the Monitoring and Evaluation (M&E) as a very significant feature of the QC-LCCAP 2017-2027. The chapter describes how the Quezon City CWG aimed at learning from the activities done during the preparation of the QC-LCCAP 2017-2027, (i.e. what was done and how it was done) as the QC-LCCAP 2017-2027 is a long term plan, the strategies have the tendency to be changed/revised depending if the strategies and corresponding Programs, Projects, and Activities (PPAs) are working or not. This chapter also explains how the completion of these projects will become the indication for the Quezon City Local Governmentâ€™s decision makers about when the plans are working, and when the environment changed, and how the plans can be revised with the changing environment"]

    df_city_2020_2 = df_city_2020[df_city_2020['Question Name'].apply(lambda x: True if x in selected_question_city.keys() else False)]
    df_city_2020_3 = df_city_2020_2[df_city_2020_2['Response Answer'].apply(lambda x: True if x in selected_response else False)]

    df_city_sumary = pd.pivot_table(df_city_2020_3,
                                    index=['Organization'], columns=['Question Name'],
                   values='Question Number', aggfunc='count')

    for i in df_city_sumary.columns:
        df_city_sumary[i] = df_city_sumary[i].apply(lambda x: 1 if x>=1 else 0)

    df_city_sumary[list(selected_question_city.keys())[8]] = df_city_sumary[list(selected_question_city.keys())[8]].apply(lambda x: 0 if x>=1 else 1)
    df_city_sumary[list(selected_question_city.keys())[13]] = df_city_sumary[list(selected_question_city.keys())[13]].apply(lambda x: 0 if x>=1 else 1)

    df_city_sumary['Adaption Planning and Governance'] = df_city_sumary[list(selected_question_city.keys())[0]]*list(selected_question_city.values())[0]+df_city_sumary[list(selected_question_city.keys())[14]]*list(selected_question_city.values())[14]+df_city_sumary[list(selected_question_city.keys())[15]]*list(selected_question_city.values())[15]
    df_city_sumary['Emission Data and Other Resources'] = df_city_sumary[list(selected_question_city.keys())[1]]*list(selected_question_city.values())[1]+df_city_sumary[list(selected_question_city.keys())[2]]*list(selected_question_city.values())[2]+df_city_sumary[list(selected_question_city.keys())[3]]*list(selected_question_city.values())[3]+df_city_sumary[list(selected_question_city.keys())[4]]*list(selected_question_city.values())[4]
    df_city_sumary['Climate Hazards'] = df_city_sumary[list(selected_question_city.keys())[5]]*list(selected_question_city.values())[5]+df_city_sumary[list(selected_question_city.keys())[6]]*list(selected_question_city.values())[6]+df_city_sumary[list(selected_question_city.keys())[7]]*list(selected_question_city.values())[7]+df_city_sumary[list(selected_question_city.keys())[8]]*list(selected_question_city.values())[8]
    df_city_sumary['Climate Risk and Vulnerability Assessment'] = df_city_sumary[list(selected_question_city.keys())[9]]*list(selected_question_city.values())[9]+df_city_sumary[list(selected_question_city.keys())[10]]*list(selected_question_city.values())[10]
    df_city_sumary['Climate Action and GHG Emission Mitigation Planning and Target Setting'] = df_city_sumary[list(selected_question_city.keys())[11]]*list(selected_question_city.values())[11]+df_city_sumary[list(selected_question_city.keys())[12]]*list(selected_question_city.values())[12]+df_city_sumary[list(selected_question_city.keys())[13]]*list(selected_question_city.values())[13]+df_city_sumary[list(selected_question_city.keys())[16]]*list(selected_question_city.values())[16]
    df_city_sumary['Collaboration'] = df_city_sumary[list(selected_question_city.keys())[17]]*list(selected_question_city.values())[17]
    df_city_sumary['Water Security'] = df_city_sumary[list(selected_question_city.keys())[18]]*list(selected_question_city.values())[18]+df_city_sumary[list(selected_question_city.keys())[19]]*list(selected_question_city.values())[19]
    df_city_sumary['Water Security'] = df_city_sumary[list(selected_question_city.keys())[20]]*list(selected_question_city.values())[20]+df_city_sumary[list(selected_question_city.keys())[21]]*list(selected_question_city.values())[21]

    # %% [code]
    df_city_2020_2_1 = df_city_2020[df_city_2020['Row Name'].apply(lambda x: True if x in selected_question_city.keys() else False)]
    df_city_2020_3_1 = df_city_2020_2_1[df_city_2020_2_1['Response Answer'].apply(lambda x: True if x in selected_response else False)]

    df_city_sumary_1 = pd.pivot_table(df_city_2020_3_1,
                                    index=['Organization'], columns=['Row Name'],
                   values='Question Number', aggfunc='count')

    for i in df_city_sumary_1.columns:
        df_city_sumary_1[i] = df_city_sumary_1[i].apply(lambda x: 1 if x>=1 else 0)

    df_city_sumary_1['Food'] = df_city_sumary_1[list(selected_question_city.keys())[22]]*list(selected_question_city.values())[22]+df_city_sumary_1[list(selected_question_city.keys())[23]]*list(selected_question_city.values())[23]+df_city_sumary_1[list(selected_question_city.keys())[24]]*list(selected_question_city.values())[24] + df_city_sumary_1[list(selected_question_city.keys())[25]]*list(selected_question_city.values())[25]
    df_city_sumary_1['Waste Management'] = df_city_sumary_1[list(selected_question_city.keys())[26]]*list(selected_question_city.values())[26]+df_city_sumary_1[list(selected_question_city.keys())[27]]*list(selected_question_city.values())[27]+df_city_sumary_1[list(selected_question_city.keys())[28]]*list(selected_question_city.values())[28]+df_city_sumary_1[list(selected_question_city.keys())[29]]*list(selected_question_city.values())[29]+df_city_sumary_1[list(selected_question_city.keys())[30]]*list(selected_question_city.values())[30]

    # %% [code]
    all_city = pd.DataFrame(index=np.unique(df_city_2020['Organization']))
    KPI = ['Adaption Planning and Governance',
            'Emission Data and Other Resources',
            'Climate Hazards',
            'Climate Risk and Vulnerability Assessment',
            'Climate Action and GHG Emission Mitigation Planning and Target Setting',
            'Collaboration',
            'Water Security',
            'Water Security',
            'Food', 
            'Waste Management']

    city_KPI = all_city.merge(df_city_sumary[list(set(df_city_sumary.columns).intersection(set(KPI)))], how='left', left_index=True, right_index=True)
    city_KPI = city_KPI.merge(df_city_sumary_1[list(set(df_city_sumary_1.columns).intersection(set(KPI)))], how='left', left_index=True, right_index=True)
    
    if city_name is None:
        print(sn.heatmap(city_KPI.head()));
    else:
        try:
            print(sn.heatmap(city_KPI.loc[city_name]));
        except:
            print('City is not present in the data')   
    # %% [code]
    #print('Below are the city KPIs')

