import pandas as pd
import os

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print(train.head())
# # ui.R
# 
# shinyUI(fluidPage(
#   titlePanel("Text Analytics"),
#   
#   sidebarLayout(
#     sidebarPanel(
#       helpText("Please key in your thoughts so that we can analyze your mood"),
#       
#       textInput("var", label="What you think", value = "", width = NULL, placeholder = NULL),
#             selectInput("grf", 
#                         label = "Choose a format to display",
#                         choices = c("Polarity Vs Time", "Number of tweets Vs Polarity",
#                                     "No of tweets Vs emotion","Word cloud"),
#                         selected = "Polarity Vs Time"),
#           selectInput("oper", 
#                   label = "Choose an operator to display",
#                   choices = c("Airtel", "Rcom",
#                               "Vodafone","Lebara"),
#                   selected = "Airtel")
#       
#       
# #       sliderInput("range", 
# #                   label = "Range of interest:",
# #                   min = 0, max = 100, value = c(0, 100))
#       ),
#     
#     mainPanel(
#      tabsetPanel(
#         tabPanel('Know the mood of the customer', imageOutput("image2")),
#         tabPanel('Plots', plotOutput("gPlot"))
#         # tabPanel('Test', textOutput("ouput$text"))
#         
#       )
#           
#     )
#   )
# ))