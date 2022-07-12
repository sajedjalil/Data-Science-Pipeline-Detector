import pandas as pd
import os

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print(train.head())
# # server.R
# 
# shinyServer(
#   function(input, output) {
#     library("installr", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("RCurl", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("ROAuth", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("pacman", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("base64enc", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("bit", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("bit64", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("rjson", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("DBI", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("twitteR", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("plyr", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("RColorBrewer", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("dichromat", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("munsell", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("gtable", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("reshape2", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("scales", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("ggplot2", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("slam", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("wordcloud", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("NLP", lib.loc="C:/Program Files/R/R-3.2.3/library")
#     library("tm", lib.loc="C:/Program Files/R/R-3.1.2/library")
#     library("Rstem", lib.loc="C:/Program Files/R/R-3.1.2/library")
#     library("sentiment", lib.loc="C:/Program Files/R/R-3.1.2/library")
#     library("png", lib.loc="C:/Program Files/R/R-3.2.3/library")
# #        output$text<- renderText({ 
# #       class_pol = classify_polarity(input$var, algorithm="bayes")
# #       if(class_pol[,4]=="positive")
# #       paste("Positive")
# #       else if(class_pol[,4]=="neutral")
# #       paste("Neutral")
# #       else if(class_pol[,4]=="negative")
# #       paste("Negative")
# #    })
#     output$gPlot <- renderPlot({
#       if(input$oper=="Airtel")
#       {sent_df<-Airtel_sent_df
#       twet<-Airtel
#       twet1<-"Airtel"}
#       else if(input$oper=="Vodafone")
#       {sent_df<-vodafone_sent_df
#       twet<-vodafone
#       twet1<-"Vodafone"}
#       else if(input$oper=="Rcom")
#       {sent_df<-Rcom_sent_df
#       twet<-Rcom
#       twet1<-"Rcom"}
#       else if(input$oper=="Lebara")
#       {sent_df<-lebara_sent_df
#       twet<-lebara
#       twet1<-"Lebara"}
#       if(input$grf=="Polarity Vs Time")
#       {
#         sent_df$date<-0
#         i<-1;while(i<=nrow(sent_df)){sent_df$date[i]<-as.character(twet[[i]]$created);i<-i+1}
#         sent_df$date<-as.Date(sent_df$date)
#         ## to plot stacked graph on ggplot
#         ggplot(sent_df, aes(x=date,fill=polarity)) + geom_bar() +scale_fill_brewer(palette="RdGy") +labs(x="polarity categories", y="number of tweets") +ggtitle(print(paste("Sentiment Analysis of Tweets about",twet1,"\n(Time line)"))) +theme(plot.title = element_text(size=12, face="bold"))
#       }
#       else if(input$grf=="Number of tweets Vs Polarity")
#       {
#         ggplot(sent_df, aes(x=polarity)) +
#           geom_bar(aes(y=..count.., fill=polarity)) +
#           scale_fill_brewer(palette="RdGy") +
#           labs(x="polarity categories", y="number of tweets") +
#           ggtitle(print(paste("Sentiment Analysis of Tweets about",twet1,"\n(classification by polarity)"))) +
#           theme(plot.title = element_text(size=12, face="bold"))
#       }
#       else if(input$grf=="No of tweets Vs emotion")
#       {ggplot(sent_df, aes(x=emotion)) +
#           geom_bar(aes(y=..count.., fill=emotion)) +
#           scale_fill_brewer(palette="Dark2") +
#           labs(x="emotion categories", y="number of tweets") +
#           ggtitle(print(paste("Sentiment Analysis of Tweets about",twet1,"\n(classification by Emotion)"))) +
#           theme(plot.title = element_text(size=12, face=
#                                             "bold"))}
#       else if (input$grf=="Word cloud")
#       {# Separate the text by emotions and visualize the words with a comparison cloud
#         # separating text by emotion
#         emos = levels(factor(sent_df$emotion))
#         nemo = length(emos)
#         emo.docs = rep("", nemo)
#         for (i in 1:nemo)
#         {
#           tmp = some_txt[emotion == emos[i]]
#           emo.docs[i] = paste(tmp, collapse=" ")
#         }
#         
#         # remove stopwords
#         emo.docs = removeWords(emo.docs, stopwords("english"))
#         # create corpus
#         corpus = Corpus(VectorSource(emo.docs))
#         tdm = TermDocumentMatrix(corpus)
#         tdm = as.matrix(tdm)
#         colnames(tdm) = emos
#         comparison.cloud(tdm, colors = brewer.pal(nemo, "Dark2"),scale = c(3,.5), random.order = FALSE, title.size = 1.5)}
#     })
#      output$image2 <- renderImage({
#         #class_pol = classify_polarity(input$var, algorithm="bayes")
#         class_pol_1<-class_pol$value(input$var)
#         #class_pol$train<-data.db.trained.lexical
#         #input$ggmat<-scan(what=double (0))
# #          if (is.null(length(strsplit(input$var,split=" ")[[1]])))
# #            return()
#         if (class_pol_1[,4] == "positive") 
#         {
#               return(list(
#               src = "images.png",
#               contentType = "image/png",
#               alt = "Positive"
#             ))
#           }
#         else if (class_pol_1[,4] =="neutral")  {
#               return(list(
#               src = "neutral.jpg",
#               filetype = "image/jpeg",
#               alt = "neutral"
#             ))
#           }
#         else if (class_pol_1[,4] =="negative")  {
#               return(list(
#               src = "Negative.jpg",
#               filetype = "image/jpeg",
#               alt = "Negative"
#             ))
#           }
#         else if (class_pol_1[,4] =="sarcastic")  {
#               return(list(
#               src = "Sarcasm.jpg",
#               filetype = "image/jpeg",
#               alt = "Sarcastic"
#             ))
#           }
#       }, deleteFile = FALSE)
#   #})
#   }
# )