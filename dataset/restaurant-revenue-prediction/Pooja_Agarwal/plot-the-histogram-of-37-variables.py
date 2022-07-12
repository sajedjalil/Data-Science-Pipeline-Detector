install.packages("reshape")
library(ggplot2)
library(reshape)

myTrainData <- read.csv("../train.csv")

# ignore columns 1-4 which are 
# Id,Datefromstart,	City, Group,Type

d <- melt(myTrainData[,-c(1:4)])

ggplot(d,aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()
