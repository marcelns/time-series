

# clear environmment
rm(list = ls())

# import libraries
# ggplot is a excellent r graphic library
library("ggplot2")
# the forecast principles and practive package book
library("fpp2")

# US consumption expenditure  quarterly percentage changes
autoplot(uschange[,c("Consumption","Income")]) +
  ylab("% change") + xlab("Year")

View(uschange)

fit.consMR <- tslm(
  Consumption ~ Income + Production + Savings + Unemployment,
  data=uschange)
summary(fit.consMR)

0.26729/0.03721
