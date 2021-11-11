# clear environmment
rm(list = ls())

# import libraries
# ggplot is a excellent r graphic library
library("ggplot2")

# Time series can be thought as a list of numbers indexed by time
# Time series is a ts object in R
y <- ts(c(123,39,78,52,110), start = 2012)
plot(y)

# set directory for data
setwd('C:/Users/Marcelo/Desktop/time-series/exercises/data')
# load data
load(file = "a10.rda")

View(a10)

ggseasonplot(a10, year.labels = True)

help(ggplot2)
