# clear environmment
rm(list = ls())

# import libraries
# ggplot is a excellent r graphic library
library("ggplot2")
# the forecast principles and practive package book
library("fpp2")


# Time series can be thought as a list of numbers indexed by time
# Time series is a ts object in R
y <- ts(c(123,39,78,52,110), start = 2012)
plot(y)

# set directory for data
setwd('C:/Users/Marcelo/Desktop/time-series/exercises/data')
# load data
load(file = "a10.rda")
View(a10)

# seasonal plot
ggseasonplot(a10, year.labels = T)

# Seasonal subseries plots
ggsubseriesplot(a10) +
  ylab("$ million") +
  ggtitle("Seasonal subseries plot: antidiabetic drug sales")

# scatter plot
autoplot(elecdemand[,c("Demand","Temperature")], facets=TRUE) +
  xlab("Year: 2014") + ylab("") +
  ggtitle("Half-hourly electricity demand: Victoria, Australia")

qplot(Temperature, Demand, data=as.data.frame(elecdemand)) +
  ylab("Demand (GW)") + xlab("Temperature (Celsius)")
