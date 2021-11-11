# clear environmment
rm(list = ls())

# Time series can be thought as a list of numbers indexed by time
# Time series is a ts object in R
y <- ts(c(123,39,78,52,110), start = 2012)
plot(y)



