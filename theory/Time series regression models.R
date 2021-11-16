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


# clear environmment again
rm(list = ls())

# US consumption expenditure  quarterly percentage changes
autoplot(uschange[,c("Consumption","Income")]) +
  ylab("% change") + xlab("Year")

View(uschange)

fit.consMR <- tslm(
  Consumption ~ Income + Production + Savings + Unemployment,
  data=uschange)
summary(fit.consMR)

checkresiduals(fit.consMR)

df <- as.data.frame(uschange)
df[,"Residuals"]  <- as.numeric(residuals(fit.consMR))
p1 <- ggplot(df, aes(x=Income, y=Residuals)) +
  geom_point()
p2 <- ggplot(df, aes(x=Production, y=Residuals)) +
  geom_point()
p3 <- ggplot(df, aes(x=Savings, y=Residuals)) +
  geom_point()
p4 <- ggplot(df, aes(x=Unemployment, y=Residuals)) +
  geom_point()
gridExtra::grid.arrange(p1, p2, p3, p4, nrow=2)

cbind(Fitted = fitted(fit.consMR),
      Residuals=residuals(fit.consMR)) %>%
  as.data.frame() %>%
  ggplot(aes(x=Fitted, y=Residuals)) + geom_point()


# expurious regression
aussies <- window(ausair, end=2011)
fit <- tslm(aussies ~ guinearice)
summary(fit)

checkresiduals(fit)

# clear environmment
rm(list = ls())
## reading data
beer2 <- window(ausbeer, start=1992)
autoplot(beer2) + xlab("Year") + ylab("Megalitres")

# modelling

fit.beer <- tslm(beer2 ~ trend + season)
summary(fit.beer)

# model result
cbind(Data=beer2, Fitted=fitted(fit.beer)) %>%
  as.data.frame() %>%
  ggplot(aes(x = Data, y = Fitted,
             colour = as.factor(cycle(beer2)))) +
  geom_point() +
  ylab("Fitted") + xlab("Actual values") +
  ggtitle("Quarterly beer production") +
  scale_colour_brewer(palette="Dark2", name="Quarter") +
  geom_abline(intercept=0, slope=1)
