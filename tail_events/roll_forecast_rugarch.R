# https://cran.r-project.org/web/packages/qrmtools/vignettes/ARMA_GARCH_VaR.html


rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
libraries = c("rugarch", "qrmtools")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)


## Model specification (for simulation)
nu <- 3 # d.o.f. of the standardized distribution of Z_t
fixed.p <- list(mu = 0, # our mu (intercept)
                ar1 = 0.5, # our phi_1 (AR(1) parameter of mu_t)
                ma1 = 0.3, # our theta_1 (MA(1) parameter of mu_t)
                omega = 4, # our alpha_0 (intercept)
                alpha1 = 0.4, # our alpha_1 (GARCH(1) parameter of sigma_t^2)
                beta1 = 0.2, # our beta_1 (GARCH(1) parameter of sigma_t^2)
                shape = nu) # d.o.f. nu for standardized t_nu innovations
armaOrder <- c(1,1) # ARMA order
garchOrder <- c(1,1) # GARCH order
varModel <- list(model = "sGARCH", garchOrder = garchOrder)
spec <- ugarchspec(varModel, mean.model = list(armaOrder = armaOrder),
                   fixed.pars = fixed.p, distribution.model = "std") # t standardized residuals

## Simulate (X_t)
n <- 1000 # sample size (= length of simulated paths)
x <- ugarchpath(spec, n.sim = n, m.sim = 1, rseed = 271) # n.sim length of simulated path; m.sim = number of paths
## Note the difference:
## - ugarchpath(): simulate from a specified model
## - ugarchsim():  simulate from a fitted object

## Extract the resulting series
X <- fitted(x) # simulated process X_t = mu_t + epsilon_t for epsilon_t = sigma_t * Z_t
sig <- sigma(x) # volatilities sigma_t (conditional standard deviations)
eps <- x@path$residSim # unstandardized residuals epsilon_t = sigma_t * Z_t
## Note: There are no extraction methods for the unstandardized residuals epsilon_t
##       for uGARCHpath objects (only for uGARCHfit objects; see below).

## Sanity checks (=> fitted() and sigma() grab out the right quantities)
stopifnot(all.equal(X,   x@path$seriesSim, check.attributes = FALSE),
          all.equal(sig, x@path$sigmaSim,  check.attributes = FALSE))

plot(X,   type = "l", xlab = "t", ylab = expression(X[t]))
plot(sig, type = "h", xlab = "t", ylab = expression(sigma[t]))
plot(eps, type = "l", xlab = "t", ylab = expression(epsilon[t]))

## Fit an ARMA(1,1)-GARCH(1,1) model
window.size = 800
X_train = X[1:window.size]
spec <- ugarchspec(varModel, mean.model = list(armaOrder = armaOrder),
                   distribution.model = "std") # without fixed parameters here
fit <- ugarchfit(spec, data = X_train) # fit
fspec <- getspec(fit) # specification of the fitted process

garchroll <- ugarchroll(fspec, data = X, n.start = window.size, 
                        window.size=window.size, refit.window = "moving", refit.every = 10)
preds <- as.data.frame(garchroll)

n = length(X)
plot(X[(length(X)-100):length(X)], type="l", col="blue")
lines(preds$Mu[(nrow(preds)-100):nrow(preds)], col="green")

# Using forecast
setfixed(fspec) <- as.list(coef(fit)) # set the parameters to the fitted ones
new_pred = c()
for (i in 1:10){
  pred <- ugarchforecast(fspec, data = X[(n-i+1):n], n.ahead = 1, n.roll = 0, out.sample = 0)
  pred = pred@forecast$seriesFor
  new_pred = c(new_pred, pred)
}


