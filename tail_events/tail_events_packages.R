# Install and load packages
libraries = c("rjson", "fGarch", "xts", "forecast", "parallel", "doParallel") 

lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)
