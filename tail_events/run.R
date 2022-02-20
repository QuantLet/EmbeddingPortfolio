rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
# libraries = c("fGarch", "FinTS", "forecast", "xts")
# lapply(libraries, function(x) if (!(x %in% installed.packages())) {
#   install.packages(x)
# })
# lapply(libraries, library, quietly = TRUE, character.only = TRUE)


source("utils.R")
# library(fGarch)
# library(xts)

libraries = c("rjson") # , "forecast", "parallel", "doParallel"
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# Input
config_path = "./config/dataset1.json"
save=TRUE
save_dir = "output"
#-------------------------------------------------
# Script
config = fromJSON(file=config_path)
run = config$run
if (run == "train") {
  data_path = config$datapath$train
}
if (run == "test") {
  data_path = config$datapath$test
}

if (save){
  dataset = strsplit(config_path, "/")[[1]][length(strsplit(config_path, "/")[[1]])]
  dataset = strsplit(dataset, ".json")[[1]]
  cwd = getwd()
  save_dir = file.path(cwd, save_dir)
  save_dir = file.path(save_dir, dataset)
  save_dir = file.path(save_dir, run)
  
  if (!dir.exists(save_dir)){
    dir.create(save_dir)
  }
  time <- Sys.time()
  save_path = gsub(' ', '', gsub('-', '', gsub(':', '', time)))
  save_path = paste0(save_path, "_activation_probas.csv")
  save_path = file.path(save_dir, save_path)
} else {
  save_path = NULL
}

predict_proba_wrapper = function(train_data, val_data, config, fit_model, next_proba){
  # Find best model on train set
  best_model = model_selection(train_data, config$model.params, fit_model)
  
  probas = predict_proba(train_data, val_data, config$window_size, best_model$model, 
                         fit_model, next_proba)
  
  return (probas)
}

t1 = Sys.time()
counter = 1
for (cv in 1:length(config$data_specs)){
  print(cv)
  # load dataset
  data_spec = config$data_specs[[cv]]
  data = get_cv_data(data_path,
                     data_spec$start, 
                     data_spec$val_start, 
                     data_spec$end,
                     config$window_size)
  factors = colnames(data$train)
  cv_activation_probas = setNames(data.frame(matrix(ncol = length(factors), 
                                                    nrow = nrow(data$val))), 
                                  factors)
  if (counter == 1) {
    activation_probas = data.frame(matrix(ncol = length(factors), nrow = 0))
  }
  
  for (ind in 1:length(factors)){
    factor.name = factors[ind]
    print(factor.name)
    train_data = data$train[,ind]
    val_data = data$val[,ind]
    
    probas = tryCatch(
      predict_proba_wrapper(train_data, val_data, config, fit_model, next_proba),
      error = function(e) data.frame(rep(NA, nrow(val_data))), 
      silent = FALSE)
    
    colnames(probas) = factor.name
    cv_activation_probas[factor.name] = probas
  }
  cv_activation_probas = xts(cv_activation_probas, order.by = index(val_data))
  activation_probas = rbind(activation_probas, cv_activation_probas)
  print(head(activation_probas))
  counter = counter + 1
}
activation_probas = as.xts(activation_probas)
t2 = Sys.time()
print(paste("Total time:", t2-t1))

write.csv(activation_probas, save_path, row.names = TRUE)