rm(list = ls(all = TRUE))
graphics.off()
source("utils.R")
library(rjson)

# ------------------------ Input ------------------------
# Input
config_path = "./config/dataset1.json"
save = TRUE
save_dir = "output"
debug = FALSE


# ------------------------ Script ------------------------

config = fromJSON(file = config_path)
run = config$run
if (run == "train") {
  data_path = config$datapath$train
}
if (run == "test") {
  data_path = config$datapath$test
}

if (save) {
  dataset = strsplit(config_path, "/")[[1]][length(strsplit(config_path, "/")[[1]])]
  dataset = strsplit(dataset, ".json")[[1]]
  cwd = getwd()
  save_dir = file.path(cwd, save_dir)
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  save_dir = file.path(save_dir, dataset)
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  save_dir = file.path(save_dir, run)
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  time <- Sys.time()
  save_dir = file.path(save_dir, gsub(' ', '', gsub('-', '', gsub(':', '', time))))
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  time <- Sys.time()
  save_path = gsub(' ', '', gsub('-', '', gsub(':', '', time)))
  train_save_path = paste0(save_path, "_train_activation_probas.csv")
  train_save_path = file.path(save_dir, train_save_path)
  save_path = paste0(save_path, "_activation_probas.csv")
  save_path = file.path(save_dir, save_path)
} else {
  save_path = NULL
}

predict_proba_wrapper = function(train_data, test_data, config, fit_model, next_proba) {
  # Find best model on train set
  best_model = model_selection(train_data, config$model.params, fit_model)

  probas = predict_proba(train_data, test_data, config$window_size, best_model$model,
                         fit_model, next_proba)

  return(probas)
}

t1 = Sys.time()
counter = 1
for (cv in 1:length(config$data_specs)) {
  print(cv)
  cv_save_dir = file.path(save_dir, cv)
  if (!dir.exists(cv_save_dir)) {
    dir.create(cv_save_dir)
  }
  # load dataset
  data_spec = config$data_specs[[cv]]
  if (run == "train"){
    test_start = data_spec$val_start
    end = data_spec$test_start
  }
  if (run == "test"){
    test_start = data_spec$test_start
    end = data_spec$end
  }
  data = get_cv_data(data_path,
                     data_spec$start,
                     test_start,
                     end,
                     config$window_size)
  print(paste0("Last train: ", index(tail(data$train, 1))[1]))
  print(paste0("First test: ", index(data$test)[1]))
  
  factors = colnames(data$train)
  # Initialize tables
  if (counter == 1) {
    activation_probas = data.frame(matrix(ncol = length(factors), nrow = 0))
    train_activation_probas = data.frame(matrix(ncol = length(factors), nrow = 0))
  }
  cv_train_activation_probas = setNames(data.frame(matrix(ncol = length(factors),
                                                          nrow = nrow(data$train))),
                                        factors)
  cv_activation_probas = setNames(data.frame(matrix(ncol = length(factors),
                                                    nrow = nrow(data$test))),
                                  factors)
  for (ind in 1:length(factors)) {
    # if (ind > 1){
    #   break
    # }
    factor.name = factors[ind]
    print(factor.name)
    train_data = data$train[, ind]
    test_data = data$test[, ind]

    best_model = model_selection(train_data, config$model.params, fit_model, parallel = !debug)
    model_path = file.path(cv_save_dir, paste0(factors[ind], "model.rds"))
    saveRDS(best_model$model, file = model_path)

    # Get proba on train set
    
    if (!is.null(best_model$model)){
      dist_func = get_dist_functon(best_model$model@fit$params$cond.dist)
      n.fitted = length(best_model$model@fitted)
      train_probas = unname(sapply(-best_model$model@fitted / best_model$model@sigma.t, dist_func))
      nans = nrow(train_data) - length(train_probas)
      if (nans > 0){
        train_probas = c(rep(NaN, nans), train_probas) # c(rep(NaN, nans), as.vector(train_probas[,1]))
      }
      probas = predict_proba(train_data, test_data, config$window_size, best_model$model,
                             fit_model, next_proba, parallel = !debug)
      
    } else {
      train_probas = rep(NaN, length(index(train_data)))
      probas = rep(NaN, length(index(test_data)))
    }
    train_probas = xts(train_probas,  order.by = index(train_data))
    colnames(train_probas) = factor.name
    probas = xts(probas,  order.by = index(test_data))
    colnames(probas) = factor.name

    cv_train_activation_probas[factor.name] = train_probas
    cv_activation_probas[factor.name] = probas
  }
  cv_train_activation_probas = xts(cv_train_activation_probas, order.by = index(train_data))
  cv_activation_probas = xts(cv_activation_probas, order.by = index(test_data))
  write.zoo(cv_train_activation_probas,
            file = file.path(cv_save_dir, "train_activation_probas.csv"),
            sep = ",")
  write.zoo(cv_activation_probas,
            file = file.path(cv_save_dir, "activation_probas.csv"),
            sep = ",")
  
  if (nrow(train_activation_probas) > 0){
    cv_train_activation_probas = cv_train_activation_probas[index(cv_train_activation_probas) > last_train_date,]
  }
  
  train_activation_probas = rbind(train_activation_probas, cv_train_activation_probas)
  train_activation_probas = as.xts(train_activation_probas)
  activation_probas =  rbind(activation_probas, cv_activation_probas)
  activation_probas = as.xts(activation_probas)

  print("Train probas")
  print(tail(train_activation_probas))
  print("Test probas")
  print(tail(activation_probas))
  counter = counter + 1
  last_train_date = index(data$train)[nrow(data$train)]
  
}
t2 = Sys.time()
print(paste("Total time:", t2 - t1))

write.zoo(train_activation_probas, file = train_save_path, sep = ",")
write.zoo(activation_probas, file = save_path, sep = ",")


