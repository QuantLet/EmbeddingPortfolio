source("utils.R")

run = function(config, save_dir=NULL, debug=FALSE, arima = TRUE){
  if (is.null(save_dir)){
    save = FALSE
  } else {
    save = TRUE
  }
  counter = 1
  max_cv = config$cvs - 1
  for (cv in 0:max_cv) {
    print(paste("CV to go:", max_cv - cv + 1))
    if (save) {
      cv_save_dir = file.path(save_dir, cv)
      if (!dir.exists(cv_save_dir)) {
        dir.create(cv_save_dir)
      }
    }
    # load dataset
    data = get_cv_data(config$dataset, cv, window_size = config$window_size)
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
    # Fit a ARMA-GARCH model and predict probas oos for each factor
    for (ind in 1:length(factors)) {
      # if (ind > 1){
      #   break
      # }
      factor.name = factors[ind]
      print(factor.name)
      train_data = data$train[, ind]
      test_data = data$test[, ind]
      
      # Model selection
      best_model = model_selection(train_data, config$model.params, fit_model, parallel = !debug, arima = arima)
      if (save){
        model_path = file.path(cv_save_dir, paste0(factor.name, "_model.rds"))
        if (debug) {
          print(paste("Saving model to", model_path))
        }
        saveRDS(best_model$model, file = model_path)
      }
      # Now get probas
      if (!is.null(best_model$model)) {
        # First get proba on train set for proba threshold tuning
        dist_func = get_dist_functon(best_model$model@fit$params$cond.dist)
        n.fitted = length(best_model$model@fitted)
        train_probas = unname(sapply(-best_model$model@fitted / best_model$model@sigma.t, dist_func))
        nans = nrow(train_data) - length(train_probas)
        if (nans > 0) {
          train_probas = c(rep(NaN, nans), train_probas) # c(rep(NaN, nans), as.vector(train_probas[,1]))
        }
        # Now predict probas on test set
        probas = predict_proba(train_data, test_data, config$window_size, best_model$model,
                               fit_model, next_proba, parallel = !debug, arima=arima)
        probas = probas$proba
      } else {
        train_probas = rep(NaN, length(index(train_data)))
        probas = rep(NaN, length(index(test_data)))
      }
      # Update dataframes
      train_probas = xts(train_probas, order.by = index(train_data))
      colnames(train_probas) = factor.name
      probas = xts(probas, order.by = index(test_data))
      colnames(probas) = factor.name
      cv_train_activation_probas[factor.name] = train_probas
      cv_activation_probas[factor.name] = probas
    }
    cv_train_activation_probas = xts(cv_train_activation_probas, order.by = index(train_data))
    cv_activation_probas = xts(cv_activation_probas, order.by = index(test_data))
    
    if (save){
      write.zoo(cv_train_activation_probas,
                file = file.path(cv_save_dir, "train_activation_probas.csv"),
                sep = ",")
      write.zoo(cv_activation_probas,
                file = file.path(cv_save_dir, "activation_probas.csv"),
                sep = ",")
    }
    
    if (!is.null(config$n_factors)){
      # Update with only the latest train dates (last month)
      if (nrow(train_activation_probas) > 0) {
        cv_train_activation_probas = cv_train_activation_probas[index(cv_train_activation_probas) > last_train_date,]
      }
      train_activation_probas = rbind(train_activation_probas, cv_train_activation_probas)
      train_activation_probas = as.xts(train_activation_probas)
      
      activation_probas = rbind(activation_probas, cv_activation_probas)
      activation_probas = as.xts(activation_probas)
      print("Train probas")
      print(tail(train_activation_probas))
      print("Test probas")
      print(tail(activation_probas))
      print("NaNs CV:")
      print(paste("Train:", sum(is.na(cv_train_activation_probas))))
      print(paste("Test:", sum(is.na(cv_activation_probas))))
      
      last_train_date = index(data$train)[nrow(data$train)]
    }
    
    # Finally
    counter = counter + 1
    
  }
  return (list(train=train_activation_probas, test=activation_probas))
}
