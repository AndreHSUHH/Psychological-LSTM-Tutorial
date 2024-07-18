library(reticulate) # Python engine in R
library(tidyverse) # Data Wrangling & Visualization
library(here)


library(keras)
library(tensorflow)

# Define hyperparameters
FLAGS <- flags(
  flag_numeric("Neurons", 40),
  flag_integer("LSTM_Layer", 2),
  flag_integer("Dense_Layer", 1),
  flag_integer("BatchSize", 16),
  flag_string("Activation Function", "relu"),
  flag_string("Recurrent AF", "tanh"),
  flag_numeric("Dropout", 0.2)
)

MinMax_Scaler <- function(x){
  Max <- max(x, na.rm = T)
  Min <- min(x, na.rm = T)
  if((Max - Min) != 0){
    Scaled <- (x - min(x)) / (max(x) - min(x))
  }
  else{
    Scaled <- 0.5
  }
  return(Scaled)
}

Sequence_Splicer <- function(df, n_steps_in, c_iv, c_av, n_steps_out = 1) {
  # Splits the multivariate time sequence
  # Creating a list for both variables
  IDs <- unique(df$ID)
  t_p <- nrow(df[df$ID == IDs[1], ])
  
  X <- array(0, dim = c(length(n_steps_in:(t_p-n_steps_out))*length(IDs), n_steps_in, ncol(df)-1))
  y <- list()
  
  for(p in 1:length(IDs)){
    df_spliced <- df[df$ID == IDs[p], ]
    y_p <- list()
    for(i in 1:(nrow(df_spliced) - n_steps_in - n_steps_out + 1)){
      lookback <- i + n_steps_in - 1
      forecast <- lookback + n_steps_out
      # Splitting the sequences into: x = past values and features, y = values ahead
      seq_x <- df_spliced[i:lookback, c_iv, drop = FALSE]
      seq_y <- df_spliced[(lookback+1):forecast, c_av, drop = FALSE]
      array_indice <- i + (length(n_steps_in:(t_p-n_steps_out)) * (p - 1))
      for(j in 1:ncol(seq_x)){
        X[array_indice, , j] <- unlist(seq_x[, j])}
      y_p[[i]] <- as.data.frame(t(seq_y))
    }
    y[[p]] <- y_p
  }
  
  
  y <- as.array(as.matrix(bind_rows(y)))
  
  return(list(X, y))
}

r_squared <- function(y_true, y_pred){
  K <- backend()
  K$sum(K$square(y_pred - K$mean(y_true))) / K$sum(y_true - K$mean(y_true))
}


Build_LSTM <- function(Neurons, input_shape, lstm_layers, dense_layers, ActivationFunction = "tanh", recurrent_activation_function = "tanh", dropout_rate = 0.5, output_dimension = 3) {
  
  # Initiate Model
  model <- keras_model_sequential()
  
  if(lstm_layers > 1){
    # Add LSTM Layer
    for(i in 1:(lstm_layers - 1)){
      model %>% 
        layer_lstm(units = Neurons, 
                   activation = ActivationFunction, 
                   recurrent_activation = recurrent_activation_function, 
                   return_sequences = TRUE) %>%
        layer_dropout(rate = dropout_rate)
    }
    
  }
  
  # Last LSTM Layer
  model %>%
    layer_lstm(units = Neurons, 
               activation = ActivationFunction,
               recurrent_activation = recurrent_activation_function)
  
  # Add Dense Layer
  for(i in 1:dense_layers){
    model %>%
      layer_dense(units = Neurons / 2)
  }
  
  # Add Output Layer
  model %>% 
    layer_dense(units = output_dimension)
  
  # and compile model
  model %>%
    compile(optimizer = "adam",
            loss = "mse", 
            metrics = keras$metrics$RootMeanSquaredError())
  
  #metrics = c(custom_metric(name = "rmse", 
  #                             metric_fn = keras$metrics$RootMeanSquaredError), 
  #               custom_metric(name = "r_squared",
  #                             metric_fn = r_squared)))
  
  
  return(model)
}


Path <- paste0(here(), "/01_data/Multistep.csv")
EMA_Data <- read.csv(Path, header = T)

EMA_Data <- data.frame(ID = unique(EMA_Data$ID)) %>%
  bind_cols(as.data.frame(contr.sum(69))) %>%
  inner_join(EMA_Data)


# Train-Test Split
Trainset <- EMA_Data %>% group_by(ID) %>% slice(1:48)
Testset <- EMA_Data %>% group_by(ID) %>% slice(33:56) # consider the 16 input time points

Trainlist <- Trainset %>% 
  group_by(ID) %>%
  # (1) Imputation
  mutate(across(69:88, ~if_else(is.na(.), mean(., na.rm = TRUE), .))) %>%
  # (2) Scaling
  mutate(across(69:88, ~MinMax_Scaler(.))) %>%
  # (3) Moving Windows
  Sequence_Splicer(n_steps_in = lookback_window, 
                   c_iv = 2:ncol(Trainset), 
                   c_av = ncol(Trainset), 
                   n_steps_out = Timepoints_to_predict)


Testlist <- Testset %>% 
  group_by(ID) %>%
  # (1) Imputation
  mutate(across(69:88, ~if_else(is.na(.), mean(., na.rm = TRUE), .))) %>%
  # (2) Scaling
  mutate(across(69:88, ~MinMax_Scaler(.))) %>%
  # (3) Moving Windows
  Sequence_Splicer(n_steps_in = lookback_window, 
                   c_iv = 2:ncol(Testset), 
                   c_av = ncol(Testset), 
                   n_steps_out = Timepoints_to_predict)

InputDimensions <- dim(Trainlist[[1]])[2:3]

LSTM_NN <- Build_LSTM(Neurons = FLAGS$Neurons, 
                      input_shape = InputDimensions, 
                      lstm_layers = FLAGS$LSTM_Layer, 
                      dense_layers = FLAGS$Dense_Layer, 
                      ActivationFunction = FLAGS$`Activation Function`, 
                      recurrent_activation_function = FLAGS$`Recurrent AF`, 
                      dropout_rate = FLAGS$Dropout, 
                      output_dimension = Timepoints_to_predict)

early_stopping <- callback_early_stopping(
  monitor = "val_root_mean_squared_error",
  patience = 20,
  mode = "max",
  restore_best_weights = TRUE
)

fit(LSTM_NN,
    x = Trainlist[[1]],
    y = Trainlist[[2]],
    batch_size = FLAGS$BatchSize,
    epochs = 1000,
    callbacks = list(early_stopping),
    validation_data = Testlist,
    verbose = 0)

Predictions <- predict(LSTM_NN, Testlist[[1]], verbose = 0)


Score <- sqrt(mean((Predictions - Testlist[[2]])^2))



FileName <- paste0(FLAGS$Neurons,"_", FLAGS$LSTM_Layer, "_", FLAGS$Dense_Layer, "_", FLAGS$BatchSize, "_", FLAGS$`Activation Function`, "_", FLAGS$`Recurrent AF`, "_", FLAGS$Dropout)

write.table(Score, file = paste0(here(), "/01_data/Hyperparameter Tuning/Scores/", FileName))

