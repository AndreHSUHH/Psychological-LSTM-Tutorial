

# Check if TensorFlow is installed
python_installed <- reticulate::py_available()
tf_installed <- reticulate::py_module_available("tensorflow")
keras_installed <- reticulate::py_module_available("keras")

if(!python_installed){
  reticulate::install_python()
}

if(!tf_installed){
  tensorflow::install_tensorflow(envname = "r_tensorflow")
}
if(!keras_installed){
  keras::install_keras()
}