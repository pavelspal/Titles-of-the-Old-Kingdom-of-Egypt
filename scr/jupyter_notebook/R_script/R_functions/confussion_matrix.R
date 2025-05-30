# Define the function
confusion_matrices <- function(model, data=NULL, response=NULL,
                               c_values=c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
                              ) {
    if (is.null(data)) {
        data <- model$model
    }
    if (is.null(response)) {
        response <- all.vars(formula(model))[1]
    }
    
    # Predict probabilities from the model
    predicted_probs <- predict(model, data, type = "response")
    
    # Initialize an empty string to accumulate the output
    output <- ""
    
    # Loop through each cut-off value
    for (c in c_values) {
        # Generate predictions based on the current cut-off
        predicted_classes <- ifelse(predicted_probs >= c, 1, 0)
        
        # Create the confusion matrix
        confusion <- table(Predicted = predicted_classes, Actual = data[[response]])
        # Convert the confusion matrix to text using format()
        confusion_text <- paste(capture.output(print(confusion)),
                                collapse = "\n")
        
        # Create the output string for the current cut-off
        output <- paste0(output,
                         "\nConfusion Matrix for Cut-Off = ", c, ":\n",
                         confusion_text,
                         "\n"
                        )
    }

    # Return the final combined output string
    return(output)
}