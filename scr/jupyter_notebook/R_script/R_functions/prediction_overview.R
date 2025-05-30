# Function to generate all combinations, predictions, and counts
prediction_overview <- function(model, data=NULL) {
    if (is.null(data)) {
        data <- model$model
    }
    
    # Identify predictor variables used in the model
    predictor_names <- attr(terms(model), "term.labels")  # Extract predictors from the model
    
    # Get the levels for each predictor (both factor and continious)
    predictor_levels <- lapply(predictor_names, function(var) {
    if (is.factor(data[[var]])) {
      levels(data[[var]])
    } else {
      unique(data[[var]])
    }
    })
    
    # Create all possible combinations of predictor levels
    combinations <- expand.grid(predictor_levels)
    names(combinations) <- predictor_names
    
    # Count occurrences of each combination in the original data
    data_subset <- data[predictor_names]
    count_table <- as.data.frame(table(data_subset))
    names(count_table)[-length(names(count_table))] <- names(data_subset)
    
    # Merge the counts into the combinations table
    combinations_with_counts <- merge(combinations, count_table, by = predictor_names, all.x = TRUE)
    names(combinations_with_counts)[ncol(combinations_with_counts)] <- "count"
    
    # Replace NA counts with 0 (for combinations not present in the data)
    combinations_with_counts$count[is.na(combinations_with_counts$count)] <- 0
    
    # Generate predicted probabilities for each combination
    combinations_with_counts$predicted_probability <- predict(model,
                                                              newdata = combinations_with_counts,
                                                              type = "response")
    # Generate output text
    predicted_probs_text <- paste(capture.output(print(combinations_with_counts)),
                                collapse = "\n")
    output <- paste("\npredicted probability:\n",
                    predicted_probs_text,
                    "\n"
                    )
    
    return(output)
}