source("R_functions/prediction_overview.R")
source("R_functions/confussion_matrix.R")

# Function to sanitize filenames
save_summary <- function(model, f_name, path_save, path_data, apendix=FALSE) {
    # Extract the formula from the model
    model_formula <- formula(model)
    # Convert the formula to a string
    formula_string <- deparse(model_formula)
    # Replace the tilde (~) with a hyphen (-)
    formula_text <- gsub("~", "-", formula_string)
    # If formula_text has more rows, collapse it
    formula_text <- paste(formula_text, collapse = "")
    
    # get model summary
    model_summary <- summary(model)
    # Capture the model summary
    summary_output <- capture.output(model_summary)

    # Make apendix 
    apendix_new <- ""
    if (apendix) {  # Ensure apendix is TRUE
        predictions_txt <- prediction_overview(model)
        confussion_matrix_txt <- confusion_matrices(model)
        apendix_new = paste0(predictions_txt, '\n', confussion_matrix_txt)
        if (!is.null(apendix)) {
            apendix_new <- paste0(apendix, '\n', apendix_new)
        }
    }

    # Get response and predictors variables
    response <-  all.vars(formula(model))[1]
    preditors <- paste(all.vars(formula(model))[-1], collapse = "; ")
    # Add the current date
    formula_txt <- paste("fomula:", formula_text)
    response_txt <- paste("response:", response)
    predictor_txt <- paste("predictor:", preditors)
    current_date <- paste("date:", Sys.Date())
    file_name_txt <- paste("file name:", f_name)
    path_data_txt <- paste("data source path:", path_data)
    summary_output <- c(summary_output,
                        formula_txt,
                        response_txt,
                        predictor_txt,
                        current_date,
                        file_name_txt,
                        path_data_txt,
                        "", "APENDIX:", apendix_new)

    # Save to a .txt file
    path = paste0(path_save, f_name, '.txt')
    if (nchar(formula_text) < 15) {
      path = paste0(path_save, f_name, "__", formula_text, '.txt')
    }
    writeLines(summary_output, path)
    return(model_summary)
}