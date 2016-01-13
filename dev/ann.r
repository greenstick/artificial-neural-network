#!/usr/bin/env Rscript

# # # # # # # # # # # 
# Utility Functions #
# # # # # # # # # # #

# 
# Vector Functions
# 

    # Insert an Element into a Vector at a Specified Position (position 0 default)
    .insert              <- function (vector, element, position = 1) {
        length <- length(vector)
        if (position == 1) {
            return (c(element, vector[1:length]))
        } else if (position > length) {
            return (c(vector[1:length], element))
        } else {
            return (c(vector[1:(position - 1)], element, vector[position:length]))
        }
    }

    # Generates a Numeric Vector of Length n with a Specified Value or Random Values Between Lower & Upper Bounds (default behavior)
    .generateVector      <- function (n, value = FALSE, lower = -0.5, upper = 0.5) {
        if (is.numeric(value == FALSE)) {
            print("Error: .generateVector() - Value Should Be a Numeric Type")
            v = FALSE
        } else if (identical(value, FALSE)) {
            v <- c(runif(n, min = lower, max = upper))
        } else {
            v <- c(rep(value, n))
        }
        v
    }

# 
# Matrix Functions
# 


    # Normalize Matrix Columns (safe parameter ignores columns that sum to 0, set to FALSE for a small gain in speed when max value of all columns is known to be > 0)
    .normalizeMatrixCols   <- function (m, safe = TRUE) {
        if (safe == TRUE) {
            for (i in 1:ncol(m)) {
                if (sum(m[,i]) > 0) {
                    m[,i] <- (m[,i] - min(m[,i])) / (max(m[,i]) - min(m[,i]))
                }
            }
        } else {
            m <- m %*% diag(1 / colSums(m))
        }
        m
    }

    # Normalize Matrix Rows (safe parameter ignores rows that sum to 0, set to FALSE for a small gain in speed when max value of all rows is known to be > 0)
    .normalizeMatrixRows    <- function (m, safe = TRUE) {
        if (safe == TRUE) {
            for (i in 1:nrow(m)) {
                if (sum(m[i,]) > 0) {
                    m[i,] <- (m[i,] - min(m[i,])) / (max(m[i,]) - min(m[i,]))
                }
            }
        } else {
            m <- m %*% diag(1 / rowSums(m))
        }
        m
    }

    # Rotate matrix 90 degrees clockwise
    .rotateMatrix           <- function (m, clockwise = TRUE) {
        if (clockwise == TRUE) {
            m <- t(m[nrow(m):1,])
        } else {
            m <- t(m)[ncol(m):1,]
        }
        m
    }

# 
# Pretty Print Formatting Functions
# 

    # How Many Decimals?
    .decimals            <- function (x, d) {
        y <- format(round(x, d), nsmall = d)
        y
    }

# 
# Kernels / Activation Functions
# 

    # Sigmoid function (Default Kernel) 
    sigmoid             <- function (x) {
        y <- 1 / (1 + exp(-x))
        y
    }

    # Signum function (Experimental)
    signum              <- function (x) {
        y <- sign(x)
        y[y == 0] <- 1
        y
    }

    # Fast Sigmoid Approximation (Experimental)
    fSigmoid            <- function (x) {
        y <- x / (1 + abs(x))
        y
    }

    # Rectified Linear Unit (Experimental)
    relu                <- function (x) {
        y <- log(1 + exp(x))
        y
    }

    # SoftMax (Experimental)
    softmax             <- function (x, lamda = 2) {
        y <- 1 / (1 + exp((x - mean(x))/(lamda * sd(x) / 2 * pi)))
        y
    }

# # # # # # # # # # # # # # # 
# Artificial Neural Network #
# # # # # # # # # # # # # # #

# 
# Artifical Neural Network Training Method
# 

# Params - 
#
# ANN.train.data            required - matrix of numerics                               data to train on
# ANN.train.classes         required - vector of integers                               known sample classes (or values in regression)
# ANN.train.epochs          optional - integer                  default = 1             number of ANN.train.epochs to train
# ANN.train.kernel          optional - string                   default = "sigmoid"     activation function (sigmoid, signum, fSigmoid, relu)
# ANN.train.etaW            optional - float                    default = 0.1           weights weights learning parameter
# ANN.train.etaH            optional - float                    default = 0.01          hidden layer learning parameter
# ANN.train.alphaW          optional - float                    default = 0             weights layer momentum parameter
# ANN.train.alphaH          optional - float                    default = 0             hidden layer momentum parameter
# ANN.train.annealing       optional - boolean                  default = FALSE         (experimental) reduces eta value(s) using the following formula: eta = eta - (eta * (1 - successfulEpochClassifications / totalEpochClassifications))
# ANN.train.hidden.nodes    optional - integer                  default = 1             nodes in hidden layer
# ANN.train.dropout         optional - float                    default = 0.5           (not implemented) proportion of neurons to drop out (1.0 = no dropout)
# ANN.train.weights.limit   optional - numeric                  default = 0.05          starting weights limit for weight layer
# ANN.train.hidden.limit    optional - numeric                  default = 0.5           starting weights limit for hiden layer
# ANN.train.classification  optional - boolean                  default = TRUE          train classification (TRUE) or regression model (FALSE)
# ANN.train.verbose         optional - boolean                  default = FALSE         display verbose console output

ANN.train                   <- function (ANN.train.data, ANN.train.classes, ANN.train.epochs = 1, ANN.train.kernel = "sigmoid", ANN.train.etaW = 0.1, ANN.train.etaH = 0.01, ANN.train.alphaW = 0, ANN.train.alphaH = 0, ANN.train.annealing = FALSE, ANN.train.hidden.nodes = 20, ANN.train.dropout = 0.5, ANN.train.weights.limit = 0.05, ANN.train.hidden.limit = 0.5, ANN.train.classification = TRUE, ANN.train.verbose = FALSE) {
    # Configuration 
    ANN.train.classes.unique    <- length(unique(ANN.train.classes))
    ANN.train.data.ncol         <- ncol(ANN.train.data)
    ANN.train.data.nrow         <- nrow(ANN.train.data)
    ANN.train.input.weights      <- list()
    ANN.train.hidden.weights    <- list()
    ANN.train.targets           <- matrix(0, nrow = nrow(ANN.train.data), ncol = ANN.train.classes.unique)
    kernelFunction              <- match.fun(ANN.train.kernel)  
    # Generate Dummy Variables for Classification
    for (i in 1:nrow(ANN.train.targets)) {
        ANN.train.targets[i,ANN.train.classes[i]] <- 1
    }
    # Generate n - 1 
    for (i in 1:ANN.train.classes.unique) {
        ANN.train.input.weights[[i]]     <- matrix(runif(((ANN.train.data.ncol) * ANN.train.hidden.nodes), min = - ANN.train.weights.limit, max = ANN.train.weights.limit), nrow = ANN.train.data.ncol, ncol = ANN.train.hidden.nodes, byrow = T)
        ANN.train.hidden.weights[[i]]   <- matrix(.generateVector(ANN.train.hidden.nodes + 1, lower = - ANN.train.hidden.limit, upper = ANN.train.hidden.limit), nrow = ANN.train.hidden.nodes + 1, ncol = 1, byrow = T) # Add node for bias term
    }
    # Epoch Loop   
    print("=========================================")
    print("-- Training . . .")
    print("=========================================")  
    for (n in 1:ANN.train.epochs) {
        # Configure Epoch
        ANN.train.classifications   <- matrix(0, nrow = length(ANN.train.classes), ncol = ANN.train.classes.unique)
        iRandomization              <- sample(ANN.train.data.nrow)
        ANN.train.metrics.hit       <- 0
        ANN.train.metrics.miss      <- 0
        ANN.train.metrics.total     <- 0
        if (ANN.train.verbose == TRUE) {
            print("=========================================")
            print(paste("-- Epoch", n, "Start"))
            print("=========================================")
        }
        for (i in iRandomization) {
            # Sample Loop     
            ANN.train.metrics.total <- ANN.train.metrics.total + 1
            cHit                    <- 0
            cMiss                   <- 0
            trainTarget             <- order(ANN.train.targets[i,], decreasing = TRUE)[1]
            if (ANN.train.verbose == TRUE) {
                print("-- New Sample ---------------------------")
                print(paste("       Sample", i, "of Epoch", n))
            }
            for (j in 1:ANN.train.classes.unique) {
                if (ANN.train.verbose == TRUE) {
                    print(paste("-- Class", j, "Network"))
                    print("   Forward-Propagating...")
                }
                # Forward Propagation
                ANN.train.layer.hidden  <- kernelFunction(ANN.train.data[i,] %*% ANN.train.input.weights[[j]])
                ANN.train.layer.hidden  <- c(.insert(ANN.train.layer.hidden, element = 1)) # Insert Bias Term to Hidden Layer
                ANN.train.sample.output <- kernelFunction(ANN.train.layer.hidden %*% ANN.train.hidden.weights[[j]])
                ANN.train.classifications[i, j] <- ANN.train.sample.output
                # Metric Computation & Communication   
                ANN.train.sample.error  <- 0.5 * (ANN.train.targets[i, j] - ANN.train.sample.output) ^ 2 # SE For Monitoring Error Reduction
                ANN.train.sample.distance <- abs(ANN.train.targets[i, j] - ANN.train.sample.output) # An Easily Interpretable Error Measure (Raw Distance)
                if (ANN.train.verbose == TRUE) {
                    print(paste("       Known Class:           ", trainTarget))
                    print(paste("       Computed Value:        ", .decimals(ANN.train.sample.output, 8)))
                    print(paste("       Raw Distance:          ", .decimals(ANN.train.sample.distance, 8)))
                    print(paste("       Computed SSE:          ", .decimals(ANN.train.sample.error, 8)))
                    print("   Back-Propagating...")
                }
                # Back Propagation
                delta.w                 <- drop((ANN.train.targets[i, j] - ANN.train.sample.output) * ANN.train.sample.output * (1 - ANN.train.sample.output))
                delta.h                 <- (ANN.train.layer.hidden * (1 - ANN.train.layer.hidden) * ANN.train.hidden.weights[[j]] * delta.w)[-c(1)] # Compute delta.h & Remove Bias Term
                change.hidden           <- ANN.train.etaH * ANN.train.layer.hidden * delta.w
                updatedHidden           <- as.matrix(ANN.train.layer.hidden) + as.matrix(change.hidden)
                ANN.train.layer.hidden  <- updatedHidden
                change.input            <- ANN.train.etaW * t(ANN.train.data[i,, drop = FALSE]) %*% delta.h
                ANN.train.input.weights[[j]] <- ANN.train.input.weights[[j]] + change.input
            }
            ANN.train.sample.class.computed <- order(ANN.train.classifications[i,], decreasing = TRUE)[1]
            if (trainTarget == ANN.train.sample.class.computed) {
                ANN.train.metrics.hit                 <- ANN.train.metrics.hit + 1
                if (ANN.train.verbose == TRUE) print("       classification Status:  Hit! :)")
            } else {
                ANN.train.metrics.miss                <- ANN.train.metrics.miss + 1
                if (ANN.train.verbose == TRUE) print("       Classification Status:  Miss :(")
            }
            if (ANN.train.verbose == TRUE) {
                print(paste("           Epoch Hits / Total:", ANN.train.metrics.hit, "/", ANN.train.metrics.total))
                print(paste("           Epoch Hit Percent: ", .decimals((ANN.train.metrics.hit/ANN.train.metrics.total) * 100, 2)))
                print("-- Sample Done --------------------------")
            }
        }
        ANN.train.metrics.accuracy    <- ANN.train.metrics.hit / ANN.train.metrics.total
        # Annealing
        if (ANN.train.annealing == TRUE) {
            ANN.train.etaW                <- ANN.train.etaW - (ANN.train.etaW * (1 - ANN.train.metrics.accuracy))
            ANN.train.etaH                <- ANN.train.etaH - (ANN.train.etaH * (1 - ANN.train.metrics.accuracy))
        }
        if (ANN.train.verbose == TRUE) {
            print("=========================================")
            print(paste("-- Epoch", n, "Done"))
            print("=========================================")
        } else {
            print(paste("-- Epoch", n, "/", ANN.train.epochs))
            print(paste("       Hits / Total:", ANN.train.metrics.hit, "/", ANN.train.metrics.total))
            print(paste("       Hit Percent: ", .decimals((ANN.train.metrics.accuracy) * 100, 2)))
        }
    }
    print("=========================================")
    print("-- Training Complete")
    print("=========================================") 
    print("-- Report -------------------------------")
    print("   Rounded Hits Last Epoch:")
    print(paste("       Train Hits / Total:     ", ANN.train.metrics.hit, "/", ANN.train.metrics.total))
    print(paste("       Train Hit Percent:      ", .decimals((ANN.train.metrics.hit/ANN.train.metrics.total) * 100, 2))) 
    # Return Results
    list (
        classes.unique  = ANN.train.classes.unique,
        inferences      = ANN.train.classifications,
        input.weights   = ANN.train.input.weights,   
        hidden.weights  = ANN.train.hidden.weights, 
        kernel          = ANN.train.kernel
    )
}

# 
# Artifical Neural Network Validation Method
# 

# PARAMS:
# 
# ANN.valid.data                        required - matrix of numerics                               validation data 
# ANN.valid.classes                     required - vector of integers                               known sample classes (or values in regression)
# ANN.valid.classes.unique              required - integer                                          number of unique classes in training data
# ANN.valid.input.weights.calibrated     required - matrix of numerics                               calibrated weights matrix from trained model
# ANN.valid.hidden.weights.calibrated   required - matrix of numerics                               calibrated hidden layer from trained model
# ANN.valid.kernel                      required - string                                           kernel function from trained model
# ANN.valid.verbose                     optional - boolean                  default = FALSE         display verbose console output 

ANN.validate                <- function (ANN.valid.data, ANN.valid.classes, ANN.valid.classes.unique, ANN.valid.input.weights.calibrated, ANN.valid.hidden.weights.calibrated, ANN.valid.kernel, ANN.valid.verbose = FALSE) {
    # Configure
    kernelFunction              <- match.fun(ANN.valid.kernel) 
    ANN.valid.data.nrow         <- nrow(ANN.valid.data)
    ANN.valid.classes.predicted <- vector()
    ANN.valid.targets           <- matrix(0, nrow = length(ANN.valid.classes), ncol = ANN.valid.classes.unique)
    ANN.valid.classifications   <- matrix(0, nrow = nrow(ANN.valid.data), ncol = ANN.valid.classes.unique)
    ANN.valid.metrics.distance  <- matrix(0, nrow = nrow(ANN.valid.data), ncol = ANN.valid.classes.unique)
    ANN.valid.metrics.se        <- matrix(0, nrow = nrow(ANN.valid.data), ncol = ANN.valid.classes.unique)
    ANN.valid.metrics.hit       <- 0
    ANN.valid.metrics.miss      <- 0
    ANN.valid.metrics.total     <- 0
    # Generate Dummy Variables for Classification
    for (i in 1:nrow(ANN.valid.targets)) {
        ANN.valid.targets[i,ANN.valid.classes[i]] <- 1
    }
    print("=========================================")
    print("-- Validating . . .")
    print("=========================================")
    for (i in 1:ANN.valid.data.nrow) {
        ANN.valid.metrics.total     <- ANN.valid.metrics.total + 1
        ANN.valid.sample.target     <- order(ANN.valid.targets[i,], decreasing = TRUE)[1]
        if (ANN.valid.verbose == TRUE) {
            print("-- New Sample ---------------------------")
            print(paste("       Validation Sample", i))
        }
        for (j in 1:ANN.valid.classes.unique) {
            # Forward Propagation
            ANN.valid.layer.hidden      <- kernelFunction(ANN.valid.data[i,] %*% ANN.valid.input.weights.calibrated[[j]])
            ANN.valid.layer.hidden      <- c(.insert(ANN.valid.layer.hidden, element = 1.0)) # Insert Bias Term to Hidden Layer
            ANN.valid.output            <- kernelFunction(ANN.valid.layer.hidden %*% ANN.valid.hidden.weights.calibrated[[j]])
            ANN.valid.classifications[i, j]<- ANN.valid.output
            # Metric Computation     
            ANN.valid.metrics.se[i, j]  <- 0.5 * (ANN.valid.targets[i, j] - ANN.valid.output) ^ 2 # For Monitoring Error Reduction
            ANN.valid.metrics.distance[i, j]<- abs(ANN.valid.targets[i, j] - ANN.valid.output) # An Easily Interpretable Error Measure
        }
        ANN.valid.sample.class.computed <- order(ANN.valid.classifications[i,], decreasing = TRUE)[1]
        if (ANN.valid.verbose == TRUE) {
            print(paste("       Known Class:           ", ANN.valid.sample.target))
            print(paste("       Computed Class:        ", ANN.valid.sample.class.computed))
        }
        if (ANN.valid.sample.target == ANN.valid.sample.class.computed) {
            ANN.valid.metrics.hit       <- ANN.valid.metrics.hit + 1
            if (ANN.valid.verbose == TRUE) print("       classification Status:  hit! :)")
        } else {
            ANN.valid.metrics.miss      <- ANN.valid.metrics.miss + 1
            if (ANN.valid.verbose == TRUE) print("       Classification Status:  miss :(")
        }
        if (ANN.valid.verbose == TRUE) {
            print(paste("           Valid Hits / Total: ", ANN.valid.metrics.hit, "/", ANN.valid.metrics.total))
            print(paste("           Valid Hit Percent:  ", .decimals((ANN.valid.metrics.hit/ANN.valid.metrics.total) * 100, 2)))
            print("-- Sample Done --------------------------")
        }
    }
    for (i in 1:nrow(ANN.valid.classifications)) {
        ANN.valid.classes.predicted <- c(ANN.valid.classes.predicted, order(ANN.valid.classifications[i,], decreasing = TRUE)[1])
    }
    print("=========================================")
    print("-- Validation Complete")
    print("=========================================")
    print("-- Report")
    print("   Rounded Hits:")
    print(paste("       Valid Hits / Total:     ", ANN.valid.metrics.hit, "/", ANN.valid.metrics.total))
    print(paste("       Valid Hit Percent:      ", .decimals((ANN.valid.metrics.hit/ANN.valid.metrics.total) * 100, 2)))
    # Return Results
    list (
        inferences      = ANN.valid.classifications,
        classes         = ANN.valid.classes.predicted,
        classes.unique  = ANN.valid.classes.unique,
        se              = ANN.valid.metrics.se,
        distance        = ANN.valid.metrics.distance,
        hits            = ANN.valid.metrics.hit,
        misses          = ANN.valid.metrics.miss,
        total           = ANN.valid.metrics.total,
        percent         = .decimals((ANN.valid.metrics.hit/ANN.valid.metrics.total) * 100, 2)
    )
}

# 
# Artifical Neural Network Classification Method
# 

# PARAMS:
# 
# ANN.classify.data                         required - matrix of numerics                           validation data 
# ANN.classify.classes.unique               required - integer                                      number of unique classes in training data
# ANN.classify.input.weights.calibrated     required - matrix of numerics                           calibrated weights matrix from trained model
# ANN.classify.hidden.weights.calibrated    required - matrix of numerics                           calibrated hidden layer from trained model
# ANN.classify.kernel                       required - string                                       kernel function from trained model
# ANN.classify.verbose                      optional - boolean                  default = FALSE     display verbose console output 

ANN.classify                <- function (ANN.classify.data, ANN.classify.classes.unique, ANN.classify.input.weights.calibrated, ANN.classify.hidden.weights.calibrated, ANN.classify.kernel, ANN.classify.verbose = TRUE) {
    # Configure
    kernelFunction              <- match.fun(ANN.classify.kernel) 
    ANN.classify.data.nrow      <- nrow(ANN.classify.data)
    ANN.classify.classifications<- matrix(0, nrow = nrow(ANN.classify.data), ncol = ANN.classify.classes.unique)
    ANN.classify.classes        <- vector()
    print("=========================================")
    print("-- Classifying . . .")
    print("=========================================")
    for (i in 1:ANN.classify.data.nrow) {
        if (ANN.classify.verbose == TRUE) {
            print("-- New Sample ---------------------------")
            print("   Forward-Propagating...")
            print(paste("       Test Sample", i))
        }
        for (j in 1:ANN.classify.classes.unique) {
            # Forward Propagation
            ANN.classify.layer.hidden           <- kernelFunction(ANN.classify.data[i,] %*% ANN.classify.input.weights.calibrated[[j]])
            ANN.classify.layer.hidden           <- c(.insert(ANN.classify.layer.hidden, element = 1.0)) # Insert Bias Term to Hidden Layer
            ANN.classify.output                 <- kernelFunction(ANN.classify.layer.hidden %*% ANN.classify.hidden.weights.calibrated[[j]])
            ANN.classify.classifications[i, j]  <- ANN.classify.output
        }
        # Metric Computation & Communication      
        if (ANN.classify.verbose == TRUE) {
            print(paste("       Computed Class:        ", order(ANN.classify.classifications[i,], decreasing = TRUE)[1]))
            print(paste("       Computed Likelihood:   ", max(ANN.classify.classifications[i,])))
            print("-- Sample Done --------------------------")
        }
    }
    for (i in 1:nrow(ANN.classify.classifications)) {
        ANN.classify.classes    <- c(ANN.classify.classes, order(ANN.classify.classifications[i,], decreasing = TRUE)[1])
    }
    print("=========================================")
    print("-- Classifications Complete")
    print("=========================================")
    # Return Results
    list (
        inferences  = ANN.classify.classifications,
        classes     = ANN.classify.classes
    )
}

# # # # # # # # #
# Sample Usage  #
# # # # # # # # #

# Get Time
time                    <- gsub(" ", "_", gsub(":", "-", Sys.time()))

# Import Data - 28x28 Images
print("Loading data . . .")
data.train.raw          <- read.csv("data/train.csv")
data.test.raw           <- read.csv("data/test.csv")
print("Done")

# 
# Feature Engineering
# 

print("Generating Features . . .")

# Generate 4x4 'Mega' Pixels with Saturation
for (i in 1:nrow(data.train.raw)) {
    col <- vector()
    for (j in 0:27) {
        name        <- paste("mps", j + 1, sep = "")
        a           <- (j * 28) + 2
        b           <- ((j + 1) * 28) + 1
        data.train.raw[i, paste(name)] <- sum(data.train.raw[i,a:b])
    }
}

for (i in 1:nrow(data.test.raw)) {
    col <- vector()
    for (j in 0:27) {
        name        <- paste("mps", j + 1, sep = "")
        a           <- (j * 28) + 2
        b           <- ((j + 1) * 28) + 1
        data.test.raw[i, paste(name)] <- sum(data.test.raw[i,a:b])
    }
}
print("Done")

# 
# Data Preparation
# 

# Get Dimensions
data.train.rowCount     <- nrow(data.train.raw)
data.train.colCount     <- ncol(data.train.raw)

# Split Data into 90:10 Train & Validation Sets
data.split              <- data.train.rowCount * 0.9
data.train              <- data.train.raw[0:data.split,]
data.valid              <- data.train.raw[data.split:data.train.rowCount,]

# Extract Known Classes (0 - 9); Increment by 1 for 1-Based Indexing in R (1 - 10) (because we're actually classifying digits!)
data.train.classes      <- data.train$label + 1
data.valid.classes      <- data.valid$label + 1

# Remove Known Classes From Train & Validation Sets
data.train              <- as.matrix(data.train)[,2:data.train.colCount]
data.valid              <- as.matrix(data.valid)[,2:data.train.colCount]
data.test               <- as.matrix(data.test.raw)

# Normalize Matrices
data.train              <- .normalizeMatrixCols(data.train)
data.valid              <- .normalizeMatrixCols(data.valid)
data.test               <- .normalizeMatrixCols(data.test)

# 
# Train / Validation Statistical Comparison
# 

# Print Summaries
print("")
print("Train Data Summary:")
summary(data.train.classes)
print("")
print("Validation Data Summary:")
summary(data.valid.classes)

# Test For Significant Differences
t.test(data.train.classes, data.valid.classes)
var.test(data.train.classes, data.valid.classes)

# 
# ANN Model Training / Validation / Classification 
# 

# Model Training
trainedModel            <- ANN.train (
    ANN.train.data                          = data.train,
    ANN.train.classes                       = data.train.classes,
    ANN.train.epochs                        = 1,
    ANN.train.kernel                        = "sigmoid",
    ANN.train.etaW                          = 0.02,
    ANN.train.etaH                          = 0.2,
    ANN.train.alphaW                        = 0,
    ANN.train.alphaH                        = 0,
    ANN.train.annealing                     = TRUE,
    ANN.train.hidden.nodes                  = 2500,
    ANN.train.weights.limit                 = 0.05,
    ANN.train.hidden.limit                  = 0.05,
    ANN.train.classification                = TRUE,
    ANN.train.verbose                       = FALSE
)

# Model Validation
validatedModel          <- ANN.validate (
    ANN.valid.data                          = data.valid, 
    ANN.valid.classes                       = data.valid.classes,
    ANN.valid.classes.unique                = trainedModel$classes.unique,
    ANN.valid.input.weights.calibrated      = trainedModel$input.weights, 
    ANN.valid.hidden.weights.calibrated     = trainedModel$hidden.weights,
    ANN.valid.kernel                        = trainedModel$kernel,
    ANN.valid.verbose                       = FALSE
)

# Generate Inferences
classifications         <- ANN.classify (
    ANN.classify.data                       = data.test,
    ANN.classify.classes.unique             = trainedModel$classes.unique,
    ANN.classify.input.weights.calibrated   = trainedModel$input.weights,
    ANN.classify.hidden.weights.calibrated  = trainedModel$hidden.weights,
    ANN.classify.kernel                     = trainedModel$kernel,
    ANN.classify.verbose                    = FALSE
)

# 
# Save Inferences
# 

# Format
ImageID                 <- vector()
Label                   <- vector()

# Classes Computed as Dummy Variables (1 - 10); to get Actual Number (0 - 9) Subtract 1
for (i in 1:length(classifications$classes)) {
    ImageID                 <- c(ImageID, i)
    Label                   <- c(Label, classifications$classes[i] - 1)
}

# Save Output
inferencesCSV           <- data.frame(ImageID, Label)
fileName                <- paste("inferences/run_", time, ".csv", sep = "")
write.csv(inferencesCSV, file = fileName, row.names = FALSE, quote = FALSE)
