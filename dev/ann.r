#!/usr/bin/env Rscript

#
# Setup
#

# Get Session Information
# sessionInfo()

# Set Directory
# setwd("") 

# Set Seed
# set.seed(1337)

# 
# Some Private (.x) Utility Functions
# 

    # Insert an Element into a Vector at a Specified Position (position 0 default)
    .insert              <- function (vector, element, position = 0) {
        length <- length(vector)
        if (position == 0) {
            return (c(element, vector[0:(length)]))
        } else if (position > length) {
            return (c(vector[1:(length)], element))       
        } else {
            return (c(vector[1:(position - 1)], element, vector[(position):length]))
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
# Print Formatting Functions
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

    # Signum function (Test)
    signum              <- function (x) {
        y <- sign(x)
        y[y == 0] <- 1
        y
    }

    # Fast Sigmoid Approximation (Test)
    fSigmoid            <- function (x) {
        y <- x / (1 + abs(x))
        y
    }

    # Rectified Linear Unit (ReLU) (Test)
    relu                <- function (x) {
        y <- log(1 + exp(x))
        y
    }

    # SoftMax (Test)
    softmax             <- function (x, lamda = 2) {
        y <- 1 / (1 + exp((x - mean(x))/(lamda * sd(x) / 2 * pi)))
        y
    }

# 
# Artifical Neural Network Training Function
# 

# Params - 
#
# inputTrainingData     required - matrix of numerics                               data to train on
# trainingClasses       required - vector of integers                               known sample classes (or values in regression)
# epochs                optional - integer                  default = 20            number of epochs to train
# kernel                optional - string                   default = "sigmoid"     activation function (sigmoid, signum, fSigmoid, relu)
# etaW                  optional - float                    default = 0.1           weights learning parameter
# etaH                  optional - float                    default = 0.01          hidden layer learning parameter
# annealing             optional - float                    default = 0.0           (experimental) annealing evaluated per epoch via formula (eta = eta - (eta * annealing) for etaW & etaW
# hiddenNodes           optional - integer                  default = 1             nodes in hidden layer
# dropout               optional - float                    default = 0.5           proportion of neurons to drop out (1.0 = no dropout)
# dataWeightsLimit      optional - numeric                  default = 0.05          starting weights limit for weight layer
# hiddenWeightsLimit    optional - numeric                  default = 0.5           starting weights limit for hiden layer
# classification        optional - boolean                  default = TRUE          train classification (TRUE) or regression model (FALSE)
# verboseTrain          optional - boolean                  default = FALSE         display verbose console output 

ANN.train               <- function (inputTrainingData, trainingClasses, epochs = 1, kernel = "sigmoid", etaW = 0.1, etaH = 0.01, annealing = 0.0, hiddenNodes = 20, dropout = 0.5, dataWeightsLimit = 0.05, hiddenWeightsLimit = 0.5, classification = TRUE, verboseTrain = FALSE) {
    # Configuration 
    kernelFunction          <- match.fun(kernel)  
    uniqueClasses           <- length(unique(trainingClasses))
    inputTrainingLengthC    <- ncol(inputTrainingData)
    inputTrainingLengthR    <- nrow(inputTrainingData)
    trainingWeights         <- list()
    hiddenWeights           <- list()
    trainingTargets         <- matrix(0, nrow = nrow(inputTrainingData), ncol = uniqueClasses)
    # Generate Dummy Variables for Classification
    for (i in 1:nrow(trainingTargets)) {
        trainingTargets[i,trainingClasses[i]] <- 1
    }
    # Generate n - 1 
    for (i in 1:uniqueClasses) {
        trainingWeights[[i]]    <- matrix(runif(((inputTrainingLengthC) * hiddenNodes), min = - dataWeightsLimit, max = dataWeightsLimit), nrow = inputTrainingLengthC, ncol = hiddenNodes, byrow = T)
        hiddenWeights[[i]]      <- matrix(.generateVector(hiddenNodes + 1, lower = - hiddenWeightsLimit, upper = hiddenWeightsLimit), nrow = hiddenNodes + 1, ncol = 1, byrow = T)
    }
    # Epoch Loop   
    print("=========================================")
    print("-- Training . . .")
    print("=========================================")  
    for (n in 1:epochs) {
        # Configure Epoch
        distanceList            <- matrix(0, nrow = length(trainingClasses), ncol = uniqueClasses)
        classifications         <- matrix(0, nrow = length(trainingClasses), ncol = uniqueClasses)
        iRandomization          <- sample(inputTrainingLengthR)
        hit                     <- 0
        miss                    <- 0
        grand                   <- 0
        if (verboseTrain == TRUE) {
            print("=========================================")
            print(paste("-- Epoch", n, "Start"))
            print("=========================================")
        }
        for (i in iRandomization) {
            # Sample Loop     
            grand               <- grand + 1
            cHit                <- 0
            cMiss               <- 0
            trainTarget         <- order(trainingTargets[i,], decreasing = TRUE)[1]
            if (verboseTrain == TRUE) {
                print("-- New Sample ---------------------------")
                print(paste("       Sample", i, "of Epoch", n))
            }
            for (j in 1:uniqueClasses) {
                if (verboseTrain == TRUE) {
                    print(paste("-- Class", j, "Network"))
                    print("   Forward-Propagating...")
                }  
                # Forward Propagation
                hiddenLayer         <- kernelFunction(inputTrainingData[i,] %*% trainingWeights[[j]])
                hiddenLayer         <- c(.insert(hiddenLayer, element = 1)) # Insert Bias Term to Hidden Layer
                output              <- kernelFunction(hiddenLayer %*% hiddenWeights[[j]])
                classifications[i, j] <- output
                # Metric Computation & Communication   
                error               <- 0.5 * (trainingTargets[i, j] - output) ^ 2
                distance            <- abs(trainingTargets[i, j] - output) # An Easily Interpretable Error Measure
                distanceList[i, j]  <- distance # For Monitoring Error Reduction
                if (verboseTrain == TRUE) {
                    print(paste("       Known Class:           ", trainTarget))
                    print(paste("       Computed Value:        ", .decimals(output, 8)))
                    print(paste("       Raw Distance:          ", .decimals(distance, 8)))
                    print(paste("       Computed SSE:          ", .decimals(error, 8)))
                    print("   Back-Propagating...")
                }
                # Back Propagation
                deltaP              <- drop((trainingTargets[i, j] - output) * output * (1 - output))
                deltaH              <- (hiddenLayer * (1 - hiddenLayer) * hiddenWeights[[j]] * deltaP)[-c(1)] # Compute deltaH & Remove Bias Term
                hiddenChange        <- etaH * hiddenLayer * deltaP
                updatedHidden       <- as.matrix(hiddenLayer) + as.matrix(hiddenChange)
                hiddenLayer         <- updatedHidden
                weightChange        <- etaW * t(inputTrainingData[i,, drop = FALSE]) %*% deltaH
                updatedWeights      <- trainingWeights[[j]] + weightChange
                trainingWeights[[j]]<- updatedWeights
            }
            # Subtract 1 'Cause R Uses 1-Based Indexing
            trainClass          <- order(classifications[i,], decreasing = TRUE)[1]
            if (trainTarget == trainClass) {
                hit                 <- hit + 1
                if (verboseTrain == TRUE) print("       classification Status:  hit! :)")
            } else {
                miss                <- miss + 1
                if (verboseTrain == TRUE) print("       Classification Status:  miss :(")
            }
            if (verboseTrain == TRUE) {
                print(paste("           Epoch Hits / Total:", hit, "/", grand))
                print(paste("           Epoch Hit Percent: ", .decimals((hit/grand) * 100, 2)))
                print("-- Sample Done --------------------------")
            }
        }
        # Annealing
        etaW                <- etaW - (etaW * annealing)
        etaH                <- etaH - (etaH * annealing)
        if (verboseTrain == TRUE) {
            print("=========================================")
            print(paste("-- Epoch", n, "Done"))
            print("=========================================")
        } else {
            print(paste("-- Epoch", n, "/", epochs))
            print(paste("       Hits / Total:", hit, "/", grand))
            print(paste("       Hit Percent: ", .decimals((hit/grand) * 100, 2)))
        }
    }
    print("=========================================")
    print("-- Training Complete")
    print("=========================================") 
    print("-- Report -------------------------------")
    print("   Rounded Hits Last Epoch:")
    print(paste("       Train Hits / Total:     ", hit, "/", grand))
    print(paste("       Train Hit Percent:      ", .decimals((hit/grand) * 100, 2))) 
    # Return Results
    list (
        uniqueClasses       = uniqueClasses,
        inferences          = classifications,
        trainedWeights      = trainingWeights, 
        trainedHidden       = hiddenWeights, 
        kernel              = kernel
    )
}

# 
# Artifical Neural Network Validation Function
# 

# PARAMS:
# 
# inputValidData        required - matrix of numerics                               validation data 
# validClasses          required - vector of integers                               known sample classes (or values in regression)
# uniqueClasses         required - integer                                          number of unique classes in training data
# calibratedWeights     required - matrix of numerics                               calibrated weights matrix from trained model
# calibratedHidden      required - matrix of numerics                               calibrated hidden layer from trained model
# kernel                required - string                                           kernel function from trained model
# verboseValidate       optional - boolean                  default = FALSE         display verbose console output 

ANN.validate            <- function (inputValidData, validClasses, uniqueClasses, calibratedWeights, calibratedHidden, kernel, verboseValidate = FALSE) {
    # Configure
    kernelFunction          <- match.fun(kernel) 
    inputValidLengthR       <- nrow(inputValidData)
    validTargets            <- matrix(0, nrow = length(validClasses), ncol = uniqueClasses)
    classifications         <- matrix(0, nrow = nrow(inputValidData), ncol = uniqueClasses)
    validDistanceList       <- matrix(0, nrow = nrow(inputValidData), ncol = uniqueClasses)
    validSEList             <- matrix(0, nrow = nrow(inputValidData), ncol = uniqueClasses)
    validHit                <- 0
    validMiss               <- 0
    validGrand              <- 0
    predictedClasses        <- vector()
    # Generate Dummy Variables for Classification
    for (i in 1:nrow(validTargets)) {
        validTargets[i,validClasses[i]] <- 1
    }
    print("=========================================")
    print("-- Validating . . .")
    print("=========================================")
    for (i in 1:inputValidLengthR) {
        validGrand              <- validGrand + 1
        cHit                    <- 0
        cMiss                   <- 0
        validTarget             <- order(validTargets[i,], decreasing = TRUE)[1]
        if (verboseValidate == TRUE) {
            print("-- New Sample ---------------------------")
            print(paste("       Validation Sample", i))
        }
        for (j in 1:uniqueClasses) {
            # Forward Propagation
            validHiddenLayer        <- kernelFunction(inputValidData[i,] %*% calibratedWeights[[j]])
            validHiddenLayer        <- c(.insert(validHiddenLayer, element = 1.0)) # Insert Bias Term to Hidden Layer
            validOutput             <- kernelFunction(validHiddenLayer %*% calibratedHidden[[j]])
            classifications[i, j]   <- validOutput
            # Metric Computation     
            validError              <- 0.5 * (validTargets[i, j] - validOutput) ^ 2
            validDistance           <- abs(validTargets[i, j] - validOutput) # An Easily Interpretable Error Measure
            validSEList[i, j]       <- validError
            validDistanceList[i, j] <- validDistance # For Monitoring Error Reduction
        }
        validClass              <- order(classifications[i,], decreasing = TRUE)[1]
        if (verboseValidate == TRUE) {
            print(paste("       Known Class:           ", validTarget))
            print(paste("       Computed Class:        ", validClass))
        }
        if (validTarget == validClass) {
            validHit                <- validHit + 1
            if (verboseValidate == TRUE) print("       classification Status:  hit! :)")
        } else {
            validMiss               <- validMiss + 1
            if (verboseValidate == TRUE) print("       Classification Status:  miss :(")
        }
        if (verboseValidate == TRUE) {
            print(paste("           Valid Hits / Total: ", validHit, "/", validGrand))
            print(paste("           Valid Hit Percent:  ", .decimals((validHit/validGrand) * 100, 2)))
            print("-- Sample Done --------------------------")
        }
    }
    for (i in 1:nrow(classifications)) {
        predictedClasses        <- c(predictedClasses, order(classifications[i,], decreasing = TRUE)[1])
    }
    print("=========================================")
    print("-- Validation Complete")
    print("=========================================")
    print("-- Report")
    print("   Rounded Hits:")
    print(paste("       Valid Hits / Total:     ", validHit, "/", validGrand))
    print(paste("       Valid Hit Percent:      ", .decimals((validHit/validGrand) * 100, 2)))
    # Return Results
    list (
        inferences      = classifications,
        classes         = predictedClasses,
        uniqueClasses   = uniqueClasses,
        se              = validSEList,
        distance        = validDistanceList,
        hits            = validHit,
        total           = validGrand,
        percent         = .decimals((validHit/validGrand) * 100, 2)
    )
}

# 
# Artifical Neural Network Classification Function
# 

# PARAMS:
# 
# inputTestData         required - matrix of numerics                               validation data 
# uniqueClasses         required - integer                                          number of unique classes in training data
# calibratedWeights     required - matrix of numerics                               calibrated weights matrix from trained model
# calibratedHidden      required - matrix of numerics                               calibrated hidden layer from trained model
# kernel                required - string                                           kernel function from trained model
# verboseClassify       optional - boolean                  default = FALSE         display verbose console output 

ANN.classify            <- function (inputTestData, uniqueClasses, calibratedWeights, calibratedHidden, kernel, verboseClassify = TRUE) {
    # Configure
    kernelFunction          <- match.fun(kernel) 
    inputTestLengthR        <- nrow(inputTestData)
    classifications         <- matrix(0, nrow = nrow(inputTestData), ncol = uniqueClasses)
    classes                 <- vector()
    print("=========================================")
    print("-- Classifying . . .")
    print("=========================================")
    for (i in 1:inputTestLengthR) {
        if (verboseClassify == TRUE) {
            print("-- New Sample ---------------------------")
            print("   Forward-Propagating...")
            print(paste("       Test Sample", i))
        }
        for (j in 1:uniqueClasses) {
            # Forward Propagation
            testHiddenLayer         <- kernelFunction(inputTestData[i,] %*% calibratedWeights[[j]])
            testHiddenLayer         <- c(.insert(testHiddenLayer, element = 1.0)) # Insert Bias Term to Hidden Layer
            testOutput              <- kernelFunction(testHiddenLayer %*% calibratedHidden[[j]])
            classifications[i, j]   <- testOutput
        }
        # Metric Computation & Communication      
        if (verboseClassify == TRUE) {
            print(paste("       Computed Class:        ", order(classifications[i,], decreasing = TRUE)[1]))
            print(paste("       Computed Likelihood:   ", max(classifications[i,])))
            print("-- Sample Done --------------------------")
        }
    }
    for (i in 1:nrow(classifications)) {
        class                   <- order(classifications[i,], decreasing = TRUE)[1]
        classes                 <- c(classes, class)
    }
    print("=========================================")
    print("-- Classifications Complete")
    print("=========================================")
    # Return Results
    list (
        inferences  = classifications,
        classes     = classes
    )
}

# 
# Sample Usage
# 

# Get Time
time                    <- gsub(" ", "_", gsub(":", "-", Sys.time()))

# Import Data
print("Loading data . . .")
data.raw                <- read.csv("data/train.csv")
data.test.raw           <- read.csv("data/test.csv")
print("Done")

# Split Data into 90:10 Train & Validation Sets
data.rowCount           <- nrow(data.raw)
data.colCount           <- ncol(data.raw)
split                   <- data.rowCount * 0.9
data.train              <- data.raw[0:split,]
data.valid              <- data.raw[split:data.rowCount,]

# Extract Known Classes (0 - 9); Increment by 1 for 1-Based Indexing in R (1 - 10)
data.train.classes      <- data.train$label + 1
data.valid.classes      <- data.valid$label + 1

# Remove Known Classes From Train & Validation Sets
data.train              <- as.matrix(data.train)[,2:data.colCount]
data.valid              <- as.matrix(data.valid)[,2:data.colCount]
data.test               <- as.matrix(data.test.raw)

# Model Training
trainedModel            <- ANN.train (
    inputTrainingData       = data.train,
    trainingClasses         = data.train.classes,
    epochs                  = 5,
    kernel                  = "sigmoid",
    etaW                    = 0.03,
    etaH                    = 0.3,
    annealing               = 0.1,
    hiddenNodes             = 600,
    dataWeightsLimit        = 0.05,
    hiddenWeightsLimit      = 0.05,
    classification          = TRUE,
    verboseTrain            = FALSE
)

# Model Validation
validatedModel          <- ANN.validate (
    inputValidData          = data.valid, 
    validClasses            = data.valid.classes,
    uniqueClasses           = trainedModel$uniqueClasses,
    calibratedWeights       = trainedModel$trainedWeights, 
    calibratedHidden        = trainedModel$trainedHidden,
    kernel                  = trainedModel$kernel,
    verboseValidate         = FALSE
)

# Generate Inferences
classifications         <- ANN.classify (
    inputTestData           = data.test,
    uniqueClasses           = trainedModel$uniqueClasses,
    calibratedWeights       = trainedModel$trainedWeights,
    calibratedHidden        = trainedModel$trainedHidden,
    kernel                  = trainedModel$kernel,
    verboseClassify         = FALSE
)

# 
# Output Inferences & Known Classes
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
