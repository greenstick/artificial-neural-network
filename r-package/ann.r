#!/usr/bin/env Rscript

#
# Setup
#

print("Status: Setup")

# Get Session Information
# sessionInfo()

# Set Directory
# setwd("") 

# Set Seed
set.seed(1337)

# 
# Some Utility Functions
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
# Artifical Neural Network Training Function
# 

ANN.train               <- function (epochs = 2, inputTrainingData, trainingTargets, kernel = "sigmoid", etaP = 0.1, etaH = 0.01, hiddenNodes = 20, dataWeightsLimit = 0.05, hiddenWeightsLimit = 0.5, plotData = list(), visualizeWeights = FALSE, classification = TRUE, verboseTrain = TRUE) {
    # Configuration 
    kernelFunction          <- match.fun(kernel)  
    inputTrainingDataDim    <- dim(inputTrainingData)
    inputTrainingLengthC    <- inputTrainingDataDim[1]
    inputTrainingLengthR    <- inputTrainingDataDim[2]
    trainingWeights         <- matrix(runif(((inputTrainingLengthR) * hiddenNodes), min = - dataWeightsLimit, max = dataWeightsLimit), nrow = inputTrainingLengthR, ncol = hiddenNodes, byrow = T)
    hiddenWeights           <- matrix(.generateVector(hiddenNodes + 1, lower = - hiddenWeightsLimit, upper = hiddenWeightsLimit), nrow = hiddenNodes + 1, ncol = 1, byrow = T)
    geneExpression          <- data.frame()
    if (length(plotData) > 0) {
        if (plotData$SSE == TRUE) {
            par(mfrow = c(1, 1))
            plot(x = NULL, y = NULL, xlim = c(1, epochs), ylim = c(0, 1), cex.main = 0.8, ylab = "Classification SSE", xlab = "Epoch", main = "Scatter Plot of Sample Classification \n SSE by Epoch")
        } else if (plotData$distance == TRUE) {
            par(mfrow = c(1, 1))
            plot(x = NULL, y = NULL, xlim = c(1, epochs), ylim = c(0, 1), cex.main = 0.8, ylab = "Classification Error Distance", xlab="Epoch", main = "Scatter Plot of Sample Classification \n Error Distance by Epoch")
        }
    }
    # Epoch Loop   
    if (verboseTrain == FALSE) {
        print("=========================================")
        print("-- Training . . .    --------------------")
        print("=========================================")  
    }
    for (n in 1:epochs) {
        # Configure Epoch
        distanceList        <- vector()
        iRandomization      <- sample(inputTrainingLengthC)
        hit                 <- 0
        miss                <- 0
        grand               <- 0
        meanSSE             <- vector()
        meanDistance        <- vector()
        classifications     <- vector()
        if (verboseTrain == TRUE) {
            print("=========================================")
            print(paste("-- Epoch", n, "Start"))
            print("=========================================")
        }
        # Sample Loop         
        for (i in iRandomization) {
            grand           <- grand + 1
            total           <- i * n
            if (verboseTrain == TRUE) {
                print("-- New Sample ---------------------------")
                print("   Forward-Propagating...")
                print(paste("       Sample", i, "of Epoch", n))
            }
            # Forward Propagation
            hiddenLayer     <- kernelFunction(inputTrainingData[i,] %*% trainingWeights)
            hiddenLayer     <- c(.insert(hiddenLayer, element = 1)) # Insert Bias Term to Hidden Layer
            output          <- kernelFunction(hiddenLayer %*% hiddenWeights)
            classifications <- c(classifications, output)
            # Metric Computation & Communication   
            error           <- 0.5 * (trainingTargets[i] - output) ^ 2
            distance        <- abs(trainingTargets[i] - output) # An Easily Interpretable Error Measure
            distanceList    <- c(distanceList, distance) # For Monitoring Error Reduction
            rounded         <- round(output)
            if (verboseTrain == TRUE) {
                print(paste("       Known Class:           ", trainingTargets[i]))
                print(paste("       Computed Class:        ", round(output)))
                print(paste("       Computed Value:        ", .decimals(output, 8)))
                print(paste("       Raw Distance:          ", .decimals(distance, 8)))
                print(paste("       Computed SSE:          ", .decimals(error, 8)))
            }
            if (abs(rounded - trainingTargets[i]) == 0) {
                hit <- hit + 1
                if (verboseTrain == TRUE) print("       Rounded Hit Status:     hit! :)")
            } else {
                miss <- miss + 1
                if (verboseTrain == TRUE) print("       Rounded Hit Status:     miss :(")
            }
            if (verboseTrain == TRUE) {
                print(paste("           Epoch Hits / Total:", hit, "/", grand))
                print(paste("           Epoch Hit Percent: ", .decimals((hit/grand) * 100, 2)))
                print("   Back-Propagating...")
            }
            # Back Propagation
            deltaP          <- drop((trainingTargets[i] - output) * output * (1 - output))
            deltaH          <- (hiddenLayer * (1 - hiddenLayer) * hiddenWeights * deltaP)[-c(1)] # Compute deltaH & Remove Bias Term
            hiddenChange    <- etaH * hiddenLayer * deltaP
            updatedHidden   <- as.matrix(hiddenLayer) + as.matrix(hiddenChange)
            hiddenLayer     <- updatedHidden
            weightChange    <- etaP * t(inputTrainingData[i,, drop = FALSE]) %*% deltaH
            updatedWeights  <- trainingWeights + weightChange
            trainingWeights <- updatedWeights
            # Plot Points & Communication
            if (length(plotData) > 0) {
                if (plotData$SSE == TRUE) {
                    meanSSE     <- c(.insert(meanSSE, element = error)) 
                    points(x = n, y = error , col = 'orange')
                } else if (plotData$distance == TRUE) {
                    meanDistance<- c(.insert(meanDistance, element = distance))
                    points(x = n, y = distance, col = 'blue')   
                }
            }
            if (verboseTrain == TRUE) print("-- Sample Done --------------------------")
        }
        if (verboseTrain == TRUE) {
            print("=========================================")
            print(paste("-- Epoch", n, "Done"))
            print("=========================================")
        }
        if (length(plotData) > 0) {
            if (plotData$SSE == TRUE) {
                points(x = n, y = mean(meanSSE) , col = 'green')
            } else if (plotData$distance) {
                points(x = n, y = mean(meanDistance), col = 'red')   
            }
        }
    }
    if (verboseTrain == FALSE) {
        print("=========================================")
        print("-- Training Complete     ----------------")
        print("=========================================") 
        print("-- Report -------------------------------")
        print("   Rounded Hits Last Epoch:")
        print(paste("       Train Hits / Total:     ", hit, "/", grand))
        print(paste("       Train Hit Percent:      ", .decimals((hit/grand) * 100, 2))) 
    }
    if (length(plotData) > 0) {
        if (plotData$SSE == TRUE) {
            legend(1, 1, legend = c('Mean SSE', 'SSE per Sample'), pch = 1, col = c("green", "orange"), cex = 0.6)   
        } else if (plotData$distance == TRUE) {
            legend(1, 1, legend = c('Mean Error Distance', 'Error Distance per Sample'), pch = 1, col = c("red", "blue"), cex = 0.6)   
        } else if (plotData$weightMeans$plot == TRUE) {
            geneExpression <- as.data.frame(t(apply(trainingWeights, 1, mean)))[-c(1)]
            names(geneExpression) <- geneNames
            image(t(as.matrix(geneExpression)), axis = FALSE, main = "Heat Map of Mean Weights Computed for \n Gene Expression Levels", axes = FALSE)
        }
    }
    # Return Results
    list (
        classifications     = classifications,
        trainedWeights      = trainingWeights, 
        trainedHidden       = hiddenWeights, 
        kernel              = kernel,
        meanGeneExpression  = geneExpression
    )
}

# 
# Artifical Neural Network Validation Function
# 

ANN.validate        <- function (inputValidData, validTargets = vector(), calibratedWeights = computedWeights$trainedWeights, calibratedHidden = computedWeights$trainedHidden, kernel = kernel, verboseValidate = TRUE) {
    # Configure
    kernelFunction      <- match.fun(kernel) 
    inputValidDataDim   <- dim(inputValidData)
    inputValidLengthC   <- inputValidDataDim[1]
    inputValidLengthR   <- inputValidDataDim[2]
    validDistanceList   <- vector()
    classes             <- vector()
    validHit            <- 0
    validMiss           <- 0
    validGrand          <- 0
    print("=========================================")
    print("-- Validating Model ---------------------")
    print("=========================================")
    for (i in 1:inputValidLengthC) {
        validGrand           <- validGrand + 1
        if (verboseValidate == TRUE) {
            print("-- New Sample ---------------------------")
            print("   Forward-Propagating...")
            print(paste("       Valid Sample", i))
        }
        # Forward Propagation
        inputValidSample    <- inputValidData[i,]
        validHiddenLayer    <- kernelFunction(inputValidSample %*% calibratedWeights)
        validHiddenLayer    <- c(.insert(validHiddenLayer, element = 1.0)) # Insert Bias Term to Hidden Layer
        validOutput         <- kernelFunction(validHiddenLayer %*% calibratedHidden)
        classes             <- c(classes, validOutput)
        # Metric Computation & Communication      
        if (verboseValidate == TRUE) {
            print(paste("       Computed Value:        ", .decimals(validOutput, 8)))
            print(paste("       Computed Class:        ", round(validOutput)))
        }
        if (length(validTargets > 0)) {
            validRounded        <- round(validOutput)
            validError          <- 0.5 * (validTargets[i] - validOutput) ^ 2
            validDistance       <- abs(validTargets[i] - validOutput) # An Easily Interpretable Error Measure
            validDistanceList   <- c(validDistanceList, validDistance) # For Monitoring Error Reduction  
            if (verboseValidate == TRUE) {
                print(paste("       Known Class:           ", validTargets[i]))
                print(paste("       Raw Distance:          ", .decimals(validDistance, 8)))
                print(paste("       Computed SSE:          ", .decimals(validError, 8)))
            }
            if (abs(validRounded - validTargets[i]) == 0) {
                validHit    <- validHit + 1
                if (verboseValidate == TRUE) print("       Rounded Hit Status:     Hit! :)")
            } else {
                validMiss   <- validMiss + 1
                if (verboseValidate == TRUE) print("       Rounded Hit Status:     Miss :(")
            }
            if (verboseValidate == TRUE) {
                print(paste("           Valid Hits / Total: ", validHit, "/", validGrand))
                print(paste("           Valid Hit Percent:  ", .decimals((validHit/validGrand) * 100, 2)))
                print("-- Sample Done --------------------------")
            }
        }
    }
    print("=========================================")
    print("-- Validation Complete ------------------")
    print("=========================================")
    print("-- Report -------------------------------")
    print("   Rounded Hits:")
    print(paste("       Valid Hits / Total:     ", validHit, "/", validGrand))
    print(paste("       Valid Hit Percent:      ", .decimals((validHit/validGrand) * 100, 2)))
    # Return Results
    list (
        inferences  = classes,
        classes     = round(classes),
        hits        = validHit,
        total       = validGrand,
        percent     = .decimals((validHit/validGrand) * 100, 2)
    )
}

ANN.classify        <- function (inputTestData, calibratedWeights = computedWeights$trainedWeights, calibratedHidden = computedWeights$trainedHidden, kernel = kernel, verboseClassify = TRUE) {
    # Configure
    kernelFunction      <- match.fun(kernel) 
    inputTestDataDim    <- dim(inputTestData)
    inputTestLengthC    <- inputTestDataDim[1]
    inputTestLengthR    <- inputTestDataDim[2]
    testDistanceList    <- vector()
    classes             <- vector()
    print("=========================================")
    print("-- Classifying Data ---------------------")
    print("=========================================")
    for (i in 1:inputTestLengthC) {
        if (verboseClassify == TRUE) {
            print("-- New Sample ---------------------------")
            print("   Forward-Propagating...")
            print(paste("       Test Sample", i))
        }
        # Forward Propagation
        inputTestSample    <- inputTestData[i,]
        testHiddenLayer    <- kernelFunction(inputTestSample %*% calibratedWeights)
        testHiddenLayer    <- c(.insert(testHiddenLayer, element = 1.0)) # Insert Bias Term to Hidden Layer
        testOutput         <- kernelFunction(testHiddenLayer %*% calibratedHidden)
        classes             <- c(classes, testOutput)
        # Metric Computation & Communication      
        if (verboseClassify == TRUE) {
            print(paste("       Computed Value:        ", .decimals(testOutput, 8)))
            print(paste("       Computed Class:        ", round(testOutput)))
            print("-- Sample Done --------------------------")
        }
    }
    print("=========================================")
    print("-- Classifications Complete -------------")
    print("=========================================")
    # Return Results
    list (
        inferences  = classes,
        classes     = round(classes)
    )
}