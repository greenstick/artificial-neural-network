# 
# Sample Usage
# 

library("ANN")

# Import Data
data.train.raw          <- read.csv("data/training.csv")
data.train.classes      <- read.csv("data/training-classes.csv")
data.valid.raw          <- read.csv("data/validating.csv")
data.valid.classes      <- read.csv("data/validating-classes.csv")

# Coerce Data Frames to Matrix
data.train              <- as.matrix(data.train.raw)
data.valid              <- as.matrix(data.valid.raw)

# Pivot Known Sample Classes for Input
data.train.classes      <- t(data.train.classes)
data.valid.classes      <- t(data.valid.classes)

# Model Training
trainedModel            <- ANN.train (
    epochs                  = 20,
    inputTrainingData       = data.train,
    trainingTargets         = data.train.classes,
    kernel                  = "sigmoid",
    etaP                    = 0.02, 
    etaH                    = 0.02, 
    hiddenNodes             = 30, 
    dataWeightsLimit        = 0.5, 
    hiddenWeightsLimit      = 0.5, 
    plotData                = list(
        SSE                     = FALSE,
        distance                = FALSE,
        weightMeans             = list(
            plot                    = FALSE,
            lables                  = ""
        )
    ),
    classification          = TRUE,
    verboseTrain            = TRUE
)

# 
# Sample Model Validation
# 

# Model Validation
validatedModel          <- ANN.validate (
    inputValidData          = data.valid, 
    validTargets            = data.valid.classes,
    calibratedWeights       = trainedModel$trainedWeights, 
    calibratedHidden        = trainedModel$trainedHidden,
    kernel                  = trainedModel$kernel,
    verboseValidate         = TRUE
)

# 
# Sample Classification
# 

# Test Data (Combine Train & Validation Data. Note this is bad, and only done for illustrative purposes.)
data.classify           <- rbind(data.train, data.valid)
data.classify.classes   <- as.matrix(c(data.train.classes, data.valid.classes))

# Randomize Order
randomOrder             <- as.matrix(sample(nrow(data.classify)))

# Re-order Data & Known Classes
data.classify           <- data.classify[randomOrder,]
data.classify.classes   <- data.classify.classes[randomOrder,]

# Generate Inferences
classifications         <- ANN.classify (
    inputTestData           = data.classify,
    calibratedWeights       = trainedModel$trainedWeights,
    calibratedHidden        = trainedModel$trainedHidden,
    kernel                  = trainedModel$kernel,
    verboseClassify         = FALSE
)

# 
# Output Infernences & Known Classes
# 

print(classifications)
print("Known Classes")
print(data.classify.classes)