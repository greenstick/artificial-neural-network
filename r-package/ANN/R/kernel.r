# 
# Kernel Activation Functions
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