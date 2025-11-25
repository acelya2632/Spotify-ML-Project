# ML UTILITIES - Reusable Functions

# Compiled from Notebook 3, 4, 5, and 6
# Author: Açelya Yıldırım

using Random
using Statistics

println("Loading ML Utilities...")

# ONE-HOT ENCODING

"""
    oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})

Convert categorical labels to one-hot encoded matrix.

# Arguments
- `feature`: Vector of labels to encode
- `classes`: Vector of all possible classes

# Returns
- Boolean matrix of size (length(feature), length(classes))
"""
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    numPatterns = length(feature)
    encoded = falses(numPatterns, numClasses)
    
    for i in 1:numPatterns
        classIdx = findfirst(isequal(feature[i]), classes)
        if classIdx !== nothing
            encoded[i, classIdx] = true
        end
    end
    
    return encoded
end

"""
    oneHotEncoding(feature::AbstractArray{<:Any,1})

Convert categorical labels to one-hot encoded matrix (auto-detect classes).
"""
function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature)
    return oneHotEncoding(feature, classes)
end

"""
    oneHotEncoding(feature::AbstractArray{Bool,1})

Convert boolean vector to one-hot encoded matrix.
"""
function oneHotEncoding(feature::AbstractArray{Bool,1})
    return hcat(.!feature, feature)
end

# CONFUSION MATRIX

"""
    confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

Calculate confusion matrix metrics for binary classification.

# Returns
Tuple of: (accuracy, errorRate, sensitivity, specificity, ppv, npv, fscore, confMatrix)
"""
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    
    total = length(outputs)
    
    accuracy = (TP + TN) / total
    errorRate = (FP + FN) / total
    
    # Handle edge cases
    sensitivity = (TP + FN) == 0 ? ((TN == total) ? 1.0 : 0.0) : TP / (TP + FN)
    specificity = (TN + FP) == 0 ? ((TP == total) ? 1.0 : 0.0) : TN / (TN + FP)
    ppv = (TP + FP) == 0 ? ((TN == total) ? 1.0 : 0.0) : TP / (TP + FP)
    npv = (TN + FN) == 0 ? ((TP == total) ? 1.0 : 0.0) : TN / (TN + FN)
    
    fscore = (sensitivity == 0.0 && ppv == 0.0) ? 0.0 : 
             2 * (ppv * sensitivity) / (ppv + sensitivity)
    
    confMatrix = [TN FP; FN TP]
    
    return accuracy, errorRate, sensitivity, specificity, ppv, npv, fscore, confMatrix
end

"""
    confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

Binary classification with threshold for real-valued outputs.
"""
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    boolOutputs = outputs .>= threshold
    return confusionMatrix(boolOutputs, targets)
end

"""
    confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)

Multiclass confusion matrix with one-hot encoded inputs.
"""
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs, 2) == size(targets, 2) "outputs and targets must have same number of columns"
    
    numClasses = size(targets, 2)
    
    # Special case: single column = binary
    if numClasses == 1
        return confusionMatrix(outputs[:,1], targets[:,1])
    end
    
    @assert numClasses != 2 "For binary classification (2 classes), use vector version"
    
    # Calculate per-class metrics
    sensitivities = zeros(numClasses)
    specificities = zeros(numClasses)
    ppvs = zeros(numClasses)
    npvs = zeros(numClasses)
    fscores = zeros(numClasses)
    
    # Confusion matrix
    confMatrix = zeros(Int, numClasses, numClasses)
    
    for i in 1:numClasses
        for j in 1:numClasses
            confMatrix[i,j] = sum(targets[:,i] .& outputs[:,j])
        end
    end
    
    # Calculate metrics for each class
    for i in 1:numClasses
        TP = confMatrix[i,i]
        FN = sum(confMatrix[i,:]) - TP
        FP = sum(confMatrix[:,i]) - TP
        TN = sum(confMatrix) - TP - FN - FP
        
        sensitivities[i] = (TP + FN == 0) ? 1.0 : TP / (TP + FN)
        specificities[i] = (TN + FP == 0) ? 1.0 : TN / (TN + FP)
        ppvs[i] = (TP + FP == 0) ? 1.0 : TP / (TP + FP)
        npvs[i] = (TN + FN == 0) ? 1.0 : TN / (TN + FN)
        fscores[i] = (sensitivities[i] == 0 && ppvs[i] == 0) ? 0.0 :
                     2 * (ppvs[i] * sensitivities[i]) / (ppvs[i] + sensitivities[i])
    end
    
    # Aggregate metrics
    if weighted
        # Weighted by class frequency
        weights = sum(targets, dims=1)[1,:] / size(targets, 1)
        sensitivity = sum(sensitivities .* weights)
        specificity = sum(specificities .* weights)
        ppv = sum(ppvs .* weights)
        npv = sum(npvs .* weights)
        fscore = sum(fscores .* weights)
    else
        # Macro average
        sensitivity = mean(sensitivities)
        specificity = mean(specificities)
        ppv = mean(ppvs)
        npv = mean(npvs)
        fscore = mean(fscores)
    end
    
    accuracy = sum(diag(confMatrix)) / sum(confMatrix)
    errorRate = 1 - accuracy
    
    return accuracy, errorRate, sensitivity, specificity, ppv, npv, fscore, confMatrix
end

"""
    confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)

Multiclass with real-valued outputs.
"""
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; 
                        threshold::Real=0.5, weighted::Bool=true)
    boolOutputs = outputs .>= threshold
    return confusionMatrix(boolOutputs, targets, weighted=weighted)
end

"""
    confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, 
                   classes::AbstractArray{<:Any,1}; weighted::Bool=true)

Multiclass with label vectors.
"""
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, 
                        classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Convert to one-hot
    outputs_encoded = oneHotEncoding(outputs, classes)
    targets_encoded = oneHotEncoding(targets, classes)
    
    return confusionMatrix(outputs_encoded, targets_encoded, weighted=weighted)
end

"""
    confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)

Multiclass with auto-detected classes.
"""
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; 
                        weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs, targets, classes, weighted=weighted)
end

# ACCURACY 

"""
    accuracy(outputs::AbstractArray{Bool}, targets::AbstractArray{Bool})

Calculate accuracy for boolean predictions.
"""
function accuracy(outputs::AbstractArray{Bool}, targets::AbstractArray{Bool})
    return sum(outputs .== targets) / length(targets)
end

# CROSS-VALIDATION

"""
    crossvalidation(N::Int64, k::Int64)

Generate k-fold cross-validation indices for N samples.
"""
function crossvalidation(N::Int64, k::Int64)
    baseVector = 1:k
    repeatedVector = repeat(baseVector, ceil(Int, N/k))
    indices = repeatedVector[1:N]
    return shuffle!(indices)
end

"""
    crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)

Stratified k-fold CV for binary classification.
"""
function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = zeros(Int64, length(targets))
    indices[targets] = crossvalidation(sum(targets), k)
    indices[.!targets] = crossvalidation(sum(.!targets), k)
    return indices
end

"""
    crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)

Stratified k-fold CV for multiclass (one-hot encoded).
"""
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N = size(targets, 1)
    indices = zeros(Int64, N)
    
    for class in 1:size(targets, 2)
        indices[targets[:, class]] = crossvalidation(sum(targets[:, class]), k)
    end
    
    return indices
end

"""
    crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)

Stratified k-fold CV for multiclass (label vector).
"""
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    oneHotTargets = oneHotEncoding(targets)
    return crossvalidation(oneHotTargets, k)
end

# HOLDOUT 

"""
    holdOut(N::Int, P::Real)

Split N samples into training and test sets.
P is the proportion of test samples (0 to 1).
"""
function holdOut(N::Int, P::Real)
    @assert (P >= 0.0) && (P <= 1.0) "P must be between 0 and 1"
    
    numTest = Int(round(N * P))
    numTrain = N - numTest
    
    indices = randperm(N)
    
    return indices[1:numTrain], indices[numTrain+1:end]
end

"""
    holdOut(N::Int, Pval::Real, Ptest::Real)

Split N samples into training, validation, and test sets.
"""
function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert (Pval >= 0.0) && (Pval <= 1.0) "Pval must be between 0 and 1"
    @assert (Ptest >= 0.0) && (Ptest <= 1.0) "Ptest must be between 0 and 1"
    @assert (Pval + Ptest <= 1.0) "Pval + Ptest must be <= 1"
    
    numTest = Int(round(N * Ptest))
    numVal = Int(round(N * Pval))
    numTrain = N - numTest - numVal
    
    indices = randperm(N)
    
    return (indices[1:numTrain], 
            indices[numTrain+1:numTrain+numVal], 
            indices[numTrain+numVal+1:end])
end

# NORMALIZATION

"""
    normalizeMinMax!(inputs::AbstractArray{<:Real,2})

Normalize input features to [0, 1] range using min-max scaling.
Modifies the input array in-place.
"""
function normalizeMinMax!(inputs::AbstractArray{<:Real,2})
    for col in 1:size(inputs, 2)
        minVal = minimum(inputs[:, col])
        maxVal = maximum(inputs[:, col])
        if maxVal > minVal
            inputs[:, col] = (inputs[:, col] .- minVal) ./ (maxVal - minVal)
        end
    end
    return inputs
end

"""
    normalizeMinMax!(trainInputs, testInputs)

Normalize test inputs using training data statistics.
"""
function normalizeMinMax!(trainInputs::AbstractArray{<:Real,2}, 
                         testInputs::AbstractArray{<:Real,2})
    for col in 1:size(trainInputs, 2)
        minVal = minimum(trainInputs[:, col])
        maxVal = maximum(trainInputs[:, col])
        if maxVal > minVal
            trainInputs[:, col] = (trainInputs[:, col] .- minVal) ./ (maxVal - minVal)
            testInputs[:, col] = (testInputs[:, col] .- minVal) ./ (maxVal - minVal)
        end
    end
    return trainInputs, testInputs
end

# NORMALIZATION

"""
    normalizeZeroMean!(inputs::AbstractArray{<:Real,2})

Normalize to zero mean and unit variance.
"""
function normalizeZeroMean!(inputs::AbstractArray{<:Real,2})
    for col in 1:size(inputs, 2)
        μ = mean(inputs[:, col])
        σ = std(inputs[:, col])
        if σ > 0
            inputs[:, col] = (inputs[:, col] .- μ) ./ σ
        end
    end
    return inputs
end

"""
    normalizeZeroMean!(trainInputs, testInputs)

Normalize test using training statistics.
"""
function normalizeZeroMean!(trainInputs::AbstractArray{<:Real,2}, 
                           testInputs::AbstractArray{<:Real,2})
    for col in 1:size(trainInputs, 2)
        μ = mean(trainInputs[:, col])
        σ = std(trainInputs[:, col])
        if σ > 0
            trainInputs[:, col] = (trainInputs[:, col] .- μ) ./ σ
            testInputs[:, col] = (testInputs[:, col] .- μ) ./ σ
        end
    end
    return trainInputs, testInputs
end

# ============================================
println("ML Utilities loaded")
println("   Available functions:")
println("   ├─ oneHotEncoding")
println("   ├─ confusionMatrix (multiple versions)")
println("   ├─ accuracy")
println("   ├─ crossvalidation (multiple versions)")
println("   ├─ holdOut")
println("   ├─ normalizeMinMax!")
println("   └─ normalizeZeroMean!")
println()
