# Functions from Notebook 4.1

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    
    total = length(outputs)
    
    accuracy = (TP + TN) / total
    errorRate = (FP + FN) / total
    
    if (TP + FN) == 0
        sensitivity = (TN == total) ? 1.0 : 0.0
    else
        sensitivity = TP / (TP + FN)
    end
    
    if (TN + FP) == 0
        specificity = (TP == total) ? 1.0 : 0.0
    else
        specificity = TN / (TN + FP)
    end
    
    if (TP + FP) == 0
        ppv = (TN == total) ? 1.0 : 0.0
    else
        ppv = TP / (TP + FP)
    end
    
    if (TN + FN) == 0
        npv = (TP == total) ? 1.0 : 0.0
    else
        npv = TN / (TN + FN)
    end
    
    if sensitivity == 0.0 && ppv == 0.0
        fscore = 0.0
    else
        fscore = 2 * (ppv * sensitivity) / (ppv + sensitivity)
    end
    
    confMatrix = [TN FP; FN TP]
    
    return accuracy, errorRate, sensitivity, specificity, ppv, npv, fscore, confMatrix
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    boolOutputs = outputs .>= threshold
    return confusionMatrix(boolOutputs, targets)
end

function accuracy(outputs::AbstractArray{Bool}, targets::AbstractArray{Bool})
    return sum(outputs .== targets) / length(targets)
end