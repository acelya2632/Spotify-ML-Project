module MyLibrary

using Random
using Statistics

# Export functions to make them available when using the module
export normalizeMinMax!
export oneHotEncoding!
export split_data

"""
    normalizeMinMax!(data::Matrix{T}) where T

Normalize numerical data to the range [0, 1] using Min-Max scaling.

# Formula
    x_normalized = (x - min(x)) / (max(x) - min(x))

# Arguments
- `data::Matrix{T}`: Input matrix to be normalized (will be modified in-place)

# Returns
- Normalized matrix with values in [0, 1]

# Example
```julia
using .MyLibrary
data = Float32[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
normalized = normalizeMinMax!(data)
```

# Note
- Handles edge case where max == min (sets to 0 to avoid division by zero)
- Operates column-wise (each feature normalized independently)
"""
function normalizeMinMax!(data::Matrix{T}) where T
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)
    range_vals = max_vals .- min_vals
    
    # Avoid division by zero for constant columns
    range_vals[range_vals .== 0] .= 1.0
    
    # Normalize
    return (data .- min_vals) ./ range_vals
end


"""
    oneHotEncoding!(labels::AbstractVector)

Convert categorical labels into one-hot encoded binary matrix.

# Arguments
- `labels::AbstractVector`: Vector of categorical labels (e.g., ["Rock", "Mine", "Rock"])

# Returns
- `AbstractArray{Bool,2}`: Binary matrix where each column represents one unique class
  - Rows correspond to samples
  - Columns correspond to classes (in order of appearance)
  - Value is `true` if sample belongs to that class, `false` otherwise

# Example
```julia
using .MyLibrary
labels = ["Rock", "Mine", "Rock", "Mine", "Rock"]
encoded = oneHotEncoding!(labels)
# Result: 5×2 Bool matrix
# [true false; false true; true false; false true; true false]
```

# Note
- Classes are ordered by first appearance in the input vector
- For binary classification, creates 2 columns
"""
function oneHotEncoding!(labels::AbstractVector)
    unique_labels = unique(labels)
    n_samples = length(labels)
    n_classes = length(unique_labels)
    
    # Create one-hot matrix
    encoded = hcat([labels .== label for label in unique_labels]...)
    
    return encoded
end


"""
    split_data(input_data, output_data, train_ratio=0.9)

Split dataset into training and testing sets with random shuffling.

# Arguments
- `input_data::AbstractMatrix`: Feature matrix (n_samples × n_features)
- `output_data::AbstractMatrix`: Label matrix (n_samples × n_classes), typically one-hot encoded
- `train_ratio::Float64`: Proportion of data for training (default: 0.9 = 90%)

# Returns
- Tuple of 4 matrices: `(train_input, train_output, test_input, test_output)`

# Example
```julia
using .MyLibrary
X = rand(Float32, 100, 5)  # 100 samples, 5 features
y = oneHotEncoding!(rand(["A", "B"], 100))  # Binary labels

train_X, train_y, test_X, test_y = split_data(X, y, 0.8)
# 80 training samples, 20 test samples
```

# Notes
- Data is randomly shuffled before splitting (uses `randperm`)
- Ensures no data leakage between train and test
- Input and output must have same number of rows (samples)
- Train ratio must be between 0 and 1
"""
function split_data(
    input_data::AbstractMatrix, 
    output_data::AbstractMatrix, 
    train_ratio::Float64=0.9
)
    # Validate inputs
    n_samples = size(input_data, 1)
    @assert n_samples == size(output_data, 1) "Input and output must have same number of rows"
    @assert 0.0 < train_ratio < 1.0 "Train ratio must be between 0 and 1"
    
    # Generate random permutation of indices
    shuffled_indices = randperm(n_samples)
    
    # Calculate split point
    train_size = floor(Int, train_ratio * n_samples)
    
    # Split indices
    train_indices = shuffled_indices[1:train_size]
    test_indices = shuffled_indices[train_size+1:end]
    
    # Create splits
    train_input = input_data[train_indices, :]
    train_output = output_data[train_indices, :]
    test_input = input_data[test_indices, :]
    test_output = output_data[test_indices, :]
    
    return train_input, train_output, test_input, test_output
end

end # module MyLibrary
