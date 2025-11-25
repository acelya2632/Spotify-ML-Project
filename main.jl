# Main.jl - Spotify Hit Prediction
# Master control file

using Random
Random.seed!(42)

println("Starting Spotify ML Project...")

# Include all modules
include("utils/data_loading.jl")
include("utils/preprocessing.jl")

# Run approaches
include("src/approach1_binary_classification.jl")
include("src/approach2_multiclass_classification.jl")
include("src/approach3_regression.jl")
include("src/approach4_temporal_engineering.jl")

println("All approaches completed")