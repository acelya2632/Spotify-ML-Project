**ML1-Notebook7**

***Açelya Yıldırım (Group 17)***

***14.11.2025***

This submission contains:

- `AcelyaYildirim_Unit_7_Ensemble_Models.ipynb` - Main Jupyter notebook with all solutions
- `ml_utilities.jl` - Reusable utility functions (imported from previous units)

## How to Run

### Prerequisites

- Julia 1.9.3 or higher
- Jupyter Lab/Notebook
- Required Julia packages (installed in the notebook):
  - MLJ
  - MLJBase
  - MLJEnsembles
  - MLJLinearModels
  - DecisionTree
  - NaiveBayes
  - LIBSVM
  - XGBoost
  - CategoricalArrays
  - Random
  - Statistics

### Running the Notebook

1. **Place files in the same directory:**
   ```
   project/
   ├── AcelyaYildirim_Unit_7_Ensemble_Models.ipynb
   └── ml_utilities.jl
   ```

2. **Open Jupyter:**
   ```bash
   jupyter notebook AcelyaYildirim_Unit_7_Ensemble_Models.ipynb
   ```

3. **Run cells sequentially:**
   - Start from the first cell and run each cell in order
   - The first code cell loads `ml_utilities.jl`
   - All subsequent cells depend on this import

## About ml_utilities.jl

### Functions Included:

**From Unit 4 (Metrics):**
- `oneHotEncoding()` - Convert labels to one-hot encoding
- `confusionMatrix()` - Calculate confusion matrix metrics (binary & multiclass)
- `accuracy()` - Calculate accuracy

**From Unit 5 (Cross-Validation):**
- `crossvalidation()` - Generate stratified k-fold CV indices

**From Unit 6 (Preprocessing):**
- `holdOut()` - Train/test split
- `normalizeMinMax!()` - Min-max normalization
- `normalizeZeroMean!()` - Z-score normalization