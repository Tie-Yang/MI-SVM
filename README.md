# MI-SVM
Overcoming the Accuracy Paradox in Small-Sample Bioinformatics: A Computationally Efficient MI-SVM Pipeline for Imbalanced Protein Localization

# MI-SVM: Lightweight Mutual Information-SVM for Imbalanced Protein Localization

Implementation of the paper **Overcoming the Accuracy Paradox in Imbalanced Protein Localization via a Lightweight MI-SVM** (Submitted to MLNN 2026).  
A lightweight two-stage pipeline combining **Mutual Information-based feature selection** and **Cost-Sensitive SVM** to solve the **accuracy paradox** in imbalanced protein subcellular localization prediction (E.coli dataset).

## 📚 Abstract
Accurate prediction of protein subcellular localization is critical for bioinformatics and drug discovery, but benchmark datasets (e.g., E.coli) suffer from **small sample size** and **severe class imbalance**, leading to the accuracy paradox (high overall accuracy but poor minority class prediction). Deep learning models require heavy computation and easily overfit on small tabular data, while traditional classifiers favor majority classes.  
We propose a CPU-friendly MI-SVM pipeline: (1) MI-based feature selection filters non-informative features to reduce noise; (2) Cost-Sensitive SVM with RBF kernel penalizes minority class misclassifications. Experimental results show MI-SVM maintains 88.12% global accuracy (same as KNN) and improves Macro Recall from 66.74% to 69.58%, effectively rescuing biologically critical minority instances without computational overhead.

## 🔧 Environment Setup
The project is implemented with **pure CPU** (no GPU/accelerator required), compatible with Windows/Linux/macOS.
1. Python version: 3.8, 3.9, 3.10 (recommended)
2. Install dependent packages:
   
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

## 🚀 Quick Start (One-click Reproduce)
All operations are CPU-based, and the experiment can be completed in 10 seconds on a consumer-grade computer.

### Step 1: Download E.coli Dataset

Download the raw dataset from UCI Machine Learning Repository - E.coli, get the file ecoli.data, and place it in the root directory of this project (same level as MI-SVM.py).

### Step 2: Run Main Experiment Code

Execute the main code to train MI-SVM and baseline models (KNN/Random Forest/HistGBM), and output experimental results in paper format:

  ```bash
  python MI-SVM.py
  ```
Output includes: dataset loading info, grid search time, comparative results table, optimal pipeline parameters.

### Step 3: Plot Confusion Matrix Comparison

Generate high-resolution (300dpi) confusion matrix comparison figure (KNN vs MI-SVM) for paper illustration:

  ```bash
  python plot_cm.py
  ```
## 📊 Experimental Results
All results are obtained on CPU-only environment with 7:3 stratified train-test split (random_state=42, ensure reproducibility).The key improvement is Macro Recall (unweighted, reflecting minority class performance), while maintaining the same global accuracy as KNN.


Pipeline Comparative Experimental Results (CPU Environment)

| Method / Model            | Accuracy   | Macro Recall  | Weighted Rec | Macro F1  |
|---------------------------|------------|---------------|--------------|-----------|
| KNN (k=5)                 | 0.8812     | 0.6674        | 0.8812       | 0.6275    |
| Random Forest             | 0.8515     | 0.6369        | 0.8515       | 0.5971    |
| LightGBM (HistGBM)        | 0.8218     | 0.5153        | 0.8218       | 0.5093    |
| Proposed MI-SVM Pipeline  | 0.8812     | 0.6958        | 0.8812       | 0.6399    |

Optimal Pipeline Parameters: Features Retained (k)=all, SVM C=1, Gamma=0.2

## 📝 Citation
If you use this code/dataset/result in your research, please cite our paper:

```
Zihan Wang, Kevin Yang, and Tiebao Yang. 2026. Overcoming the Accuracy Paradox in Imbalanced Protein Localization via a Lightweight MI-SVM. In Proceedings of The 3rd International Conference on Machine Learning and Neural networks (MLNN ’26). ACM, New York, NY, USA, 6 pages. https://doi.org/XXXXXXX.XXXXXXX
```

BibTeX Format (for LaTeX):
```bibtex
@inproceedings{wang2026misvm,
  author    = {Wang, Zihan and Yang, Kevin and Yang, Tiebao},
  title     = {Overcoming the Accuracy Paradox in Imbalanced Protein Localization via a Lightweight MI-SVM},
  booktitle = {Proceedings of The 3rd International Conference on Machine Learning and Neural networks (MLNN '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  pages     = {1--6},
  doi       = {XXXXXXX.XXXXXXX}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details. 

Free for academic research/non-commercial use.
