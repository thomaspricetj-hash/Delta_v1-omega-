"""
ML Extensions for TRANSLATION_OS_V1
Production-grade classical ML utilities integrated with WeaveFormer pipeline.

Modules:
- data_sanitizer: Z-score + Isolation Forest outlier detection
- feature_analysis: PCA, ICA for motif/frame analysis
- clustering: KMeans, DBSCAN for motif families
- pattern_mining: Apriori frequent pattern mining for HSCL
- meta_predictors: LogisticRegression, RandomForest, SVM, DecisionTree for meta-control
- regression_utils: Linear, Multivariate, Lasso regression
- augmentation_policy: Q-Learning policy optimizer for augmentation scheduling
- self_training: Self-training / Co-training pseudo-labeling
"""
__all__ = [
    "DataSanitizer",
    "FeatureAnalyzer",
    "MotifClusterer",
    "PatternMiner",
    "MetaPredictor",
    "RegressionUtils",
    "AugmentationPolicy",
    "SelfTrainer",
]
