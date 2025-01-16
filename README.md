# Generalizing Methylation disease predicitions
-Next TODOS:
-find suitable datasets (is feature complex for deep learning)
-enrich data with cell type
-IJCNN submission 30.1.2025


"cell type" is known for 3974 of all 16000 ish data points with 495 different cell types i.e. mean of 8 samples per cell type -> seems very bad.
"disease status" is known for 5462 of all data points with 165 different diseases. mean of 33 samples per disease.
logistic regression gets 71.18 acc on naive all disease predictions.
crossvalidated accuracy logistic regression: 0.8638888, Variance: 0.00060956