# neural-networks-learning-assurance

### Context
Python Machine Learning Project as part of the DSTI Applied MSc in Data Science and Artificial Intelligence.


### Project Objective
To “mimic” the design, development and verification phases of a safety-critical embedded machine learning software based on the preliminary guidance material issued by the European Civil Aviation Authorities (CoDANN).
For more detailed information please refer to the attached presentation.


### Software modules architecture
* Data Management
     - data_collection.ipynb
          - Objective – To collect the pictures and record associated information (date/time, lighting condition, distance/angle Jetbot/holding point…).
          - Output – .zip file containing pictures (.jpg files) and .csv file recording the metadata + hashing information for all files.
          - Platform – Jetbot
     - data_augmentation.ipynb
          - Objective –  To check the integrity of the data collected with the robot and perform data augmentation.
          - Output – 3 different Training/Validation Datasets (based on the same original data but using different data augmentation strategies) + .csv file recording the metadata .
          - Platform – Local PC
     - training_dataset_normalization_parameters.ipynb
          - Objective – To compute the mean and standard deviation of the 3 Training/Validation datasets.
          - Output – .csv file containing the mean and standard deviation for each channel (R/G/B) to be used to normalize each training dataset as well as the average for each parameter to be used with the test dataset and in service.
          - Platform –  Colab
     - discriminator.ipynb
          - Objective – To find the best layer for Early Layer OOD Discriminator and build the SVM Model.
          - Output – Best candidate layer for early layer SVM + SVM Model.
          - Platform – Colab
     - Other files
          - TRG_DATASET_NORM_PARAM.csv – Output of training_dataset_normalization_parameters.ipynb.
          - augmented_csv.csv – Sample traceability file for Training Dataset 1.

* Learning Process Management
     - learning_process_management.ipynb
          - Objective – To select the hyperparameters (e.g. batch size, learning rate…) to be used for the training of the final models.
          - Output – Training strategy (hyperparameters…).
          - Platform – Colab 

* Model Training
     - model_training.ipynb
          - Objective – To train the 3 final models in accordance with the training strategy defined previously.
          - Output – 3 trained models.
          - Platform – Colab
     - stability_analysis_cca_cka_results.ipynb
          - Objective – To verify the stability of the training algorithm.
          - Output – CKA Similarity index for each layer and each pair of models + Result similarity metric for each pair of models.
          - Platform – Colab

* Learning Process Verification
     - learning_process_verification.ipynb
          - Objective – To verify the performance of the trained networks using the test dataset.
          - Output – Performance metrics for each model.
          - Platform – Colab
     - temperature_scaling.ipynb
          - Objective – To compute the Temperature Scaling Coefficient for each model and calibrate the trained models.
          - Output – Temperature Scaling Coefficient for each model.
          - Platform – Colab
     - model_verification_DNNV.ipynb
          - Objective – To verify the properties of the trained models using the Reluplex and Eran methodologies (through the DNNV package).
          - Output – Properties verification pass/fail for each model and each methodology.
          - Platform – Colab

* Model Implementation
     - model_optimization.ipynb
          - Objective – To prune the trained models and retrain them.
          - Output – 3 pruned models and associated performance metrics.
          - Platform – Colab
     - SVM_model_training.ipynb
          - Objective – To recompute the SVM model used to detect out of distribution images.
          - Output – SVM model.
          - Platform – Jetbot
     - inference_model.ipynb
          - Objective – To implement the final system on the Jetbot platform, i.e. live camera feed image pre-processing + pruned models + discriminator (SVM model) + voting/filtering + display + guidance.
          - Output – End-to-end solution capable of processing camera feed in real-time and raise alerts if a holding point is detected.
          - Platform – Jetbot
     - Other files
          - SVM_model.sav – Locally retrained SVM model, output of SVM_model_training.ipynb

* Inference Model Verification
     - inference_model_test.ipynb
          - Objective – To test the performance of the pruned models once embedded on the Jetbot platform using the test dataset.
          - Output – Performance metrics.
          - Platform – Jetbot

### References
* Project Overview
“Artificial Intelligence Roadmap – A human-centric approach to Ai in Aviation”, EASA Report ([document](https://www.easa.europa.eu/sites/default/files/dfu/EASA-AI-Roadmap-v1.0.pdf))
“Concepts of Design Assurance for Neural Networks (CoDANN)”, EASA AI Task Force and Daedalean AG (document)
Data Management
[DM1] “Out-of-Distribution Detection in Deep Neural Networks”, Neeraj Varshney (article)
[DM2] “A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks”, Dan Hendrycks, Kevin Gimpel (article)
[DM3] “Enhancing the reliability of out-of-distribution image detection in neural networks”, Shiyu Liang, Yixuan Li, R. Srikant (article)
[DM4] “Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data”, Yen-Chang Hsu, Yilin Shen, Hongxia Jin, Zsolt Kira (article)
