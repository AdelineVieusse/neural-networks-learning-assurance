# neural-networks-learning-assurance

Context - Python Machine Learning Project as part of the DSTI Applied MSc in Data Science and Artificial Intelligence.

Project Objective - To “mimic” the design, development and verification phases of a safety-critical embedded machine learning software based on the preliminary guidance material issued by the European Civil Aviation Authorities (CoDANN).
For more detailed information please refer to the attached presentation.

Software modules architecture
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
          - Output – .csv file containing the mean and standard deviation for each channel (R/G/B) to be used to normalize each training dataset as well as the average for each           - parameter to be used with the test dataset and in service.
Platform –  Colab
![image](https://user-images.githubusercontent.com/76960664/122056487-ebad2880-cde9-11eb-8046-6d99c0205676.png)


References
Project Overview
[PO1] “Artificial Intelligence Roadmap – A human-centric approach to Ai in Aviation”, EASA Report (document)
[PO2] “Concepts of Design Assurance for Neural Networks (CoDANN)”, EASA AI Task Force and Daedalean AG (document)
Data Management
[DM1] “Out-of-Distribution Detection in Deep Neural Networks”, Neeraj Varshney (article)
[DM2] “A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks”, Dan Hendrycks, Kevin Gimpel (article)
[DM3] “Enhancing the reliability of out-of-distribution image detection in neural networks”, Shiyu Liang, Yixuan Li, R. Srikant (article)
[DM4] “Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data”, Yen-Chang Hsu, Yilin Shen, Hongxia Jin, Zsolt Kira (article)
![image](https://user-images.githubusercontent.com/76960664/122056558-fa93db00-cde9-11eb-8663-120993082fc5.png)
