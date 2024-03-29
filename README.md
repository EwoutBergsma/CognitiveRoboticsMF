# Details
Final project of Cognitive Robotics by:
<ul>
  <li>Sohyung Kim (S3475743)</li>
  <li>Thijs Eker (S2576597)</li>
  <li>Dhawal Salvi (S4107624)</li>
  <li>Ewout Bergsma (S3441423)</li>
</ul>

# General idea
Implement this:
![Proposed pipeline](proposed_pipeline.jpg)

# Running instructions
<ul>
    <li>
        Compile the c++ code in the CPP/ folder
    </li>
    <li>
        Install the python packages in requirements.txt
    </li>
    <li>
        Build the dataset by (this might take a night, we also have a zip with the data, it's only 500mb):<br> 
        1. download the files from washington university site 
        <a href="https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/">evaluation set(containing images)</a> and <a href="https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/">point clouds</a> <br>
        2. Point the EVAL_DATASET_PATH, PC_DATASET_PATH, OUTPUT_DATASET_PATH variables to the downloaded folders<br>
        3. run <code>build_dataset.py</code> and after that <code>build_additional_dataset.py</code>
    </li>
    <li>
        Run either the <code>cross_validation_for_al_mf.py shape_descriptor confidence_threshold</code>(shape descriptor
        should be 0(VFH), 1(GOOD5) or 2(GOOD15)) or <code>cross_validation_for_non_al_mf.py</code> to generate the results
    </li>
</ul>

# File descriptions of files used in the final paper
<ul>
  <li>
    <b>CPP/include/good.h</b>: good descriptor header file from https://github.com/SeyedHamidreza/GOOD_descriptor
  </li>
  <li>
    <b>CPP/CMakeLists.txt</b>: Cmake file for building the cpp code
  </li>
  <li>
    <b>CPP/good.cpp</b>: good descriptor implementation from https://github.com/SeyedHamidreza/GOOD_descriptor
  </li>
  <li>
    <b>CPP/main.cpp</b>: the main file called by python for building feature histograms for VFH, GOOD5 and GOOD15
  </li>
  <br>
  <li>
    <b>build_additional_dataset.py</b>: additional script for also computing the GOOD5 and GOOD15 descriptions
  </li>
  <li>
    <b>build_dataset.py</b>: the main script for building the dataset, this files reads pngs and scales the to 224*224 and reads in pointclouds to compute VFH descritions.
  </li>
  <li>
    <b>cross_validation_for_al_mf.py</b>: The script used for calculating the final Active learning results for the paper
  </li>
  <li>
    <b>cross_validation_for_non_al_mf.py</b>: This script was used for calculating the final offline learning results in the paper.
  </li>
  <li>
    <b>load_dataset.py</b>: This script contains the functionality for loading the different datasets used in our research (VFH, GOOD5, GOOD15)
    + the implementation of the cross-validation.
  </li>
  <li>
    <b>mondrian_forest_classifier_with_al_strategy.py</b>: Implementation of a fit method using the described querying strategy (The AL is implemented here!)
  </li>
  <li>
    <b>requirements.txt</b>: The required pip packages for running the code
  </li>
  <li>
    <b>run_exec.py</b>: python file calling the compiled C++ code from the CPP/ folder
  </li>
  <li>
    <b>utils.py</b>: file with some definitions(like the category names)
  </li>
</ul>

# File descriptions of files NOT used in the final paper
<ul>
  <li>
    <b>create_image_features.py</b>: used to compute 4096 features from the scaled images using VGG
  </li>
  <li>
    <b>final_general_functions.py</b>: these script were used for RGB results
  </li>
  <li>
    <b>final_mf_all_image_features.py</b>: these script were used for RGB results
  </li>
  <li>
    <b>final_mf_vfh_and_all_image_features.py</b>: these script were used for RGB results
  </li>
  <li>
    <b>mrmr_feature_selection.py</b>: mrmr feature selection using skfeature-chappers package
  </li>
  <li>
    <b>mrmr_feature_selection_2.py</b>: pymrmr feature selection
  </li>
  <li>
    <b>mrmr_feature_selection_3.py</b>: multithreaded mrmr feature selection using mifs package
  </li>
  <li>
    <b>rf_hyperparam_search.py</b>: hyperparameter search for random forest
  </li>
  <li>
    <b>train_svm.py</b>: file for testing SVM on VFH data
  </li>
</ul>
