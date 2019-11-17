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
        Build the dataset by (this might take a night):<br> 
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

# File descriptions
<b>Files used in the final paper:</b>
<ul>
  <li>
    <b>CPP/include/good.h</b>: <i>good descriptor header file from https://github.com/SeyedHamidreza/GOOD_descriptor</i>
  </li>
  <li>
    <b>CPP/CMakeLists.txt</b>: <i>Cmake file for building the cpp code</i>
  </li>
  <li>
    <b>CPP/good.cpp</b>: <i>good descriptor implementation from https://github.com/SeyedHamidreza/GOOD_descriptor</i>
  </li>
  <li>
    <b>CPP/main.cpp</b>: <i>the main file called by python for building feature histograms for VFH, GOOD5 and GOOD15</i>
  </li>
  <br>
  <li>
    <b>build_additional_dataset.py</b>: <i>additional script for also computing the GOOD5 and GOOD15 descriptions</i>
  </li>
  <li>
    <b>build_dataset.py</b>: <i>the main script for building the dataset, this files reads pngs and scales the to 224*224 and reads in pointclouds to compute VFH descritions.</i>
  </li>
  <li>
    <b>cross_validation_for_al_mf.py</b>: <i>The script used for calculating the final Active learning results for the paper</i>
  </li>
  <li>
    <b>cross_validation_for_non_al_mf.py</b>: <i>This script was used for calculating the final offline learning results in the paper.</i>
  </li>
  <li>
    <b>load_dataset.py</b>: <i>This script contains the functionality for loading the different datasets used in our research (VFH, GOOD5, GOOD15)
    + the implementation of the cross-validation.</i>
  </li>
  <li>
    <b>mondrian_forest_classifier_with_al_strategy.py</b>: <i>Implementation of a fit method using the described querying strategy (The AL is implemented here!)</i>
  </li>
  <li>
    <b>requirements.txt</b>: <i>The required pip packages for running the code</i>
  </li>
  <li>
    <b>run_exec.py</b>: <i>python file calling the compiled C++ code from the CPP/ folder</i>
  </li>
  <li>
    <b>utils.py</b>: <i>file with some definitions(like the category names)</i>
  </li>
</ul>
<b>Files not used in the final paper:</b>
<ul>
  <li>
    <b>create_image_features.py</b>: <i>used to compute 4096 features from the scaled images using VGG</i>
  </li>
  <li>
    <b>final_general_functions.py</b>: <i>these script were used for RGB results</i>
  </li>
  <li>
    <b>final_mf_all_image_features.py</b>: <i>these script were used for RGB results</i>
  </li>
  <li>
    <b>final_mf_vfh_and_all_image_features.py</b>: <i>these script were used for RGB results</i>
  </li>
  <li>
    <b>mrmr_feature_selection.py</b>: <i>mrmr feature selection using skfeature-chappers package</i>
  </li>
  <li>
    <b>mrmr_feature_selection_2.py</b>: <i>pymrmr feature selection</i>
  </li>
  <li>
    <b>mrmr_feature_selection_3.py</b>: <i>multithreaded mrmr feature selection using mifs package</i>
  </li>
  <li>
    <b>rf_hyperparam_search.py</b>: <i>hyperparameter search for random forest</i>
  </li>
  <li>
    <b>train_svm.py</b>: <i>file for testing SVM on VFH data</i>
  </li>
</ul>
