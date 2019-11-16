# Details
Final project of Cognitive Robotics by:
<ul>
  <li>Sohyung Kim (S3475743)</li>
  <li>Thijs Eker (S2576597)</li>
  <li>Dhawal Salvi (S4107624)</liv>
  <li>Ewout Bergsma (S3441423)</li>
</ul>

# File descriptions
CPP/
build_additional_dataset.py	
build_dataset.py	
create_image_features.py -- NOT USED IN PAPER
cross_validation_example.py --REMOVE
cross_validation_for_al_mf.py
cross_validation_for_al_mf_intermediate_updates.py	
cross_validation_for_non_al_mf.py	
final_general_functions.py	
final_mf_all_image_features.py	-- NOT USED IN PAPER
final_mf_vfh_and_all_image_features.py	-- NOT USED IN PAPER
final_mf_vfh_features.py	-- REMOVE
jobscript_all_image_features	-- NOT USED IN PAPER
jobscript_test_al	-- REMOVE
jobscript_vfh_and_all_image_features	-- NOT USED IN PAPER
jobscript_vfh_features	-- REMOVE
load_dataset.py
mondrian_forest_classifier_with_al_strategy.py -- Implemanatation of a fit method using the described querying strategy
mrmr_feature_selection.py -- mrmr feature selection using skfeature-chappers package NOT USED IN PAPER
mrmr_feature_selection_2.py	-- multithreaded mrmr feature selection using mifs package NOT USED IN PAPER
requirements.txt	-- UPDATE THIS
rf_hyperparam_search.py	 -- hyperparameter search for random forest NOT USED IN PAPER
run_exec.py	-- python file calling the compiled C++ code from the CPP/ folders
train_svm.py	-- file for testing SVM on VFH data NOT USED IN PAPER
utils.py	-- file with tsome definitions(like the category names)

# General idea
Implement this:
![Proposed pipeline](OurProposedPipeline.png)
