# Auto-Optimization_System_2020
This repository features a system designed to simplify and automate hyperparameter tuning for machine learning models. Users can plan and select hyperparameters for time-series model training, avoiding the need to repeatedly enter commands when making changes.
This system is ideal for researchers and engineers seeking an efficient solution for model training, evaluation, and explainability.

**System Features:**
* **Streamlined Training Process:** Easily customize and train models without repetitive manual inputs, improving efficiency.
* **Automation and Flexibility:** Automatically adjusts training based on selected hyperparameters and outputs comprehensive results, including explainability analysis, in both plots and data files.

## Inputs:
The system processes datasets stored in the ```/Data``` directory. 
Users should place their datasets in the appropriate format within this directory for seamless integration into the hyperparameter tuning and model training pipeline.


## Outputs:
1. **Training Results** including learning curves and saved models (stored in ```/Output_files```)
   <img src="https://github.com/Poopogen/Model_Hyperparameter_Optimization_System_2020/blob/653167f9945c207d548a7510f1792f957e660142/Output_files/Plot/Loss_plot/mse/loss_per_epoch_withparameterinfo_mse.png" alt="Alt Text" style="width:40%; height:auto;">

   
2. **Prediction Results** including result plots and excel files (stored in ```/Prediction/output```)

 
3. **AI Explainability** (SHAP Value Plots stored in ```/Prediction/shap```):
   * Shap Values at Different Timesteps

     <img src="https://github.com/Poopogen/Model_Hyperparameter_Optimization_System_2020/blob/653167f9945c207d548a7510f1792f957e660142/Prediction/shap/timestep.png" alt="Alt Text" style="width:50%; height:auto;">
     
   * Total Shap Values

     <img src="https://github.com/Poopogen/Model_Hyperparameter_Optimization_System_2020/blob/4c09b0edada3020a15129535bd0cbcca68a7f79c/Prediction/shap/summary_plot2.png" alt="Alt Text" style="width:50%; height:auto;">
     
   * Sample-level SHAP Values

     <img src="https://github.com/Poopogen/Model_Hyperparameter_Optimization_System_2020/blob/653167f9945c207d548a7510f1792f957e660142/Prediction/shap/localplot_sample1.png" alt="Alt Text" style="width:50%; height:auto;">


