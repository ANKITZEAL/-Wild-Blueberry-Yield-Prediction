import joblib
import matplotlib as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
import shap

curr_path = os.path.dirname(os.path.realpath(__file__))


feat_cols = ['clonesize', 'honeybee', 'osmia', 'MinOfUpperTRange', 
       'MaxOfLowerTRange', 'RainingDays', 'AverageRainingDays', 'fruitset',
       'fruitmass', 'seeds']
X_test_xgb_df = pd.read_csv(curr_path + "/assets/X_test_xgb_df.csv", index_col= "id")
xgb_final = joblib.load(curr_path + "/assets/joblib_files/xgboost_blueberry_final_model.joblib")

def predict_yield(attributes: np.ndarray):
    """ Returns Blueberry Yield value"""
    # print(attributes.shape) # (1,10)

    shap__xgb_explainer = shap.TreeExplainer(xgb_final)
    shap_xgb_values = shap__xgb_explainer.shap_values(attributes)
    shap_xgb_expected_values = shap__xgb_explainer.expected_value

    # plt.figure(figsize=(9,13))
    shap.force_plot(shap_xgb_expected_values, 
                    shap_xgb_values, 
                    attributes, 
                    feat_cols, 
                    show=False, 
                    matplotlib=True).savefig(curr_path + "/assets/force_plot_custom.png",
                                             bbox_inches = 'tight')

    image = Image.open(curr_path + '/assets/force_plot_custom.png')


    pred = xgb_final.predict(attributes)
    print("Yield predicted")

    return pred[0], image