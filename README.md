# AI-Stock-Recommendation
AI stock recommendation based on LGBM &amp; feature engineering 


## Notebooks Overview

1. **01 notebook** - Explains the objective (target) of the model.
2. **02 notebook** - Introduces the features.
3. **03 notebook** - Demonstrates a CNN and RNN based model (not tuned).
4. **04 notebook** - Includes various combinations for training and eventually picks the best models for tuning.
5. **05 notebook** - Tunes the best performing models from the 04_Model_Selection notebook to finalize the model choice.
6. **06 notebook** - Conducts final fine-tuning (Optuna + CV) on LightGBM and Isolation Forest. The final result suggests using LightGBM alone.

## Model Prediction

- The prediction of the final model gives the "probability" (or "score") of the stock reaching a 5% goal in the next few days. Since the features are technical indicators and stock information, one can consider this model as a "stock technical analysis expert". It is therefore limited to technical analysis as the model cannot access other market data.
- This model should be seen more as an "avoiding guide" rather than a "trading guide". When the score is below 20, it might not be worth the risk of trading such stock at that time. When the score is above 50 (rarely), it might consider to be a trade too good to miss.
- Note that stocks with high volatility tend to reach the 5% goal more easily, resulting in an overall higher score.

## Key Points

- **Objective**: Predict the likelihood of a stock reaching a 5% increase within a short period.
- **Features**: Utilize technical indicators and stock information.
- **Model Selection**: Evaluated multiple models and combinations to select the best performers.
- **Final Model**: LightGBM, optimized with Optuna and cross-validation, outperformed other models.


