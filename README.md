#run_xgb

### Description

The run_xgb package helps to ease the xgboost training process. It automates validation steps, 
prediction initiation and accumulation, computing metrics. As the result, you can make
tests of your functions and decrease the amount of boilerplate code.

### To run

Pass train, booster and log params. Examples could be found in `data/mock/`, train and val
datasets, validation algorithm and other params

Example of run:

```
res = run_xgb(train=train,
              val=val,
              train_cols=train_cols,
              train_labels=train[label_col],
              val_labels=val[label_col],
              booster_params=booster_params,
              train_params=train_params,
              log_params=log_params,
              kfold=kfold,
              metric=mean_absolute_error)
```

### Naming

The run means to make one pass through the input data. Validation could be simple: 
kfold or time-based validation. It could be complex kfold inside time-based.

The step of validation means a change of the train-val data.
It could be one fit inside simple validation or k fits of the model inside complex validation.

The fit of the model is just the fit on the minimal available train data. For the simple 
validation the step of validation and fit of the model are the same

Inside the simple validation:

`train` - data, chosen for the validation algorithm, by a human or an outer validation algorithm
`train_train` - data, chosen from `train`, by a validation algorithm to fit the model 
`train_val` - data, chosen from `train`, by a validation algorithm to validate the model.
It is used for an early stopping.
`val` - data, chosen for the validation between `runs` by a human.

We cannot use val for the early stopping, because we will overfit on the val. And thus we will 
shift estimations.

It is a good practice to have the `stand-alone test`. Because `val` we use to estimate best 
hyperparams. 
After choosing the best model, we estimate it on the test set to get final metrics 

### Logging and saves

The logger creates a folder on the disk in the format `mm-dd-hh` and saves models, predictions,
labels and metrics inside it

