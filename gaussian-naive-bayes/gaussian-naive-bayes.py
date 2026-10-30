import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    # Write code here
    priors = {}
    cls_count = {}
    means = {}
    vars = {}

    eps = 10e-9
    # calculate mean
    n_sample_trains, n_features = len(X_train), len(X_train[0])
    for i in range(n_sample_trains):
        x_row, label = X_train[i], y_train[i]
        
        if label not in cls_count:
            cls_count[label] = 1
            means[label] = x_row
        else:
            cls_count[label] += 1 
            unnorm_mean = means[label]
            for j in range(n_features):
                unnorm_mean[j] += x_row[j]

    # calculate mean and priors
    for label, count in cls_count.items():
        for j in range(n_features):
            means[label][j] /= count
            priors[label] = cls_count[label] / n_sample_trains

    # calculate sample variance
    for i in range(n_sample_trains):
        x_row, label = X_train[i], y_train[i]

        square_list = []
        for j in range(n_features):
            square = (x_row[j] - means[label][j]) ** 2 / cls_count[label]
            square_list.append(square)

        if label not in vars:
            vars[label] = square_list
        else:
            for j in range(n_features):
                vars[label][j] += square_list[j]


    # add eps to all variance to handle 0 variance feature 
    for label, _ in vars.items():
        for j in range(n_features):
            vars[label][j] += eps

    n_sample_test = len(X_test)

    preds = []
    label_list = sorted(list(cls_count.keys()))
    # calculate log-posterior for each class 
    for i in range(n_sample_test):
        x_row = X_test[i]
        log_post_row, max_val, pred_label = [], None, None
        for label in label_list:  
            cls_mean = means[label]
            logprior = math.log(priors[label])
            loglikelihood = 0
            for j in range(n_features):
                term1 = -0.5 * math.log(2 * math.pi * vars[label][j])
                term2 = - (x_row[j] - cls_mean[j]) ** 2 / (2 * vars[label][j])
                loglikelihood +=  (term1 + term2)

            logpost = logprior + loglikelihood
            if max_val is None or logpost > max_val:
                max_val = logpost
                pred_label = label
        preds.append(pred_label)
    return preds
    
                
                
            
            