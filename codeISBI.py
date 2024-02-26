import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from pyriemann.utils.mean import mean_riemann
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace


N_RUNS=100

def mixupRiemannian(x, y):
    lam = np.random.beta(1, 1)
    batch_size = y.shape[0]
    index = np.array(torch.randperm(batch_size))

    cov_weight = [lam, 1-lam]

    meanCovs = []
    for i in range(batch_size):
        covIndexes = [i, index[i]]
        meanCov = mean_riemann(x[covIndexes], sample_weight=cov_weight)
        meanCovs.append(meanCov)

    y = lam * y + (1 - lam) * y[index]
    
    return np.array(meanCovs), np.array(y)


def create_model(n_features=40, n_signs=5, d_out=0.2):

    model = tf.keras.models.Sequential(
        [                  
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n_features, activation="relu", input_shape=(n_features, )),
            tf.keras.layers.Dropout(d_out),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(d_out),
            tf.keras.layers.Dense(n_signs, activation="softmax"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def singleTangentSpace(data, labels, groups):
    allPredictions = []
    allTestLabels = []

    for i in range(N_RUNS):
        print('#run', i+1)

        # select 1 patient randomly per class
        patientNumbers = np.array(pd.DataFrame([groups, labels]).transpose().groupby(1).agg(np.random.choice)[0])

        test_data = data[[i for i in range(len(groups)) if groups[i] in patientNumbers]]
        data_ = data[[i for i in range(len(groups)) if groups[i] not in patientNumbers]]

        test_labels = labels[[i for i in range(len(groups)) if groups[i] in patientNumbers]]
        labels_ = labels[[i for i in range(len(groups)) if groups[i] not in patientNumbers]]

        print('Data-Test:', data_.shape, test_data.shape) 


        #------#------#XdawnCovariances#------#------
        xdwn = XdawnCovariances(estimator='oas', xdawn_estimator='oas', nfilter=4).fit(data_, labels_)
        data_ = xdwn.transform(data_)
        print('After XdawnCovariances:', data_.shape)
        #------#------#XdawnCovariances#------#------    


        # augmentations
        mixedCovs, mixedLabels = mixupRiemannian(data_, labels_)

        data_ = np.concatenate((mixedCovs, data_), axis=0)
        labels_ = np.concatenate((mixedLabels, pd.get_dummies(np.append(labels_, (0,1,2,3,4)))[:-5].values), axis=0)


        #------#------#TangentSpace#------#------
        lapl = TangentSpace(metric='riemann', tsupdate=False).fit(data_, labels_)
        data_ = lapl.transform(data_)
        print('After single TangentSpace:', data_.shape)
        #------#------#TangentSpace#------#------
        

        # model
        clf = KerasClassifier(build_fn=create_model, epochs=N_EPOCHS, verbose=0)

        # training
        history = clf.fit(data_, labels_)

        
        #------#------#XdawnCovariances#------#------
        test_data = xdwn.transform(test_data)
        #------#------#XdawnCovariances#------#------
        #------#------#TangentSpace#------#------
        test_data = lapl.transform(test_data)
        #------#------#TangentSpace#------#------

        # testing
        y_pred = clf.predict_proba(test_data)


        # save results
        allPredictions.append(y_pred)
        allTestLabels.append(test_labels)
        
    return np.array(allPredictions), np.array(allTestLabels)


def multipleTangentSpace(data, labels, groups):
    allPredictions = []
    allTestLabels = []

    for i in range(N_RUNS):
        print('#run', i+1)

        # select 1 patient randomly per class
        patientNumbers = np.array(pd.DataFrame([groups, labels]).transpose().groupby(1).agg(np.random.choice)[0])

        test_data = data[[i for i in range(len(groups)) if groups[i] in patientNumbers]]
        data_ = data[[i for i in range(len(groups)) if groups[i] not in patientNumbers]]

        test_labels = labels[[i for i in range(len(groups)) if groups[i] in patientNumbers]]
        labels_ = labels[[i for i in range(len(groups)) if groups[i] not in patientNumbers]]

        print('Data-Test:', data_.shape, test_data.shape) 


        #------#------#XdawnCovariances#------#------
        xdwn = XdawnCovariances(estimator='oas', xdawn_estimator='oas', nfilter=4).fit(data_, labels_)
        data_ = xdwn.transform(data_)
        print('After XdawnCovariances:', data_.shape)
        #------#------#XdawnCovariances#------#------    


        # augmentations
        mixedCovs, mixedLabels = mixupRiemannian(data_, labels_)

        data_ = np.concatenate((mixedCovs, data_), axis=0)
        labels_ = np.concatenate((mixedLabels, pd.get_dummies(np.append(labels_, (0,1,2,3,4)))[:-5].values), axis=0)

        # threshold to select each class data (for TangentSpace mappings)
        data_threshold = 0.7
        data_mustard = data_[[i for i in range(len(labels_)) if labels_[i][0] >= data_threshold]]
        data_mustard_labels = labels_[[i for i in range(len(labels_)) if labels_[i][0] >= data_threshold]]
        data_fontan = data_[[i for i in range(len(labels_)) if labels_[i][1] >= data_threshold]]
        data_fontan_labels = labels_[[i for i in range(len(labels_)) if labels_[i][1] >= data_threshold]]
        data_tof = data_[[i for i in range(len(labels_)) if labels_[i][2] >= data_threshold]]
        data_tof_labels = labels_[[i for i in range(len(labels_)) if labels_[i][2] >= data_threshold]]
        data_pulart = data_[[i for i in range(len(labels_)) if labels_[i][3] >= data_threshold]]
        data_pulart_labels = labels_[[i for i in range(len(labels_)) if labels_[i][3] >= data_threshold]]
        data_atrspt = data_[[i for i in range(len(labels_)) if labels_[i][4] >= data_threshold]]
        data_atrspt_labels = labels_[[i for i in range(len(labels_)) if labels_[i][4] >= data_threshold]]

        #------#------#TangentSpace#------#------
        laplM = TangentSpace(metric='riemann', tsupdate=False).fit(data_mustard, data_mustard_labels)
        data_M = laplM.transform(data_)

        laplF = TangentSpace(metric='riemann', tsupdate=False).fit(data_fontan, data_fontan_labels)
        data_F = laplF.transform(data_)

        laplT = TangentSpace(metric='riemann', tsupdate=False).fit(data_tof, data_tof_labels)
        data_T = laplT.transform(data_)

        laplP = TangentSpace(metric='riemann', tsupdate=False).fit(data_pulart, data_pulart_labels)
        data_P = laplP.transform(data_)

        laplA = TangentSpace(metric='riemann', tsupdate=False).fit(data_atrspt, data_atrspt_labels)
        data_A = laplA.transform(data_)
        #------#------#TangentSpace#------#------


        data_ = np.concatenate((data_M, data_F, data_T, data_P, data_A), axis=1)
        labels_ = np.concatenate((data_mustard_labels, data_fontan_labels, data_tof_labels, data_pulart_labels, data_atrspt_labels), axis=0)
        print('After multiple TangentSpace:', data_.shape)


        # model
        clf = KerasClassifier(build_fn=create_model, epochs=N_EPOCHS, verbose=0)

        # training
        history = clf.fit(data_, labels_)


        #------#------#XdawnCovariances#------#------
        test_data = xdwn.transform(test_data)
        #------#------#XdawnCovariances#------#------
        #------#------#TangentSpace#------#------
        test_data_M = laplM.transform(test_data)
        test_data_F = laplF.transform(test_data)
        test_data_T = laplT.transform(test_data)
        test_data_P = laplP.transform(test_data)
        test_data_A = laplA.transform(test_data)
        #------#------#TangentSpace#------#------

        test_data = np.concatenate((test_data_M, test_data_F, test_data_T, test_data_P, test_data_A), axis=1)

        # testing
        y_pred = clf.predict_proba(test_data)


        # save results
        allPredictions.append(y_pred)
        allTestLabels.append(test_labels)
        
    return np.array(allPredictions), np.array(allTestLabels)