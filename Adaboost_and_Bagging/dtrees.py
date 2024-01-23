import numpy as np
import matplotlib.pyplot as plt
import math

from dtrees_utils import visualization, get_dataset_fixed


class Stump():
    def __init__(self, data, labels, weights):
        '''
        Initializes a stump (one-level decision tree) which minimizes
        a weighted error function of the input dataset.

        In this function, you will need to learn a stump using the weighted
        datapoints. Each datapoint has 2 features, whose values are bounded in
        [-1.0, 1.0]. Each datapoint has a label in {+1, -1}, and its importance
        is weighted by a positive value.

        The stump will choose one of the features, and pick the best threshold
        in that dimension, so that the weighted error is minimized.

        Arguments:
            data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
            labels: An ndarray with shape (n, ). Values are +1 or -1.
            weights: An ndarray with shape (n, ). The weights of each
                datapoint, all positive.
        '''
        # You may choose to use the following variables as a start

        # The feature dimension which the stump will decide on
        # Either 0 or 1, since the datapoints are 2D
        self.dimension = 0

        # The threshold in that dimension
        # May be midpoints between datapoints or the boundaries -1.0, 1.0
        self.threshold = -1.0

        # The predicted sign when the datapoint's feature in that dimension
        # is greater than the threshold
        # Either +1 or -1
        self.sign = 1

        self.init_parameters(data,labels,weights)
 
        

    def init_parameters(self,data,labels,weights):
        min_loss_for_dimensions=[]

        for dimension in range(2):
            min_in_dim_and_threshold = self.min_loss_in_dim(data[:,dimension],labels,weights)
            min_loss_for_dimensions.append(min_in_dim_and_threshold)
        
        min_loss = min(min_loss_for_dimensions, key = lambda l: l[0])
        self.dimension = min_loss_for_dimensions.index(min_loss)
        self.threshold = min_loss[1]
        self.sign = min_loss[2]



    def min_loss_in_dim(self,feature,labels,weights):
        losses = []
        for threshold in self.get_candidate_thresholds(feature):
            pred_labels_w_sign = self.generate_pred_labels(feature,threshold)
            for pred_label in pred_labels_w_sign:
                loss = self.loss_fn(labels,weights,pred_label[0])
                sign = pred_label[1]
                losses.append((loss,threshold,sign))
        
        min_loss_in_dim_and_threshold = min(losses, key = lambda l: l[0])
        return min_loss_in_dim_and_threshold

    
    def loss_fn(self,labels,weights, pred_labels):
        num_rows = labels.shape[0]

        sum = 0
        for i in range(num_rows):
            if labels[i]!=pred_labels[i]:
                sum+=weights[i]
        
        return sum
    
    def generate_pred_labels(self,feature,threshold):
        sign = [-1,1]
        pred_labels = []
        for s in sign:
            pred_labels_with_sign=self.generate_pred_labels_for_sign(feature,threshold,s)
            pred_labels.append((pred_labels_with_sign,s))
        return pred_labels
    
    def generate_pred_labels_for_sign(self,feature,threshold,sign):
        num_rows = feature.shape[0]

        pred_labels = []

        for i in range(num_rows):
            value = feature[i]
            label = self.get_pred_label_for_sign(value,threshold,sign)
            pred_labels.append(label)
           
        return  pred_labels
    
    def get_pred_label_for_sign(self, value,threshold,sign):

        if value>=threshold:
            return sign
        return sign * (-1)
        

    def get_candidate_thresholds(self, feature):
        feature_list = sorted(feature)
        num_rows = len(feature_list)

        candidate_thresholds=[-1]
        for i in range(1,num_rows):
            mid_point = 0.5 * (feature_list[i-1]+feature_list[i])
            candidate_thresholds.append(mid_point)
        candidate_thresholds.append(1)

        return candidate_thresholds
    

    def predict(self, data):
        num_rows = data.shape[0]
        points =[]
        for i in range(num_rows):
            if data[i][self.dimension]>=self.threshold:
                points.append(self.sign)
            else:
                points.append(self.sign*(-1))
        
        
        return np.array(points)
        '''
        Predicts labels of given datapoints.

        Arguments:
            data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].

        Returns:
            prediction: An ndarray with shape (n, ). Values are +1 or -1.
        '''
        




def bagging(data, labels, n_classifiers, n_samples, seed=0):
    '''
    Runs Bagging algorithm.

    Arguments:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
        n_classifiers: Number of classifiers to construct.
        n_samples: Number of samples to train each classifier.
        seed: Random seed for NumPy.

    Returns:
        classifiers: A list of classifiers.
    '''
    classifiers = []
    n = data.shape[0]

    for i in range(n_classifiers):
        np.random.seed(seed + i)
        sample_indices = np.random.choice(n, size=n_samples, replace=False)
        subset_data_list = []
        subset_label_list = []

        for j in sample_indices:
            subset_data_list.append(data[j])
            subset_label_list.append(labels[j])

        subset_data = np.array(subset_data_list)
        subset_labels = np.array(subset_label_list)
        classifier = Stump(subset_data, subset_labels, [1.0] * subset_labels.size)
        classifiers.append(classifier)

    return classifiers


def adaboost(data, labels, n_classifiers):
    '''
    Runs AdaBoost algorithm.

    Arguments:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
        n_classifiers: Number of classifiers to construct.

    Returns:
        classifiers: A list of classifiers.
        weights: A list of weights assigned to the classifiers.
    '''
    classifiers = []
    weights = []
    n = data.shape[0]
    data_weights = np.ones(n) / n
    

    for _ in range(n_classifiers):
        classifier = Stump(data, labels, data_weights)
        
        alpha = calc_alpha(classifier, labels, data, data_weights)

        for i in range(n):
            input_in_2d = np.expand_dims(data[i], axis=0) 
            predict = classifier.predict(input_in_2d)
            predict = predict.item()
            data_weights[i] = data_weights[i]*math.exp(-alpha*labels[i]*predict)
            z = np.sum(data_weights)
            data_weights= data_weights/z
    
        classifiers.append(classifier)
        weights.append(alpha)

    return classifiers, weights

def calc_alpha(classifier,labels,data,data_weights):
    n = data.shape[0]
    epsilon=0
    for i in range(n):
        input_in_2d = np.expand_dims(data[i], axis=0) 
        predict = classifier.predict(input_in_2d)
        predict = predict.item()
        if labels[i]!=predict:
            epsilon+=data_weights[i]
       
    return 0.5*np.log( (1-epsilon)/epsilon)


if __name__ == '__main__':
    data, labels = get_dataset_fixed()
    weights = [1.0] * labels.size
    stump = Stump(data, labels, weights)
    adaboost(data,labels,4)
   

    '''
    print(stump.dimension)
    print(stump.threshold)
    print(stump.sign)
    print("shape ")
    stump.predict(data)
    '''


    # You can play with the dataset and your algorithms here
    # classifier = Stump()
