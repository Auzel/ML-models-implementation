import torch
import kmeans_utils

def l2_norm_squared(x):
    return torch.sum(x**2)


def l2_norm_squared_shifted(x,mean):
    return l2_norm_squared(x-mean)

def get_cluster_with_min_distance(x,c):
    if l2_norm_squared_shifted(x,c[:,0]) < l2_norm_squared_shifted(x,c[:,1]):
        return 0
    return 1

def get_R(X,c):
    num_rows = X.shape[1]
    num_cols = c.shape[1]
    
    matrix_R = torch.zeros((num_rows,num_cols))

    for i in range(num_rows):
        k = get_cluster_with_min_distance(X[:,i],c)
        matrix_R[i,k]=1

    return matrix_R

def k_means(X=None, init_c=None, n_iters=3):
    """K-Means.
    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a cluster center.
    
    Return:
        c: shape [2, 2]. Each column is a cluster center.
    """
    
    # loading data and initialization of the cluster centers
    c=init_c
    if X is None:
        X, c = hw4_utils.load_data()

    # your code below
    
    for j in range(n_iters):
       
        # first solve the assignment problem given the centers c
        assignment = get_R(X,c)
        
        num_rows = assignment.shape[0]
        num_cols = assignment.shape[1]
        
        # then solve the cluster center problem given the assignments
        mean=torch.empty((X.shape[0],num_cols))
        
        for k in range(num_cols):
            weight=torch.zeros(num_cols)
            for i in range(num_rows):
                weight+=assignment[i,k]*X[:,i]
            sum_of_assignment = torch.sum(assignment[:,k])
            mean[:,k]=weight/sum_of_assignment

        c=mean
        # visulize the current clustering using hw4_utils.vis_cluster. 
        # with n_iters=3, there will be 3 figures. Put those figures in your written report. 
        
    return c


if __name__=='__main__':
    k_means()