import torch
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    """
    Given a 2D tensor X and integer degree, return a new tensor of all polynomial feature combinations
    (with constant term), sorted for each sample from smallest to largest.
    """
    # Hint: Use combinations_with_replacement and torch.prod.
    X=torch.as_tensor(X,dtype=torch.float32)
    n_samples,n_feats=X.shape
    poly_terms=[]
    for total_deg in range(0,degree+1):
        for exp_tuple in combinations_with_replacement(range(n_feats),total_deg):
            exp=torch.zeros(n_feats,dtype=torch.int32)
            for idx in exp_tuple:
                exp[idx]+=1
            term=torch.prod(X**exp,dim=1,keepdim=True)
            poly_terms.append(term)
    
    poly_matrix=torch.cat(poly_terms,dim=1)
    poly_matrix,_=torch.sort(poly_matrix,dim=1)
    return poly_matrix
