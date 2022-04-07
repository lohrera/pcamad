import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import PCA

class PCAMAD():
    
    # paper: Y. Huang et al. / PCA-MAD: A robust anomaly detection algorithm based on principal component analysis
    
    def __init__(self, n_components=None, _evalvariant=0, seed=0, verbose=0, trn_tst_split=False):
        super(PCAMAD, self)
        
        variants = {
            0: 'PCA-MAD++',
            1: 'PCA-MAD',
            3: 'PCA-1'
        }
        
        self.seed=seed
        self.pca=None
        self.M=None
        self.MAD=None
        self.n_components=n_components
        self.eigenvalues=None
        self.eigenvalues_corrected=None
        self.eigenvectors=None
        self.verbose=verbose
        self._trn_tst_split = trn_tst_split
        self._evalvariant = _evalvariant
        self._name = variants[_evalvariant]
        self.decision_scores_=None
        
        
    def fit(self, x):
        
        if self.n_components is None:
            self.n_components = x.shape[1]
        
        # 1. calculate eigenvectors e, eigenvalues lambda, x_trn-projections z_trn on D_train
        pca = PCA(n_components=self.n_components, random_state=self.seed)
        pca.fit(x)

        self.eigenvalues = pca.explained_variance_
        self.eigenvectors = pca.components_
        if self.verbose: print(f"eigenvectors e:\n{self.eigenvectors}\neigenvalues lambda:\n{self.eigenvalues} ")
        
        # -- X_prj_z (train) <- z_trn = e_T*x_trn
        df_X = pd.DataFrame(x)
        X_prj_z = pca.transform(df_X)
        df_X_prj_z = pd.DataFrame(X_prj_z)
        
        # 2a. calculate scaling factors m and mad on D_train
        (n,p) = df_X_prj_z.shape # train samples i to n / components j to p
        
        # 1st loops j2p/i2n        
        axis=0
        M, MAD=[],[]
        for j in range(0,p):
            
            S = np.power(df_X_prj_z.iloc[:, j], 2)

            m = np.median(S, axis=axis, keepdims=True) # solving median m of S
            mad = 1.4826 * np.median(np.abs(S - m), axis=axis, keepdims=True) # solving MAD mad of S

            M.append(m)
            MAD.append(mad)

        self.M = np.array(M).squeeze()
        self.MAD = np.array(MAD).squeeze()            
        self.pca = pca
        
        if not self._trn_tst_split: self.predict(x)
        

    def scale_scores(self, S_tst):
        S_pr=[]
        (m,p) = S_tst.shape # test samples i to m / components j to p
        for j in range(0,p):
            S_pr.append( (S_tst[:,j] - self.M[j]) / self.MAD[j] ) # standardized mad_scaled_zscores S_prime
        S_pr = np.array(S_pr).T
        return S_pr
        
    def predict(self, x):

        if self.pca is None:
            print(f"not yet fitted!")
            return

        # pca.transform(testdata)
        # -- X_prj_z (test) <- z_tst = e_T*x_tst
        df_X = pd.DataFrame(x)
        X_prj_z = self.pca.transform(df_X)
        
        S_tst = np.power(X_prj_z, 2) # square of z_tst

        if self._evalvariant == 3: 
            y_preds = self.calculate_weighted_distance_scores(S_tst) # PCA-1
            
        elif self._evalvariant == 0: 
            S_pr = self.scale_scores(S_tst)
            y_preds = self.calculate_corrected_outlier_scores(S_pr) # PCA-MAD++
            
        elif self._evalvariant == 1: 
            S_pr = self.scale_scores(S_tst)
            y_preds = self.calculate_outlier_scores(S_pr) # PCA-MAD
        
        self.decision_scores_ = y_preds
        
        return y_preds
        
         
    def calculate_outlier_scores(self, S_pr):

        #Eq.(7) outlier score

        outlier_scores = (self.eigenvalues * S_pr).sum(axis=1)
        return outlier_scores
    
    def calculate_corrected_outlier_scores(self, S_pr):
    
        #Eq.(9) outlier score with mad-scale-corrected eigenvalues

        variances = S_pr.var(axis=0)
        indices_sorted_desc = np.argsort(variances)[::-1] #psi
        eigenvalues_corrected = variances[indices_sorted_desc]
        outlier_scores = (eigenvalues_corrected * S_pr).sum(axis=1)

        return outlier_scores
    
    def calculate_weighted_distance_scores(self, S):        
        wosd=[]
        (m,p) = S.shape
        for j in range(0,m):
            wosd_=0
            x = S[j,:]
            for i in range(0,p):
                wosd_ += np.power(distance.euclidean(x, self.eigenvectors[i]),2)/self.eigenvalues[i]
            wosd.append(wosd_)
        wosd = np.array(wosd)        
        return wosd