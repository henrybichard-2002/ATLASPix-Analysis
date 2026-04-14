import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE

def prepare_hit_data(hit_dict, features=None):
    """
    Converts raw hit dictionary to a DataFrame for analysis.
    Handles distinct data types (uint16, int8) and missing values (-1).
    """
    if features is None:
        # Default Hit Features
        features = ['ToT', 'pToF', 'Row', 'Column'] 
        if 'BunchStatus' in hit_dict: features.append('BunchStatus')
        if 'TS2' in hit_dict: features.append('TS2') # Fine timestamp

    print(f"[Hit Data] Extracting features: {features}...", end=" ")
    
    data_map = {}
    n_total = len(hit_dict['xtalk_type'])
    
    for f in features:
        if f not in hit_dict: continue
        raw = hit_dict[f]
        
        # Convert to float for Analysis
        clean = raw.astype(float)
        
        # Handle specific sentinel values for Hits
        if f == 'pToF': clean[clean == -1] = np.nan
        if f == 'BunchStatus': clean[clean == -1] = np.nan
        
        data_map[f] = clean

    # Labels
    labels = hit_dict['xtalk_type'].astype(int)
    
    df = pd.DataFrame(data_map)
    df['label'] = labels
    
    # Filter valid classes (0=Clean, 1,2,3=Xtalk)
    df = df[df['label'].isin([0,1,2,3])].dropna()
    
    print(f"Done. ({len(df)} hits)")
    return df, features


def analyze_hit_qda(hit_dict, features=None, target_type=None):
    """
    Performs PCA Whitening + QDA on single hits.
    Visualizes separation in Energy (ToT) vs History (pToF) space.
    """
    # 1. Prep
    df, features = prepare_hit_data(hit_dict, features)
    if len(df) == 0: return

    xtalk_map = {1: "Type 1", 2: "Type 2", 3: "Type 3"}
    current_map = {target_type: xtalk_map[target_type]} if target_type else xtalk_map
    
    fig = plt.figure(figsize=(16, 4 * len(current_map)))
    grid = gridspec.GridSpec(len(current_map), 1, hspace=0.4)
    
    for row_idx, (xt, name) in enumerate(current_map.items()):
        subset = df[df['label'].isin([0, xt])]
        if len(subset) < 100: continue
            
        X = subset[features].values
        y = (subset['label'].values != 0).astype(int)

        # Pipeline
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=2, whiten=True)
        X_pca = pca.fit_transform(X_scaled)
        var = pca.explained_variance_ratio_
        
        # QDA (Regularized for discrete ToT values)
        qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
        qda.fit(X_pca, y)
        y_prob = qda.predict_proba(X_pca)[:, 1]
        
        # ROC
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        best_thresh = thresholds[np.argmax(tpr - fpr)]

        # PLOTS
        inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[row_idx], wspace=0.25)
        
        # Scatter
        ax = fig.add_subplot(inner[0])
        idx0, idx1 = np.where(y==0)[0], np.where(y==1)[0]
        n = min(len(idx0), len(idx1), 2000)
        s0 = np.random.choice(idx0, n, replace=False)
        s1 = np.random.choice(idx1, n, replace=False)
        
        # Jitter helps visualize integer ToT values
        j = np.random.normal(0, 0.05, (n, 2))
        ax.scatter(X_pca[s0,0]+j[:,0], X_pca[s0,1]+j[:,1], c='#4C72B0', alpha=0.3, s=5, label='Clean')
        ax.scatter(X_pca[s1,0]+j[:,0], X_pca[s1,1]+j[:,1], c='#C44E52', alpha=0.3, s=5, label=name)
        
        # Contour
        xm, xM = ax.get_xlim(); ym, yM = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(xm, xM, 50), np.linspace(ym, yM, 50))
        Z = qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[best_thresh], colors='k', linestyles='--')
        
        ax.set_title(f"{name} Hit Separation")
        ax.set_xlabel(f"PC1 ({var[0]:.1%})"); ax.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax.legend(loc='upper right')

        # Dist
        ax = fig.add_subplot(inner[1])
        sns.histplot(x=y_prob[y==0], bins=30, color='#4C72B0', element="step", fill=True, stat='density', alpha=0.3, ax=ax)
        sns.histplot(x=y_prob[y==1], bins=30, color='#C44E52', element="step", fill=True, stat='density', alpha=0.3, ax=ax)
        ax.axvline(best_thresh, c='k', ls='--')
        ax.set_title("QDA Score")

        # ROC
        ax = fig.add_subplot(inner[2])
        ax.plot(fpr, tpr, color='#55A868', lw=2.5, label=f'AUC={roc_auc:.2f}')
        ax.plot([0,1],[0,1], c='gray', ls='--')
        ax.set_title("ROC"); ax.legend(loc='lower right')

    plt.suptitle(f"Hit-Level QDA Analysis", fontsize=16)
    plt.tight_layout(); plt.show()
 

def analyze_hit_bdt(hit_dict, features=None, target_type=None):
    """
    Runs Gradient Boosting on single hits.
    Identifies if ToT, pToF, or Position is the dominant discriminator.
    """
    df, features = prepare_hit_data(hit_dict, features)
    if len(df) == 0: return

    xtalk_map = {1: "Type 1", 2: "Type 2", 3: "Type 3"}
    current_map = {target_type: xtalk_map[target_type]} if target_type else xtalk_map
    
    fig = plt.figure(figsize=(16, 4 * len(current_map)))
    grid = gridspec.GridSpec(len(current_map), 1, hspace=0.4)
    
    for row_idx, (xt, name) in enumerate(current_map.items()):
        subset = df[df['label'].isin([0, xt])]
        if len(subset) < 100: continue
        
        X = subset[features].values
        y = (subset['label'].values != 0).astype(int)
        
        inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[row_idx], wspace=0.25)

        # 1. Feature Importance
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(X, y)
        imps = clf.feature_importances_
        idxs = np.argsort(imps)
        
        ax = fig.add_subplot(inner[0])
        ax.barh(range(len(idxs)), imps[idxs], color='#4C72B0', edgecolor='k', alpha=0.8)
        ax.set_yticks(range(len(idxs))); ax.set_yticklabels([features[i] for i in idxs])
        ax.set_title(f"{name} Drivers")
        ax.grid(axis='x', ls='--')

        # 2. Stability (3-Fold CV)
        ax = fig.add_subplot(inner[1])
        cv = StratifiedKFold(n_splits=3)
        tprs = []; mean_fpr = np.linspace(0, 1, 50)
        
        for train, test in cv.split(X, y):
            clf.fit(X[train], y[train])
            fpr, tpr, _ = roc_curve(y[test], clf.predict_proba(X[test])[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr)); tprs[-1][0] = 0.0
            
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        
        ax.plot(mean_fpr, mean_tpr, color='#55A868', lw=2.5, label=f'AUC={mean_auc:.2f}')
        ax.plot([0,1],[0,1], ls='--', c='gray')
        ax.set_title(f"BDT Stability"); ax.legend(loc='lower right')

        # 3. Score
        ax = fig.add_subplot(inner[2])
        scores = clf.predict_proba(X)[:, 1]
        sns.histplot(x=scores[y==0], bins=30, color='#4C72B0', element="step", fill=True, label='Clean', ax=ax, stat='density')
        sns.histplot(x=scores[y==1], bins=30, color='#C44E52', element="step", fill=True, label=name, ax=ax, stat='density')
        ax.set_title("BDT Score"); ax.legend()

    plt.suptitle(f"Hit-Level BDT Analysis", fontsize=16)
    plt.tight_layout(); plt.show()  
    
    
# 1. Broad BDT Scan to find best features
analyze_hit_bdt(processed_data)

# 2. Targeted QDA Analysis (e.g., if you see ToT and pToF are best for Type 1)
analyze_hit_qda(processed_data, features=['ToT', 'pToF' 'Row'], target_type=1)