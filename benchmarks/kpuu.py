import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau
from unipka._internal.solvation import get_solvation_energy
from rdkit import Chem
import ray
from unipka.unipka import UnipKa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve



def calculate_corrected_solvation_energy(mol):
    calc = UnipKa(use_simple_smarts=False)
    sp, ref_df = calc.get_state_penalty(mol, pH=7.4)
    mol = ref_df.iloc[0].mol
    G_solv = get_solvation_energy(mol)
    return G_solv, sp

def _calculate_corrected_solvation_energy(mol):
    try: 
        return calculate_corrected_solvation_energy(mol)
    except:
        print(Chem.MolToSmiles(mol))
        return None, None


def analyze_brain_penetration(df, kpuu_column='Kpuu'):
    """
    Analyze brain penetration using solvation energy descriptors
    
    Parameters:
    df: DataFrame with columns 'G_solv', 'sp', and Kp_uu values
    kpuu_column: name of the column containing Kp,uu values
    """
    
    # Create binary brain-penetrant class (Kp,uu > 0.3, non-log transformed)
    y_binary = (df[kpuu_column] > 0.3).astype(int)
    
    print(f"Brain-penetrant compounds: {y_binary.sum()} / {len(y_binary)} ({100*y_binary.mean():.1f}%)")
    
    # Prepare descriptors
    # Schrodinger approach: G_solv only
    X1 = df[['G_solv']].values
    
    # AIMNet2/CPCM-X approach: G_solv + SP
    X2 = (df['G_solv'] + df['sp']).values.reshape(-1, 1)
    
    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train and evaluate both models
    models = {
        'G_solv': X1,
        'G_solv + SP': X2
    }
    
    results = {}
    
    for name, X in models.items():
        # Create pipeline with scaling and logistic regression
        pipeline = LogisticRegression()
        
        # Cross-validation ROC-AUC scores
        cv_scores = cross_val_score(pipeline, X, y_binary, cv=cv, scoring='roc_auc')
        
        # Fit pipeline on full dataset for ROC curve
        pipeline.fit(X, y_binary)
        y_pred_proba = pipeline.predict_proba(X)[:, 1]
        
        # Store results
        results[name] = {
            'cv_scores': cv_scores,
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'y_pred_proba': y_pred_proba,
            'model': pipeline  # Store the full pipeline
        }
        
        print(f"\n{name}:")
        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  CV scores: {[f'{score:.3f}' for score in cv_scores]}")
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_binary, result['y_pred_proba'])
        auc_score = roc_auc_score(y_binary, result['y_pred_proba'])
        
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Brain Penetration Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmarks/kpuu_results.png", dpi=300)
    plt.close()



def process_dataset():
    """Process a single dataset and return results DataFrame with metrics"""
    print(f"\n--- Processing Kpuu ---")
    
    df = pd.read_csv("/tmp/kpuu.csv")
    df.Kpuu = df.Kpuu.apply(lambda x: float(x.replace(",",".")))

    df['mol']  = [Chem.MolFromSmiles(s) for s in df['canonical SMILES']]
    df = df.dropna(subset=['mol'])

    ray_func = ray.remote(_calculate_corrected_solvation_energy)
    results = ray.get([ray_func.remote(smi) for smi in df.mol])
    df['G_solv'], df['sp'] = zip(*results)
    df = df.dropna(subset=['G_solv'])
    
    # Clean data
    old_len = len(df)
    df.dropna(subset=["G_solv"], inplace=True)
    new_len = len(df)

    print(f"Failed to generate pKa for {old_len-new_len} molecules in Kpuu")

    if df.empty:
        print(f"No data to evaluate for Kpuu after dropping rows with missing values.")
        return None, {}

    return df

def main():    
    
    df = process_dataset()

    import pdb; pdb.set_trace()

    clf = LogisticRegression()
    X = (df['G_solv'] + df['sp']).values.reshape(-1, 1)
    y = df.Kpuu > 0.3
    clf.fit(df[['G_solv','sp']].values, y)

    corrected_weights = clf.coef_[0]  # Shape: (n_features,) for binary classification
    corrected_bais = clf.intercept_[0]  # Scalar for binary classification

    print("CORRECTED:", corrected_weights, corrected_bais)

    clf = LogisticRegression()
    clf.fit(df[['G_solv']].values, y)

    uncorrected_weights = clf.coef_[0]  # Shape: (n_features,) for binary classification
    uncorrected_bias = clf.intercept_[0]  # Scalar for binary classification

    print("UNCORRECTED:", uncorrected_weights, uncorrected_bias)

    analyze_brain_penetration(df)



if __name__=="__main__":
    main()