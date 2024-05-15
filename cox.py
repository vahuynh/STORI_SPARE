import numpy as np
import pickle
from operator import itemgetter
import os
import pandas
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
from sklearn.utils import resample
import itertools
import argparse

DATA_FOLDER = 'data'

def remove_nan(data):
    """
    Remove patients with missing data.

    Parameters
    ----------
    data : tuple
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.

    Returns
    -------
    tuple
        Data tuple (same format as input data) with patients with missing data removed.

    """

    (X, features, patients, y, relapse_time) = data

    # Keep only patients for which we have all the data
    mask = ~(np.isnan(X).any(axis=1))
    X_reduced = X[mask]
    y_reduced = y[mask]
    relapse_time_reduced = relapse_time[mask]
    patients_reduced = [patient for patient, m in zip(patients, mask) if m]

    return X_reduced, features, patients_reduced, y_reduced, relapse_time_reduced


def bootstrap_sample_(data):
    """
    Create a bootstrap sample.

    Parameters
    ----------
    data : tuple
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.

    Returns
    -------
    tuple
        Bootstrap sample (X_boot, features, patients_boot, y_boot, relapse_time_boot).

    """

    (X, features, patients, y, relapse_time) = data
    n_patients = len(patients)

    idx_boot = resample(np.arange(n_patients), stratify=y)

    X_boot = X[idx_boot]
    y_boot = y[idx_boot]
    relapse_time_boot = relapse_time[idx_boot]
    patients_boot = [patients[i] for i in idx_boot]

    return X_boot, features, patients_boot, y_boot, relapse_time_boot

def bootstrap_sample(data):
    """
    Create a bootstrap sample, ensuring that there are at least 3 patients in each class.

    Parameters
    ----------
    data : tuple
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.

    Returns
    -------
    tuple
        Bootstrap sample (X_boot, features, patients_boot, y_boot, relapse_time_boot).
    """

    data_boot = bootstrap_sample_(data)
    data_boot_reduced = remove_nan(data_boot)
    y_boot = data_boot_reduced[3]

    while np.sum(y_boot) < 3:
        
        data_boot = bootstrap_sample_(data)
        data_boot_reduced = remove_nan(data_boot)
        y_boot = data_boot_reduced[3]

    return data_boot


def cox_fit_protein_univariate(data, protein, penalizer=0.):
    """
    Fit a Cox proportional hazards model for a single protein.

    Parameters
    ----------
    data : development set
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.
    protein : str
        Protein name.
    penalizer : float, optional
        Regularization parameter in the Cox model (L2 penalty).

    Returns
    -------
    tuple (concordance, n_patients, effect)
        concordance : float
            Concordance index on the development set.
        n_patients : int
            Number of patients used to fit the Cox model.
        effect : str
            Effect of the protein on the relapse time, either 'pos' (when beta >= 0) or 'neg' (when beta < 0).
    """


    (X, features, patients, y, relapse_time) = data

    idx = features.index(protein)
    X_prot= X[:, idx].reshape(-1, 1)
    feature = [protein]

    (X_prot, feature, patients, y, relapse_time) = remove_nan((X_prot, feature, patients, y, relapse_time))
    n_patients = len(patients)

    data_train = np.zeros((n_patients, 3))
    data_train[:, 0] = y
    data_train[:, 1] = relapse_time
    data_train[:, 2] = X_prot.flatten()
    collabels = ['class', 'relapse_time', protein]
    data_train = pandas.DataFrame(data_train, columns=collabels)


    # Cox regression
    # We add a small regularizer to avoid problems when a protein perfectly isolates one class.
    # This does not change much the results for the other proteins
    # (i.e. those that do not perfectly isolate one class).
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(data_train, duration_col='relapse_time', event_col='class', fit_options={'step_size': 0.5})

    res = cph.summary.values
    beta = res[0, 0]
    if beta >= 0:
        effect = 'pos'
        predicted_survival_scores = -X_prot
    else:
        effect = 'neg'
        predicted_survival_scores = X_prot

    concordance = concordance_index(event_times=relapse_time,
                                    predicted_scores=predicted_survival_scores,
                                    event_observed=y)

    return concordance, n_patients, effect


def cox_fit_protein_bivariate(data, protein1, protein2, penalizer=0.):
    """
    Fit a Cox proportional hazards model for a pair of proteins.

    Parameters
    ----------
    data : development set
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.
    protein1 : str
        First protein name.
    protein2 : str
        Second protein name.
    penalizer : float, optional
        Regularization parameter in the Cox model (L2 penalty).

    Returns
    -------
    tuple (concordance, n_patients, cph)
        concordance : float
            Concordance index on the development set.
        n_patients : int
            Number of patients used to fit the Cox model.
        cph : CoxPHFitter
            Fitted Cox model.
    """

    (X, features, patients, y, relapse_time) = data

    idx1 = features.index(protein1)
    idx2 = features.index(protein2)
    X_prot = np.concatenate((X[:, idx1].reshape(-1, 1), X[:, idx2].reshape(-1, 1)), axis=1)
    features = [protein1, protein2]

    (X_prot, features, patients, y, relapse_time) = remove_nan((X_prot, features, patients, y, relapse_time))
    n_patients = len(patients)

    data_train = np.zeros((n_patients, 4))
    data_train[:, 0] = y
    data_train[:, 1] = relapse_time
    data_train[:, 2:] = X_prot
    collabels = ['class', 'relapse_time', protein1, protein2]
    data_train = pandas.DataFrame(data_train, columns=collabels)


    # Cox regression
    # We add a small regularizer to avoid problems when a protein perfectly isolates one class.
    # This does not change much the results for the other proteins
    # (i.e. those that do not perfectly isolate one class).
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(data_train, duration_col='relapse_time', event_col='class', fit_options={'step_size': 0.5})

    predicted_survival_scores = -cph.predict_log_partial_hazard(data_train)

    concordance = concordance_index(event_times=relapse_time,
                                    predicted_scores=predicted_survival_scores,
                                    event_observed=y)

    return concordance, n_patients, cph


def sample_size(data, protein1, protein2):
    """
    Compute the number of patients with non-missing data for a pair of proteins.

    Parameters
    ----------
    data : tuple
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.
    protein1 : str
        First protein name.
    protein2 : str
        Second protein name.

    Returns
    -------
    int
        Number of patients with non-missing data for the pair of proteins.

    """

    (X, features, patients, y, relapse_time) = data

    idx1 = features.index(protein1)
    idx2 = features.index(protein2)
    X_prot = np.concatenate((X[:, idx1].reshape(-1, 1), X[:, idx2].reshape(-1, 1)), axis=1)
    features = [protein1, protein2]

    (X_prot, features, patients, y, relapse_time) = remove_nan((X_prot, features, patients, y, relapse_time))

    return len(patients)


def cox_test_protein_univariate(data, protein, effect):
    """
    Compute the concordance index of a univariate Cox model on a validation set.

    Parameters
    ----------
    data : validation set
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.
    protein : str
        Protein name.
    effect : str
        Effect of the protein on the relapse time, either 'pos' (when beta >= 0) or 'neg' (when beta < 0).

    Returns
    -------
    tuple (concordance, n_patients)
        concordance : float
            Concordance index on the validation set.
        n_patients : int
            Number of patients (those without missing data) used to compute the concordance index.

    """


    (X, features, patients, y, relapse_time) = data

    idx = features.index(protein)
    X_prot = X[:, idx].reshape(-1, 1)
    feature = [protein]

    (X_prot, feature, patients, y, relapse_time) = remove_nan((X_prot, feature, patients, y, relapse_time))
    n_patients = len(patients)

    if effect == 'pos':
        predicted_survival_scores = -X_prot
    else:
        predicted_survival_scores = X_prot

    concordance = concordance_index(event_times=relapse_time,
                                    predicted_scores=predicted_survival_scores,
                                    event_observed=y)

    return concordance, n_patients



def cox_test_protein_bivariate(data, protein1, protein2, cph):
    """
    Compute the concordance index of a bivariate Cox model on a validation set.

    Parameters
    ----------
    data : validation set
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.
    protein1 : str
        First protein name.
    protein2 : str
        Second protein name.
    cph : CoxPHFitter
        Fitted Cox model.

    Returns
    -------
    tuple (concordance, n_patients)
        concordance : float
            Concordance index on the validation set.
        n_patients : int
            Number of patients (those without missing data) used to compute the concordance index.

    """

    (X, features, patients, y, relapse_time) = data

    idx1 = features.index(protein1)
    idx2 = features.index(protein2)
    X_prot = np.concatenate((X[:, idx1].reshape(-1, 1), X[:, idx2].reshape(-1, 1)), axis=1)
    features = [protein1, protein2]

    (X_prot, features, patients, y, relapse_time) = remove_nan((X_prot, features, patients, y, relapse_time))
    n_patients = len(patients)

    data_test = np.zeros((n_patients, 4))
    data_test[:, 0] = y
    data_test[:, 1] = relapse_time
    data_test[:, 2:] = X_prot
    collabels = ['class', 'relapse_time', protein1, protein2]
    data_test = pandas.DataFrame(data_test, columns=collabels)

    predicted_survival_scores = -cph.predict_log_partial_hazard(data_test)

    concordance = concordance_index(event_times=relapse_time,
                                    predicted_scores=predicted_survival_scores,
                                    event_observed=y)

    return concordance, n_patients


def test_baseline(data, baseline):
    """
    Compute the concordance index of a univariate baseline model (using either hsCRP or FC) on a validation set.

    Parameters
    ----------
    data : validation set
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.
    baseline : str
        Baseline protein name ('hsCRP' or 'FC').

    Returns
    -------
    tuple (concordance, n_patients)
        concordance : float
            Concordance index on the validation set.
        n_patients : int
            Number of patients (those without missing data) used to compute the concordance index.

    """

    (X, features, patients, y, relapse_time) = data

    idx = features.index(baseline)
    X_baseline = X[:, idx].reshape(-1, 1)
    feature = [baseline]

    (X_baseline, feature, patients, y, relapse_time) = remove_nan((X_baseline, feature, patients, y, relapse_time))
    predicted_survival_scores = -X_baseline.flatten()

    concordance = concordance_index(event_times=relapse_time,
                                    predicted_scores=predicted_survival_scores,
                                    event_observed=y)

    return concordance, len(patients)

def test_cease_model(data, cease_model):
    """
    Compute the concordance index of a CEASE model on a validation set.

    Parameters
    ----------
    data : validation set
        Data tuple (X, features, patients, y, relapse_time).
        X: np.array, shape (n_patients, n_features)
            Data matrix.
        features: list
            List of feature names.
        patients: list
            List of patient IDs.
        y: np.array, shape (n_patients,)
            Class labels (0: no relapse, 1: relapse).
        relapse_time: np.array, shape (n_patients,)
            Relapse times.
    cease_model : str
        Type of CEASE model ('CEASE_phase0' or 'CEASE_phase1').

    Returns
    -------
    tuple (concordance, n_patients)
        concordance : float
            Concordance index on the validation set.
        n_patients : int
            Number of patients (those without missing data) used to compute the concordance index.

    """

    betas = dict()

    if cease_model == 'CEASE_phase0':

        betas['age_per_10y'] = np.log(0.86)
        betas['smoking_yes_vs_no'] = np.log(1.39)
        betas['age_at_diagnosis_A2_vs_A1'] = np.log(0.69)
        betas['age_at_diagnosis_A3_vs_A1'] = np.log(0.71)
        betas['l4_yes_vs_no'] = np.log(1.32)
        betas['disease_duration_per_5y'] = np.log(1.07)
        betas['immunosuppressant_yes_vs_no'] = np.log(0.70)
        betas['anti_tnf_type_IFX_vs_ADA'] = np.log(0.82)
        betas['second_line_anti_tnf_yes_vs_no'] = np.log(1.32)
        betas['clinical_remission_yes_vs_no'] = np.log(0.45)
        betas['CRP_per_doubling_mg_L'] = np.log(1.04)

    elif cease_model == 'CEASE_phase1':

        betas['age_per_10y'] = np.log(0.94)
        betas['smoking_yes_vs_no'] = np.log(1.31)
        betas['age_at_diagnosis_per_5y'] = np.log(0.94)
        betas['l4_yes_vs_no'] = np.log(1.15)
        betas['immunosuppressant_yes_vs_no'] = np.log(0.68)
        betas['anti_tnf_type_IFX_vs_ADA'] = np.log(0.87)
        betas['second_line_anti_tnf_yes_vs_no'] = np.log(1.13)
        betas['CRP_per_doubling_mg_L'] = np.log(1.02)
        betas['FC_per_doubling_ug_g'] = np.log(1.10)

    else:

        raise ValueError(f'Unknown CEASE model: {cease_model}')

    features_cease = betas.keys()
    betas_ = np.array([betas[feature] for feature in features_cease])

    (X, features, patients, y, relapse_time) = remove_nan(data)
    feat_idx = [features.index(feature) for feature in features_cease]
    predicted_survival_scores = -np.dot(X[:, feat_idx], betas_)

    concordance = concordance_index(event_times=relapse_time,
                                    predicted_scores=predicted_survival_scores,
                                    event_observed=y)

    return concordance, len(patients)



def write_results_univariate(results_dir, output_file, results):
    """
    Write the results of the univariate analysis to a file.

    Parameters
    ----------
    results_dir : str
        Directory where the results file will be saved.
    output_file : str
        Name of the output file.
    results : dict
        Dictionary containing the results of the univariate analysis.
    """

    res_sort = list()
    for protein in results:
        res_sort.append((protein, results[protein]['c_index_valid_mean']))

    res_sort = sorted(res_sort, key=itemgetter(1), reverse=True)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    column_format = dict()
    column_format['dataset'] = '%s'
    column_format['n_patients_STORI'] = '%d'
    column_format['n_patients_SPARE'] = '%d'
    column_format['effect_STORI'] = '%s'
    column_format['effect_SPARE'] = '%s'
    column_format['c_index_dev_STORI'] = '%.3f'
    column_format['c_index_dev_SPARE'] = '%.3f'
    column_format['c_index_dev_mean'] = '%.3f'
    column_format['c_index_valid_STORI'] = '%.3f'
    column_format['c_index_valid_SPARE'] = '%.3f'
    column_format['c_index_valid_mean'] = '%.3f'
    column_format['c_index_valid_STORI_reduced'] = '%.3f'
    column_format['c_index_valid_SPARE_reduced'] = '%.3f'
    column_format['c_index_valid_mean_reduced'] = '%.3f'


    with open('%s/%s' % (results_dir, output_file), 'w') as f:

        header = 'protein'
        for key in column_format:
            header += '\t' + key

        f.write(header + '\n')

        for protein, _ in res_sort:
            line = protein
            for key in column_format:
                if key in results[protein]:
                    line += '\t' + column_format[key] % results[protein][key]
                else:
                    line += '\t' + 'NA'
            f.write(line + '\n')



def write_results_bivariate(results_dir, output_file, results):
    """
    Write the results of the bivariate analysis to a file.

    Parameters
    ----------
    results_dir : str
        Directory where the results file will be saved.
    output_file : str
        Name of the output file.
    results : dict
        Dictionary containing the results of the bivariate analysis.
    """

    res_sort = list()
    for pair in results:
        res_sort.append((pair, results[pair]['c_index_valid_mean']))

    res_sort = sorted(res_sort, key=itemgetter(1), reverse=True)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    column_format = dict()
    column_format['n_patients_STORI'] = '%d'
    column_format['n_patients_SPARE'] = '%d'
    column_format['c_index_dev_STORI'] = '%.3f'
    column_format['c_index_dev_STORI_low'] = '%.3f'
    column_format['c_index_dev_STORI_high'] = '%.3f'
    column_format['c_index_dev_SPARE'] = '%.3f'
    column_format['c_index_dev_SPARE_low'] = '%.3f'
    column_format['c_index_dev_SPARE_high'] = '%.3f'
    column_format['c_index_dev_mean'] = '%.3f'
    column_format['c_index_dev_mean_low'] = '%.3f'
    column_format['c_index_dev_mean_high'] = '%.3f'
    column_format['c_index_valid_STORI'] = '%.3f'
    column_format['c_index_valid_STORI_low'] = '%.3f'
    column_format['c_index_valid_STORI_high'] = '%.3f'
    column_format['c_index_valid_SPARE'] = '%.3f'
    column_format['c_index_valid_SPARE_low'] = '%.3f'
    column_format['c_index_valid_SPARE_high'] = '%.3f'
    column_format['c_index_valid_mean'] = '%.3f'
    column_format['c_index_valid_mean_low'] = '%.3f'
    column_format['c_index_valid_mean_high'] = '%.3f'
    column_format['c_index_valid_STORI_reduced'] = '%.3f'
    column_format['c_index_valid_STORI_reduced_low'] = '%.3f'
    column_format['c_index_valid_STORI_reduced_high'] = '%.3f'
    column_format['c_index_valid_SPARE_reduced'] = '%.3f'
    column_format['c_index_valid_SPARE_reduced_low'] = '%.3f'
    column_format['c_index_valid_SPARE_reduced_high'] = '%.3f'
    column_format['c_index_valid_mean_reduced'] = '%.3f'
    column_format['c_index_valid_mean_reduced_low'] = '%.3f'
    column_format['c_index_valid_mean_reduced_high'] = '%.3f'

    with open('%s/%s' % (results_dir, output_file), 'w') as f:

        header = 'protein1\tprotein2'
        for key in column_format:
            header += '\t' + key

        f.write(header + '\n')

        for pair, _ in res_sort:

            if pair in ['CEASE_phase0', 'CEASE_phase1', 'hsCRP', 'FC']:
                protein1 = pair
                protein2 = pair
            else:

                protein1, protein2 = pair.split('-')

            line = f'{protein1}\t{protein2}'
            for key in column_format:
                if key in results[pair]:
                    line += '\t' + column_format[key] % results[pair][key]
                else:
                    line += '\t' + 'NA'
            f.write(line + '\n')



def baselines(stratification_type, reduced):
    """
    Do the univariate analysis for the baseline proteins (hsCRP and FC).

    Parameters
    ----------
    stratification_type : str
        Stratification type ('initial', 'before_6months', 'after_6months').
    reduced : str
        Type of reduced data (can be only 'selected_markers_reduced' for the moment).

    Returns
    -------
    dict
        Dictionary containing the results of the univariate analysis for the baseline proteins.
    """

    results = dict()

    # CEASE models
    for baseline in ['hsCRP', 'FC']:

        results[baseline] = dict()

        for dataset in ['STORI', 'SPARE']:

            # Non-reduced data
            with open(f'{DATA_FOLDER}/Dataset_{dataset}/{dataset}_data_pkl/{dataset}_{baseline}_{stratification_type}.pkl', 'rb') as f:
                data_nonreduced = pickle.load(f)

            c_index, n_patients = test_baseline(data_nonreduced, baseline)
            results[baseline][f'c_index_valid_{dataset}'] = c_index
            results[baseline][f'n_patients_{dataset}'] = n_patients

            if baseline == 'hsCRP':

                # Reduced data
                with open(f'{DATA_FOLDER}/Dataset_{dataset}/{dataset}_data_pkl/{dataset}_{reduced}_{stratification_type}.pkl', 'rb') as f:
                    data_reduced = pickle.load(f)

                if reduced == 'selected_markers_reduced':
                    X_reduced_tmp = data_reduced[0]
                    features_reduced_tmp = data_reduced[1]
                    idx_hsCRP = features_reduced_tmp.index('CRP_per_doubling_mg_L')
                    features_reduced_tmp[idx_hsCRP] = 'hsCRP'
                    X_reduced_tmp[:, idx_hsCRP] = 2**X_reduced_tmp[:, idx_hsCRP]
                    data_reduced = (X_reduced_tmp, features_reduced_tmp, data_reduced[2], data_reduced[3], data_reduced[4])

                c_index, _ = test_baseline(data_reduced, baseline)
                results[baseline][f'c_index_valid_{dataset}_reduced'] = c_index

        results[baseline]['c_index_valid_mean'] = (results[baseline]['c_index_valid_STORI'] + results[baseline]['c_index_valid_SPARE']) / 2

        if baseline == 'hsCRP':
            results[baseline]['c_index_valid_mean_reduced'] = (results[baseline]['c_index_valid_STORI_reduced'] + results[baseline]['c_index_valid_SPARE_reduced']) / 2

    return results

def cease_models(stratification_type, reduced):
    """
    Do the analysis for the CEASE models.

    Parameters
    ----------
    stratification_type : str
        Stratification type ('initial', 'before_6months', 'after_6months').
    reduced : str
        Type of reduced data (either 'reduced' or 'selected_markers_reduced').

    Returns
    -------
    dict
        Dictionary containing the results for the CEASE models.

    """

    results = dict()

    # CEASE models
    for model in ['CEASE_phase0', 'CEASE_phase1']:

        results[model] = dict()
        results[model]['dataset'] = 'clinical'

        for dataset in ['STORI', 'SPARE']:

            # Non-reduced data
            with open(f'{DATA_FOLDER}/Dataset_{dataset}/{dataset}_data_pkl/{dataset}_{model}_{stratification_type}.pkl', 'rb') as f:
                data_nonreduced = pickle.load(f)

            c_index, n_patients = test_cease_model(data_nonreduced, model)
            results[model][f'c_index_valid_{dataset}'] = c_index
            results[model][f'n_patients_{dataset}'] = n_patients

            if model == 'CEASE_phase0':

                # Reduced data
                with open(f'{DATA_FOLDER}/Dataset_{dataset}/{dataset}_data_pkl/{dataset}_{reduced}_{stratification_type}.pkl', 'rb') as f:
                    data_reduced = pickle.load(f)

                c_index, _ = test_cease_model(data_reduced, model)
                results[model][f'c_index_valid_{dataset}_reduced'] = c_index

        results[model]['c_index_valid_mean'] = (results[model]['c_index_valid_STORI'] + results[model]['c_index_valid_SPARE']) / 2

        if model == 'CEASE_phase0':
            results[model]['c_index_valid_mean_reduced'] = (results[model]['c_index_valid_STORI_reduced'] + results[model]['c_index_valid_SPARE_reduced']) / 2

    return results


def univariate_analysis(stratification_type, penalizer=0.):
    """
    Do the univariate analysis for each protein.

    Parameters
    ----------
    stratification_type : str
        Stratification type ('initial', 'before_6months', 'after_6months').
    penalizer : float, optional
        Regularization parameter in the Cox model (L2 penalty).

    Returns
    -------
    dict
        Dictionary containing the results of the univariate analysis for each protein.
    """

    print(f'Univariate analysis: {stratification_type} dataset')

    protein_types = ['hsCRP', 'FC', 'PEA_IR', 'PEA_cytokine', 'SRM']

    results = dict()

    for protein_type in protein_types:

        # Retrieve protein names
        with open(f'{DATA_FOLDER}/Dataset_STORI/STORI_data_pkl/STORI_{protein_type}_{stratification_type}.pkl', 'rb') as f:
            data_tmp = pickle.load(f)
        proteins = data_tmp[1]

        for protein in proteins:

            results[protein] = dict()
            if protein in ['hsCRP', 'FC']:
                results[protein]['dataset'] = 'clinical'
            else:
                results[protein]['dataset'] = protein_type

            for (dataset_dev, dataset_valid) in [('STORI', 'SPARE'), ('SPARE', 'STORI')]:

                with open(f'{DATA_FOLDER}/Dataset_{dataset_dev}/{dataset_dev}_data_pkl/{dataset_dev}_{protein_type}_{stratification_type}.pkl', 'rb') as f:
                    data_dev = pickle.load(f)

                with open(f'{DATA_FOLDER}/Dataset_{dataset_valid}/{dataset_valid}_data_pkl/{dataset_valid}_{protein_type}_{stratification_type}.pkl', 'rb') as f:
                    data_valid_nonreduced = pickle.load(f)

                c_index_dev, n_patients_dev, effect_dev = cox_fit_protein_univariate(data_dev, protein, penalizer)
                c_index_valid_nonreduced, _ = cox_test_protein_univariate(data_valid_nonreduced, protein, effect_dev)

                results[protein][f'n_patients_{dataset_dev}'] = n_patients_dev
                results[protein][f'effect_{dataset_dev}'] = effect_dev
                results[protein][f'c_index_dev_{dataset_dev}'] = c_index_dev
                results[protein][f'c_index_valid_{dataset_valid}'] = c_index_valid_nonreduced

                if protein != 'FC':

                    with open(f'{DATA_FOLDER}/Dataset_{dataset_valid}/{dataset_valid}_data_pkl/{dataset_valid}_reduced_{stratification_type}.pkl', 'rb') as f:
                        data_valid_reduced = pickle.load(f)

                    c_index_valid_reduced, _ = cox_test_protein_univariate(data_valid_reduced, protein, effect_dev)

                    results[protein][f'c_index_valid_{dataset_valid}_reduced'] = c_index_valid_reduced

            results[protein]['c_index_dev_mean'] = (results[protein]['c_index_dev_STORI'] + results[protein]['c_index_dev_SPARE']) / 2
            results[protein]['c_index_valid_mean'] = (results[protein]['c_index_valid_STORI'] + results[protein]['c_index_valid_SPARE']) / 2

            if protein != 'FC':
                results[protein]['c_index_valid_mean_reduced'] = (results[protein]['c_index_valid_STORI_reduced'] + results[protein]['c_index_valid_SPARE_reduced']) / 2

    return results


def bivariate_analysis(stratification_type, penalizer=0., n_bootstraps=500):
    """
    Do the bivariate analysis for each pair of proteins.

    Parameters
    ----------
    stratification_type : str
        Stratification type ('initial', 'before_6months', 'after_6months').
    penalizer : float, optional
        Regularization parameter in the Cox model (L2 penalty).
    n_bootstraps : int, optional
        Number of bootstrap samples used to compute the confidence intervals.

    Returns
    -------
    dict
        Dictionary containing the results of the bivariate analysis for each pair of proteins.

    """

    print(f'Bivariate analysis: {stratification_type} dataset')

    with open(
            f'{DATA_FOLDER}/Dataset_STORI/STORI_data_pkl/STORI_selected_markers_{stratification_type}.pkl',
            'rb') as f:
        data_STORI = pickle.load(f)

    with open(
            f'{DATA_FOLDER}/Dataset_SPARE/SPARE_data_pkl/SPARE_selected_markers_{stratification_type}.pkl',
            'rb') as f:
        data_SPARE = pickle.load(f)

    # Retrieve protein names
    proteins = data_STORI[1]

    results = dict()

    for (protein1, protein2) in itertools.combinations(proteins, 2):

        pair = f'{protein1}-{protein2}'
        results[pair] = dict()
        results[pair][f'n_patients_STORI'] = sample_size(data_STORI, protein1, protein2)
        results[pair][f'n_patients_SPARE'] = sample_size(data_SPARE, protein1, protein2)
        results[pair][f'c_index_dev_STORI_boot'] = np.zeros(n_bootstraps)
        results[pair][f'c_index_dev_SPARE_boot'] = np.zeros(n_bootstraps)
        results[pair][f'c_index_valid_STORI_boot'] = np.zeros(n_bootstraps)
        results[pair][f'c_index_valid_SPARE_boot'] = np.zeros(n_bootstraps)

        if protein1 != 'FC' and protein2 != 'FC':

            results[pair][f'c_index_valid_STORI_reduced_boot'] = np.zeros(n_bootstraps)
            results[pair][f'c_index_valid_SPARE_reduced_boot'] = np.zeros(n_bootstraps)


    for (dataset_dev, dataset_valid) in [('STORI', 'SPARE'), ('SPARE', 'STORI')]:

        print(f'Dev: {dataset_dev}, Valid: {dataset_valid}')

        with open(
                f'{DATA_FOLDER}/Dataset_{dataset_dev}/{dataset_dev}_data_pkl/{dataset_dev}_selected_markers_{stratification_type}.pkl',
                'rb') as f:
            data_dev = pickle.load(f)

        with open(
                f'{DATA_FOLDER}/Dataset_{dataset_valid}/{dataset_valid}_data_pkl/{dataset_valid}_selected_markers_{stratification_type}.pkl',
                'rb') as f:
            data_valid_nonreduced = pickle.load(f)

        with open(
                f'{DATA_FOLDER}/Dataset_{dataset_valid}/{dataset_valid}_data_pkl/{dataset_valid}_selected_markers_reduced_{stratification_type}.pkl',
                'rb') as f:
            data_valid_reduced = pickle.load(f)


        for b in range(n_bootstraps):

            print(f'Bootstrap {b+1}/{n_bootstraps}')

            data_dev_boot = bootstrap_sample(data_dev)

            for (protein1, protein2) in itertools.combinations(proteins, 2):

                pair = f'{protein1}-{protein2}'

                results[pair][f'c_index_dev_{dataset_dev}_boot'][b], _, cph = cox_fit_protein_bivariate(data_dev_boot, protein1, protein2, penalizer)
                results[pair][f'c_index_valid_{dataset_valid}_boot'][b], _ = cox_test_protein_bivariate(data_valid_nonreduced, protein1, protein2, cph)

                if protein1 != 'FC' and protein2 != 'FC':
                    results[pair][f'c_index_valid_{dataset_valid}_reduced_boot'][b], _ = cox_test_protein_bivariate(data_valid_reduced, protein1, protein2, cph)


    for (protein1, protein2) in itertools.combinations(proteins, 2):

        pair = f'{protein1}-{protein2}'

        results[pair][f'c_index_dev_mean_boot'] = (results[pair][f'c_index_dev_STORI_boot'] + results[pair][f'c_index_dev_SPARE_boot']) / 2
        results[pair][f'c_index_valid_mean_boot'] = (results[pair][f'c_index_valid_STORI_boot'] + results[pair][f'c_index_valid_SPARE_boot']) / 2

        results[pair]['c_index_dev_STORI_low'], results[pair]['c_index_dev_STORI'], results[pair]['c_index_dev_STORI_high'] = np.percentile(results[pair][f'c_index_dev_STORI_boot'], [2.5, 50, 97.5])
        results[pair]['c_index_dev_SPARE_low'], results[pair]['c_index_dev_SPARE'], results[pair]['c_index_dev_SPARE_high'] = np.percentile(results[pair][f'c_index_dev_SPARE_boot'], [2.5, 50, 97.5])
        results[pair]['c_index_dev_mean_low'], results[pair]['c_index_dev_mean'], results[pair]['c_index_dev_mean_high'] = np.percentile(results[pair][f'c_index_dev_mean_boot'], [2.5, 50, 97.5])

        results[pair]['c_index_valid_STORI_low'], results[pair]['c_index_valid_STORI'], results[pair]['c_index_valid_STORI_high'] = np.percentile(results[pair][f'c_index_valid_STORI_boot'], [2.5, 50, 97.5])
        results[pair]['c_index_valid_SPARE_low'], results[pair]['c_index_valid_SPARE'], results[pair]['c_index_valid_SPARE_high'] = np.percentile(results[pair][f'c_index_valid_SPARE_boot'], [2.5, 50, 97.5])
        results[pair]['c_index_valid_mean_low'], results[pair]['c_index_valid_mean'], results[pair]['c_index_valid_mean_high'] = np.percentile(results[pair][f'c_index_valid_mean_boot'], [2.5, 50, 97.5])


        if protein1 != 'FC' and protein2 != 'FC':

            results[pair][f'c_index_valid_mean_reduced_boot'] = (results[pair][f'c_index_valid_STORI_reduced_boot'] + results[pair][f'c_index_valid_SPARE_reduced_boot']) / 2

            results[pair]['c_index_valid_STORI_reduced_low'], results[pair]['c_index_valid_STORI_reduced'], results[pair]['c_index_valid_STORI_reduced_high'] = np.percentile(results[pair][f'c_index_valid_STORI_reduced_boot'], [2.5, 50, 97.5])
            results[pair]['c_index_valid_SPARE_reduced_low'], results[pair]['c_index_valid_SPARE_reduced'], results[pair]['c_index_valid_SPARE_reduced_high'] = np.percentile(results[pair][f'c_index_valid_SPARE_reduced_boot'], [2.5, 50, 97.5])
            results[pair]['c_index_valid_mean_reduced_low'], results[pair]['c_index_valid_mean_reduced'], results[pair]['c_index_valid_mean_reduced_high'] = np.percentile(results[pair][f'c_index_valid_mean_reduced_boot'], [2.5, 50, 97.5])

    return results


def run_univariate_analysis(args):


    results_cease = cease_models(args.stratification_type, 'reduced')
    results_univariate = univariate_analysis(args.stratification_type, penalizer=args.penalizer)

    # Add CEASE results to the univariate results
    for model in results_cease:
        results_univariate[model] = results_cease[model]

    # Write results
    results_dir = 'results_univariate'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    write_results_univariate(results_dir=results_dir,
                             output_file=f'{args.stratification_type}_cox_univariate_penalizer_{args.penalizer}.txt',
                             results=results_univariate)


def run_bivariate_analysis(args):
    """
    Run the bivariate analysis and write the results to a file.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    """

    results_dir = 'results_bivariate_%d_bootstraps' % args.n_bootstraps
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_baselines = baselines(args.stratification_type, 'selected_markers_reduced')
    results_cease = cease_models(args.stratification_type, 'selected_markers_reduced')
    results_bivariate = bivariate_analysis(args.stratification_type, penalizer=args.penalizer, n_bootstraps=args.n_bootstraps)

    # Add baselines results to the bivariate results
    for baseline in results_baselines:
        results_bivariate[baseline] = results_baselines[baseline]

    # Add CEASE results to the bivariate results
    for model in results_cease:
        results_bivariate[model] = results_cease[model]

    write_results_bivariate(results_dir,
                            output_file=f'{args.stratification_type}_cox_bivariate_penalizer_{args.penalizer}.txt',
                            results=results_bivariate)


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument('--type', type=str, required=True,
                        help='Analysis type (univariate, bivariate)')
    parser.add_argument('--stratification_type', type=str, default='initial',
                        help='Stratification type (initial, before_6months, after_6months), default: initial')
    parser.add_argument('--penalizer', type=float, default=0.01,
                        help='Cox penalizer, default: 0.01')
    parser.add_argument('--n_bootstraps', type=int, default=500,
                        help='Number of bootstrap samples (only for the bivariate analysis), default: 500')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    args = parse_args()
    if args.type == 'univariate':
        run_univariate_analysis(args)
    elif args.type == 'bivariate':
        run_bivariate_analysis(args)
    else:
        raise ValueError('Unknown analysis type')
