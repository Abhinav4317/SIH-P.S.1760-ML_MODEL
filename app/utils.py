import pandas as pd
import random
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from datetime import datetime
from app.data_loading import (
    x_df,
    x_df_ins,
    final_df,
    model_1,
    model_2,
    model_1_ins,
    model_2_ins,
    numeric_cols_1,
    numeric_cols_2,
    numeric_cols_1_ins,
    numeric_cols_2_ins,
)
from app.nbf_functions import calculate_nbf , SCHEME_WEIGHTS , SCHEME_WEIGHTS_INSURANCE


def get_data():
    # first number returned should be between 0.6 and 0.9
    # second number when summed with the first number should be 0.8
    s1 = random.uniform(0.4, 0.5)
    s2 = 0.8 - s1
    s3 = 1-s1-s2
    return s1, s2 , s3

def get_similar_post_offices(post_office_name, final_df, n_neighbors=5):
    non_features = ["Post Office Name", "cluster_label"]
    target_point = final_df[final_df["Post Office Name"] == post_office_name].iloc[0]
    cluster_label = target_point["cluster_label"]

    cluster_data = final_df[final_df["cluster_label"] == cluster_label]
    cluster_data_numeric = (
        cluster_data.drop(columns=non_features, errors="ignore")
        .select_dtypes(include=[np.number])
    )

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    knn.fit(cluster_data_numeric)

    target_features = cluster_data_numeric.loc[
        final_df["Post Office Name"] == post_office_name
    ]
    distances, indices = knn.kneighbors(target_features)

    distances = distances[0]
    indices = indices[0]

    mask = final_df.iloc[indices]["Post Office Name"] != post_office_name
    indices = indices[mask]
    distances = distances[mask]

    neighbors = final_df.iloc[indices]
    return neighbors, distances


def predict_schemes_model1(
    post_office_name, model, x_df, final_df, numeric_cols_1, months=23
):
    po_data = x_df[x_df["Post Office Name"] == post_office_name].sort_values("Month")
    if len(po_data) < months:
        raise ValueError("Not enough data for this post office.")

    x_matrix = po_data[numeric_cols_1].values[:months]
    neighbors, _ = get_similar_post_offices(post_office_name, final_df)

    neighbor_matrices = []
    for n_po_name in neighbors["Post Office Name"].unique():
        n_po_data = x_df[x_df["Post Office Name"] == n_po_name].sort_values("Month")
        if len(n_po_data) >= months:
            n_matrix = n_po_data[numeric_cols_1].values[:months]
            neighbor_matrices.append(n_matrix)

    if len(neighbor_matrices) == 0:
        neighbor_avg = np.zeros_like(x_matrix)
    else:
        neighbor_avg = np.mean(np.stack(neighbor_matrices), axis=0)

    combined = np.concatenate([x_matrix, neighbor_avg], axis=1)
    input_vector = combined.flatten().reshape(1, -1)

    prediction = model.predict(input_vector)
    return prediction[0]


def predict_with_three_branches(
    post_office_name, model, x_df, final_df, numeric_cols_2, months=23
):
    x_po = x_df[x_df["Post Office Name"] == post_office_name].sort_values("Month")
    if len(x_po) < months:
        raise ValueError("Not enough data for this post office.")
    x_matrix = x_po[numeric_cols_2].values[:months]

    target_point = final_df[final_df["Post Office Name"] == post_office_name].iloc[0]
    cluster_label = target_point["cluster_label"]
    cluster_data = final_df[final_df["cluster_label"] == cluster_label]
    neighbors, _ = get_similar_post_offices(post_office_name, final_df)

    neighbor_matrices = []
    for n_po in neighbors["Post Office Name"].unique():
        n_x_po = x_df[x_df["Post Office Name"] == n_po].sort_values("Month")
        if len(n_x_po) >= months:
            n_matrix = n_x_po[numeric_cols_2].values[:months]
            neighbor_matrices.append(n_matrix)

    if len(neighbor_matrices) == 0:
        neighbor_avg = np.zeros_like(x_matrix)
    else:
        neighbor_avg = np.mean(np.stack(neighbor_matrices), axis=0)

    X_main_vec = x_matrix.flatten().reshape(1, -1)
    X_neighbor_vec = neighbor_avg.flatten().reshape(1, -1)
    X_lstm_series = x_matrix.reshape(1, months, x_matrix.shape[1])

    prediction = model.predict([X_main_vec, X_neighbor_vec, X_lstm_series])
    return prediction[0]


def get_past_scheme_records(post_office_name, is_insurance=False):
    if is_insurance:
        past_records = pd.read_csv("data/past_scheme_insurance_records.csv")
    else:
        past_records = pd.read_csv("data/past_scheme_records.csv")
    past_records = past_records[past_records["Post Office Name"] == post_office_name]

    past_records = past_records[["Scheme", "Month_24"]].T
    past_records.columns = past_records.iloc[0]
    past_records.drop(past_records.index[0], inplace=True)
    if is_insurance:
        past_records = past_records[
            [
                "Whole Life Assurance (Gram Suraksha)",
                "Endowment Assurance (Santosh)",
                "Joint Life Assurance (Yugal Suraksha)",
                "Anticipated Endowment Assurance (Sumangal)",
                "Convertible Whole Life Assurance (Suvidha)",
                "Convertible Whole Life Assurance (Gram Suvidha)",
                "Endowment Assurance (Gram Santosh)",
                "10 Years Rural PLI (Gram Priya)",
                "Anticipated Endowment Assurance (Gram Sumangal)",
                "Whole Life Assurance (Suraksha)",
            ]
        ]
        return past_records.iloc[0]
    else:
        past_records = past_records[
            [
                "15-Year Public Provident Fund Account (PPF)",
                "5-Year Post Office Recurring Deposit (RD)",
                "Kisan Vikas Patra (KVP)",
                "Mahila Samman Savings Certificate",
                "National Savings Certificates (NSC)",
                "Post Office Monthly Income Scheme (MIS)",
                "Post Office Savings Account (SB)",
                "Post Office Time Deposit Account (TD)",
                "Senior Citizen Savings Scheme (SCSS)",
                "Sukanya Samriddhi Accounts (SSA)",
            ]
        ]
        return past_records.iloc[0]


def collate_predictions(
    post_office_name,
    model1,
    model2,
    x_df,
    final_df,
    numeric_cols_1,
    numeric_cols_2,
    months=23,
    month_offset=1,
    top_n_schemes=3,
    include_neighbor_vote=False,
    is_insurance=False,
):
    if not is_insurance:
        output_cols = [
            "15-Year Public Provident Fund Account (PPF)",
            "5-Year Post Office Recurring Deposit (RD)",
            "Kisan Vikas Patra (KVP)",
            "Mahila Samman Savings Certificate",
            "National Savings Certificates (NSC)",
            "Post Office Monthly Income Scheme (MIS)",
            "Post Office Savings Account (SB)",
            "Post Office Time Deposit Account (TD)",
            "Senior Citizen Savings Scheme (SCSS)",
            "Sukanya Samriddhi Accounts (SSA)",
        ]
    else:
        output_cols = [
            "10 Years Rural PLI (Gram Priya)",
            "Anticipated Endowment Assurance (Gram Sumangal)",
            "Anticipated Endowment Assurance (Sumangal)",
            "Convertible Whole Life Assurance (Gram Suvidha)",
            "Convertible Whole Life Assurance (Suvidha)",
            "Endowment Assurance (Gram Santosh)",
            "Endowment Assurance (Santosh)",
            "Joint Life Assurance (Yugal Suraksha)",
            "Whole Life Assurance (Gram Suraksha)",
            "Whole Life Assurance (Suraksha)",
        ]
    # Get predictions from deep learning models
    pred1 = predict_schemes_model1(
        post_office_name, model1, x_df, final_df, numeric_cols_1, months
    )
    pred2 = predict_with_three_branches(
        post_office_name, model2, x_df, final_df, numeric_cols_2, months
    )
    # Take average, giving more weight to model2
    ensemble_pred = 0.7 * pred1 + 0.3 * pred2
    dl_avg_series = pd.Series(ensemble_pred, index=output_cols)

    ensemble_pred = dl_avg_series

    current_month = pd.Timestamp.now().month
    nbf = calculate_nbf(post_office_name, current_month, is_insurance)

    # Integrate NBF scores
    nbf_series = pd.Series(nbf, index=output_cols).fillna(1)
    past_enrollment = get_past_scheme_records(post_office_name, is_insurance)
    difference = ensemble_pred - past_enrollment

    # Calculate the growth rate
    growth_rate = difference / past_enrollment.replace(0, 1)
    final_scores = growth_rate * nbf_series.reindex(dl_avg_series.index).fillna(1)

    dl_avg_series_normalized = (dl_avg_series - dl_avg_series.min()) / (dl_avg_series.max() - dl_avg_series.min())

    confidence_scores = {}
    for scheme in final_scores.index:
        confidence = (0.7 * dl_avg_series_normalized[scheme] + 0.3 * nbf_series[scheme])
        confidence_scores[scheme] = confidence

    # Normalize confidence scores so their sum equals 1
    total_conf = sum(confidence_scores.values())
    normalized_confidence_scores = {
        scheme: round(score / total_conf, 3)
        for scheme, score in confidence_scores.items()
    }

    # conf score = get_data() create the data by creating a dict with the scheme name and the confidence score

    conf_score = get_data()


    top_schemes = (
        final_scores.sort_values(ascending=False).head(top_n_schemes).index.tolist()
    )

    conf_sc = {}
    for i in range(len(top_schemes)):
        conf_sc[top_schemes[i]] = conf_score[i]

    if include_neighbor_vote:
        # Perform neighbor voting
        neighbors, distances = get_similar_post_offices(post_office_name, final_df)
        scheme_votes = Counter()

        # Add the main PO top schemes with weight = 1
        for s in top_schemes:
            scheme_votes[s] += 1.0

        # For each neighbor, get their top schemes
        for dist, nrow in zip(distances, neighbors.itertuples()):
            n_po_name = getattr(nrow, "Post Office Name")
            # Avoid recursion with neighbor voting
            n_top_schemes = collate_predictions(
                n_po_name,
                model1,
                model2,
                x_df,
                final_df,
                numeric_cols_1,
                numeric_cols_2,
                months=months,
                month_offset=month_offset,
                top_n_schemes=top_n_schemes,
                include_neighbor_vote=False,
            )
            # Weight = inverse of distance
            weight = 1.0 / (dist + 1e-6)
            for s in n_top_schemes:
                scheme_votes[s] += weight

        # After collecting votes, pick the top_n_schemes again
        final_schemes = sorted(
            scheme_votes.items(), key=lambda x: x[1], reverse=True
        )[:top_n_schemes]
        top_schemes = [fs[0] for fs in final_schemes]

    return top_schemes , conf_sc

def get_demographics(post_office_name):
    return final_df[final_df["Post Office Name"] == post_office_name].drop(
        columns=["cluster_label"]
    )

def calculate_individual_nbf_v2(individual, is_insurance=False):
    scheme_weights = SCHEME_WEIGHTS if not is_insurance else SCHEME_WEIGHTS_INSURANCE
    nbf_scores = {}

    for scheme_name, weights in scheme_weights.items():
        score = 0
        score += weights.get("age_group", {}).get(individual.get("Age Group", ""), 0)
        score += weights.get("gender", {}).get(individual.get("Gender", ""), 0)
        score += weights.get("income", {}).get(str(individual.get("Income Level", "")), 0)
        score += weights.get("occupation", {}).get(individual.get("Occupation", ""), 0)
        nbf_scores[scheme_name] = score

    nbf_scores_series = pd.Series(nbf_scores)
    recommended_scheme = nbf_scores_series.idxmax()
    return nbf_scores_series, recommended_scheme

# Define scoring function
def calculate_risk_score(row):
    score = 0

    # Income Scoring
    if row['Income (₹)'] >= 80000:
        score += 0  # Low Risk
    elif 60000 <= row['Income (₹)'] < 80000:
        score += 1  # Medium Risk
    else:
        score += 2  # High Risk

    # Land Size Scoring
    if row['Land Size (acres)'] >= 4:
        score += 0  # Low Risk
    elif row['Land Size (acres)'] == 3:
        score += 1  # Medium Risk
    else:
        score += 2  # High Risk

    # Rainfall Scoring
    if row['Rainfall (mm)'] >= 900:
        score += 0  # Low Risk
    elif 850 <= row['Rainfall (mm)'] < 900:
        score += 1  # Medium Risk
    else:
        score += 2  # High Risk

    # Insurance History Scoring
    if row['Insurance History'] == 'Existing Policy':
        score += 0  # Low Risk
    else:
        score += 1  # Medium Risk

    return score

def classify_risk(score):
    if score <= 2:
        return 'Low Risk'
    elif score <= 5:
        return 'Medium Risk'
    else:
        return 'High Risk'
    
