import pickle
import numpy as np

FEATURE_COLUMNS = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary', 'BalanceSalaryRatio', 'TenureAgeRatio', 'CreditScorePerAge'
]

# Model was trained on France=0, Germany=1, Spain=2.
# Other countries map to the nearest regional encoding.
GEOGRAPHY_MAP = {
    # France cluster (Western Europe / default)
    'France': 0, 'Belgium': 0, 'Netherlands': 0, 'Luxembourg': 0,
    'Switzerland': 0, 'Ireland': 0, 'United Kingdom': 0, 'Portugal': 0,
    'United States': 0, 'Canada': 0, 'Australia': 0, 'New Zealand': 0,
    'Japan': 0, 'South Korea': 0, 'Singapore': 0, 'Hong Kong': 0,
    'Brazil': 0, 'Mexico': 0, 'Argentina': 0, 'Chile': 0, 'Colombia': 0,
    'Peru': 0, 'Venezuela': 0, 'Ecuador': 0, 'Uruguay': 0, 'Paraguay': 0,
    'Bolivia': 0, 'Panama': 0, 'Costa Rica': 0, 'Guatemala': 0,
    'Honduras': 0, 'El Salvador': 0, 'Nicaragua': 0, 'Cuba': 0,
    'Dominican Republic': 0, 'Puerto Rico': 0, 'Jamaica': 0, 'Haiti': 0,
    'Morocco': 0, 'Algeria': 0, 'Tunisia': 0, 'Senegal': 0,
    'Ivory Coast': 0, 'Cameroon': 0, 'Madagascar': 0, 'Mali': 0,
    'Burkina Faso': 0, 'Niger': 0, 'Chad': 0, 'Guinea': 0,
    'Lebanon': 0, 'Syria': 0, 'Jordan': 0,
    # Germany cluster (Central / Northern / Eastern Europe)
    'Germany': 1, 'Austria': 1, 'Denmark': 1, 'Sweden': 1, 'Norway': 1,
    'Finland': 1, 'Iceland': 1, 'Poland': 1, 'Czech Republic': 1,
    'Slovakia': 1, 'Hungary': 1, 'Romania': 1, 'Bulgaria': 1,
    'Croatia': 1, 'Slovenia': 1, 'Serbia': 1, 'Bosnia': 1,
    'Montenegro': 1, 'North Macedonia': 1, 'Albania': 1, 'Kosovo': 1,
    'Estonia': 1, 'Latvia': 1, 'Lithuania': 1, 'Belarus': 1,
    'Ukraine': 1, 'Moldova': 1, 'Russia': 1, 'Kazakhstan': 1,
    'Georgia': 1, 'Armenia': 1, 'Azerbaijan': 1, 'Turkey': 1,
    'Israel': 1, 'Iraq': 1, 'Iran': 1, 'Saudi Arabia': 1,
    'UAE': 1, 'Kuwait': 1, 'Qatar': 1, 'Bahrain': 1, 'Oman': 1,
    'Yemen': 1, 'Pakistan': 1, 'Afghanistan': 1, 'Uzbekistan': 1,
    'China': 1, 'Mongolia': 1, 'Taiwan': 1,
    # Spain cluster (Southern Europe / Africa / Asia)
    'Spain': 2, 'Italy': 2, 'Greece': 2, 'Cyprus': 2, 'Malta': 2,
    'India': 2, 'Bangladesh': 2, 'Sri Lanka': 2, 'Nepal': 2,
    'Myanmar': 2, 'Thailand': 2, 'Vietnam': 2, 'Cambodia': 2,
    'Laos': 2, 'Malaysia': 2, 'Indonesia': 2, 'Philippines': 2,
    'Egypt': 2, 'Libya': 2, 'Sudan': 2, 'Ethiopia': 2, 'Kenya': 2,
    'Tanzania': 2, 'Uganda': 2, 'Rwanda': 2, 'Mozambique': 2,
    'Zambia': 2, 'Zimbabwe': 2, 'Malawi': 2, 'Angola': 2,
    'Nigeria': 2, 'Ghana': 2, 'South Africa': 2, 'Botswana': 2,
    'Namibia': 2, 'Congo': 2, 'Somalia': 2, 'Eritrea': 2,
}
GENDER_MAP = {'Female': 0, 'Male': 1}

MODELS = {
    'XGBoost': 'models/xgboost_model.pkl',
    'Random Forest': 'models/random_forest_model.pkl',
    'Decision Tree': 'models/decision_tree_model.pkl',
    'SVM': 'models/svm_model.pkl',
    'KNN': 'models/knn_model.pkl',
}

SCALED_MODELS = {'SVM', 'KNN'}


def load_models():
    loaded = {}
    for name, path in MODELS.items():
        with open(path, 'rb') as f:
            loaded[name] = pickle.load(f)
    return loaded


def load_scaler():
    with open('models/scaler.pkl', 'rb') as f:
        return pickle.load(f)


def preprocess(form_data):
    credit_score = float(form_data['credit_score'])
    geography = GEOGRAPHY_MAP.get(form_data['geography'], 0)
    gender = GENDER_MAP[form_data['gender']]
    age = float(form_data['age'])
    tenure = float(form_data['tenure'])
    balance = float(form_data['balance'])
    num_products = float(form_data['num_products'])
    has_cr_card = float(form_data['has_cr_card'])
    is_active = float(form_data['is_active'])
    salary = float(form_data['salary'])

    balance_salary_ratio = balance / (salary + 1)
    tenure_age_ratio = tenure / (age + 1)
    credit_per_age = credit_score / (age + 1)

    features = np.array([[
        credit_score, geography, gender, age, tenure,
        balance, num_products, has_cr_card, is_active, salary,
        balance_salary_ratio, tenure_age_ratio, credit_per_age
    ]])

    return features


def predict_all(features, models, scaler):
    results = {}
    features_scaled = scaler.transform(features)

    for name, model in models.items():
        X = features_scaled if name in SCALED_MODELS else features
        prob = model.predict_proba(X)[0][1]
        results[name] = round(float(prob), 4)

    avg_prob = round(np.mean(list(results.values())), 4)
    results['Average'] = avg_prob
    return results
