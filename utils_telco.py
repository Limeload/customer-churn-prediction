import pickle
import numpy as np

TELCO_MODELS = {
    'XGBoost':           'models/telco/xgboost.pkl',
    'Random Forest':     'models/telco/random_forest.pkl',
    'Gradient Boosting': 'models/telco/gradient_boosting.pkl',
    'Stacking':          'models/telco/stacking.pkl',
    'Voting Ensemble':   'models/telco/voting.pkl',
}
TELCO_SCALED = set()  # scaler applied to all features for telco


def load_telco_models():
    loaded = {}
    for name, path in TELCO_MODELS.items():
        with open(path, 'rb') as f:
            loaded[name] = pickle.load(f)
    return loaded


def load_telco_scaler():
    with open('models/telco/scaler.pkl', 'rb') as f:
        return pickle.load(f)


# Feature order must exactly match training (get_dummies drop_first=True)
FEATURE_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'PaperlessBilling',
    'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'ChargesPerTenure', 'TotalToMonthlyRatio', 'ServiceCount',
]


def _one_hot(prefix, value, options):
    """Return prefixed dummy columns (drop_first: skip options[0])."""
    return {f'{prefix}_{opt}': int(value == opt) for opt in options[1:]}


def preprocess_telco(f):
    monthly  = float(f.get('monthly_charges', 0))
    total    = float(f.get('total_charges', 0))
    tenure   = float(f.get('tenure', 0))
    internet = f.get('internet_service', 'DSL')   # DSL | Fiber optic | No

    # Service add-ons are only meaningful when internet is not 'No'
    def svc(key):
        val = f.get(key, 'No')
        if internet == 'No':
            return 'No internet service'
        return val  # Yes | No

    # MultipleLines depends on PhoneService
    phone = f.get('phone_service', 'No')
    multi = f.get('multiple_lines', 'No')
    if phone == 'No':
        multi_val = 'No phone service'
    else:
        multi_val = multi  # Yes | No

    # Encoded add-on values for ServiceCount
    addon_keys = ['online_security', 'online_backup', 'device_protection',
                  'tech_support', 'streaming_tv', 'streaming_movies']
    service_count = sum(1 for k in addon_keys if svc(k) == 'Yes')

    row = {
        'gender':            int(f.get('gender', 'Male') == 'Male'),
        'SeniorCitizen':     int(f.get('senior_citizen', '0') == '1'),
        'Partner':           int(f.get('partner', 'No') == 'Yes'),
        'Dependents':        int(f.get('dependents', 'No') == 'Yes'),
        'tenure':            tenure,
        'PhoneService':      int(phone == 'Yes'),
        'PaperlessBilling':  int(f.get('paperless_billing', 'No') == 'Yes'),
        'MonthlyCharges':    monthly,
        'TotalCharges':      total,
        # MultipleLines dummies (base = 'No')
        **_one_hot('MultipleLines', multi_val, ['No', 'No phone service', 'Yes']),
        # InternetService dummies (base = 'DSL')
        **_one_hot('InternetService', internet, ['DSL', 'Fiber optic', 'No']),
        # Add-on dummies (base = 'No')
        **_one_hot('OnlineSecurity',   svc('online_security'),   ['No', 'No internet service', 'Yes']),
        **_one_hot('OnlineBackup',     svc('online_backup'),     ['No', 'No internet service', 'Yes']),
        **_one_hot('DeviceProtection', svc('device_protection'), ['No', 'No internet service', 'Yes']),
        **_one_hot('TechSupport',      svc('tech_support'),      ['No', 'No internet service', 'Yes']),
        **_one_hot('StreamingTV',      svc('streaming_tv'),      ['No', 'No internet service', 'Yes']),
        **_one_hot('StreamingMovies',  svc('streaming_movies'),  ['No', 'No internet service', 'Yes']),
        # Contract dummies (base = 'Month-to-month')
        **_one_hot('Contract', f.get('contract', 'Month-to-month'),
                   ['Month-to-month', 'One year', 'Two year']),
        # PaymentMethod dummies (base = 'Bank transfer (automatic)')
        **_one_hot('PaymentMethod', f.get('payment_method', 'Bank transfer (automatic)'),
                   ['Bank transfer (automatic)', 'Credit card (automatic)',
                    'Electronic check', 'Mailed check']),
        # Engineered features
        'ChargesPerTenure':    monthly / (tenure + 1),
        'TotalToMonthlyRatio': total   / (monthly + 1),
        'ServiceCount':        service_count,
    }

    return np.array([[row[col] for col in FEATURE_COLS]])


def predict_telco(features, models, scaler):
    features_scaled = scaler.transform(features)
    results = {}
    for name, model in models.items():
        prob = model.predict_proba(features_scaled)[0][1]
        results[name] = round(float(prob), 4)
    import numpy as np
    results['Average'] = round(float(np.mean(list(results.values()))), 4)
    return results
