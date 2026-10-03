import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")

# --- Load Data and Model ---
@st.cache_resource
def load_data_model():
    df = pd.read_csv('creditcard.csv')

    if not os.path.exists('fraud_detection_model.pkl'):
        X = df.drop('Class', axis=1)
        y = df['Class']
        features = X.columns.tolist()

        scaler = StandardScaler()
        X[['Amount','Time']] = scaler.fit_transform(X[['Amount','Time']])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        pickle.dump(model, open('fraud_detection_model.pkl','wb'))
        pickle.dump(scaler, open('fraud_scaler.pkl','wb'))
        pickle.dump(features, open('feature_names.pkl','wb'))
    else:
        model = pickle.load(open('fraud_detection_model.pkl','rb'))
        scaler = pickle.load(open('fraud_scaler.pkl','rb'))
        features = pickle.load(open('feature_names.pkl','rb'))

    return df, model, scaler, features

df, model, scaler, feature_names = load_data_model()

# --- Dashboard ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", len(df))
c2.metric("Fraud Cases", len(df[df['Class']==1]))
c3.metric("Normal Cases", len(df[df['Class']==0]))
c4.metric("Fraud %", f"{len(df[df['Class']==1])/len(df)*100:.2f}%")

st.divider()

# --- Live Prediction ---
st.subheader("🔍 Live Fraud Check")

tab1, tab2 = st.tabs(["Random Check", "Manual Check"])

with tab1:
    if st.button("Check Random Transaction", type="primary"):
        sample = df.sample(1)
        sample_features = sample[feature_names].copy()
        sample_features[['Amount','Time']] = scaler.transform(sample_features[['Amount','Time']])

        pred = model.predict(sample_features)[0]
        prob = model.predict_proba(sample_features)[0][1]

        st.dataframe(sample)

        if pred == 1:
            st.error(f"🚨 FRAUD ALERT! Probability: {prob*100:.2f}%")
        else:
            st.success(f"✅ NORMAL Transaction. Fraud Probability: {prob*100:.2f}%")

with tab2:
    amount = st.number_input("Enter Transaction Amount", value=100.0)
    if st.button("Check Amount Based"):
        avg_values = df[feature_names].mean().to_dict()
        avg_values['Amount'] = amount
        avg_values['Time'] = 50000
        test_df = pd.DataFrame([avg_values])[feature_names]
        test_df[['Amount','Time']] = scaler.transform(test_df[['Amount','Time']])
        pred = model.predict(test_df)[0]
        if pred == 1:
            st.error("🚨 This amount looks like FRAUD pattern")
        else:
            st.success("✅ This amount looks NORMAL")

st.divider()

# --- Graphs ---
st.subheader("📊 Graphs")
col1, col2 = st.columns(2)

with col1:
    st.write("**Class Distribution**")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    ax.set_xticklabels(['Normal (0)', 'Fraud (1)'])
    st.pyplot(fig)

with col2:
    st.write("**Confusion Matrix**")
    if os.path.exists('confusion_matrix.png'):
        st.image('confusion_matrix.png')
    else:
        X = df.drop('Class', axis=1)
        y = df['Class']
        X[['Amount','Time']] = scaler.transform(X[['Amount','Time']])
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

st.dataframe(df.head())