import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

# Load the dataset
data = pd.read_csv('UberDataset.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Extract features and target
X = data[['START', 'STOP']]
y = data['PURPOSE']

# Convert categorical variables into numerical representations
vec = DictVectorizer(sparse=False)
X_encoded = vec.fit_transform(X.to_dict(orient='records'))

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_encoded, y)

# Streamlit app
st.title('Trip Purpose Prediction')

# Dropdown menu for selecting start location
start_location = st.selectbox('Select Start Location', sorted(data['START'].unique()))

# Dropdown menu for selecting stop location
stop_location = st.selectbox('Select Stop Location', sorted(data['STOP'].unique()))

# Predict purpose when user selects start and stop locations
if start_location != '' and stop_location != '':
    if st.button('Predict Purpose'):
        input_data = vec.transform([{'START': start_location, 'STOP': stop_location}])
        prediction_probs = clf.predict_proba(input_data)[0]
        # Zip purpose labels and prediction probabilities and sort by probabilities in descending order
        sorted_predictions = sorted(zip(clf.classes_, prediction_probs), key=lambda x: x[1], reverse=True)
        # Filter out predictions with zero probability
        filtered_predictions = [(purpose, probability) for purpose, probability in sorted_predictions if probability > 0.0]
    
        st.write("Predicted purposes (from most to least probable):")
        for purpose, probability in filtered_predictions:
            st.write(f"{purpose}: {probability:.2f}")

else:
    st.error("Please select both start and stop locations.")
