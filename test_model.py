import pickle

# Load trained model
with open("model/crop_model.pkl", "rb") as file:
    model = pickle.load(file)

# Sample input
sample_data = [[90, 42, 43, 20, 82, 6.5, 220]]


prediction = model.predict(sample_data)

print("Recommended Crop:", prediction[0])
