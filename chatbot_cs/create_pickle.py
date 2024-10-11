import pickle

# Sample data for demonstration
words = ["hello", "world"]
labels = ["greeting"]
training = [[1, 0], [0, 1]]  # Example training data
output = [1, 0]  # Example output

# Save data to a new pickle file
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

print("Pickle file created successfully.")
