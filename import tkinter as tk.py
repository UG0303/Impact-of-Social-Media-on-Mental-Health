import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SocialMediaMentalHealthPredictor:
    def __init__(self, master):
        self.master = master
        self.master.title("Social Media Mental Health Predictor")

        self.gender_label = tk.Label(master, text="Gender:")
        self.gender_label.pack()
        self.gender_entry = tk.Entry(master)
        self.gender_entry.pack()

        self.age_label = tk.Label(master, text="Age:")
        self.age_label.pack()
        self.age_entry = tk.Entry(master)
        self.age_entry.pack()

        self.occupation_label = tk.Label(master, text="Occupation:")
        self.occupation_label.pack()
        self.occupation_entry = tk.Entry(master)
        self.occupation_entry.pack()

        self.social_media_usage_label = tk.Label(master, text="Social Media Usage (hours):")
        self.social_media_usage_label.pack()
        self.social_media_usage_entry = tk.Entry(master)
        self.social_media_usage_entry.pack()

        self.target_label = tk.Label(master, text="Target Column (Name):")
        self.target_label.pack()
        self.target_entry = tk.Entry(master)
        self.target_entry.pack()

        self.load_data_button = tk.Button(master, text="Load Data", command=self.load_data)
        self.load_data_button.pack()

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.check_illness_button = tk.Button(master, text="Check Illness Type", command=self.check_illness)
        self.check_illness_button.pack()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def predict(self):
        try:
            gender = self.gender_entry.get()
            age = int(self.age_entry.get())
            occupation = self.occupation_entry.get()
            social_media_usage = float(self.social_media_usage_entry.get())
            target_column = self.target_entry.get()  # Target column name

            if not hasattr(self, 'data'):
                messagebox.showerror("Error", "Data not loaded!")
                return

            if 16 <= age <= 25:
                if social_media_usage > 2:
                    prediction_result = "harmful"
                    associated_illness = "Potential mental illnesses associated with harmful social media usage: depression, anxiety, low self-esteem, social isolation."
                    precautions = "Basic precautions:\n1. Limit social media usage.\n2. Engage in offline activities.\n3. Seek professional help if needed."
                else:
                    # Drop target column
                    X = self.data.drop(target_column, axis=1)
                    y = self.data[target_column]

                    # One-hot encode categorical variables
                    X = pd.get_dummies(X)

                    # Impute missing values
                    imputer = SimpleImputer(strategy='mean')
                    X_imputed = imputer.fit_transform(X)

                    # Split data into train and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

                    # Train RandomForestClassifier
                    clf = RandomForestClassifier()
                    clf.fit(X_train, y_train)

                    # Prepare prediction data
                    prediction_data = pd.DataFrame({"Gender": [gender], "Age": [age], "Occupation": [occupation], 
                                                    "Social_Media_Usage": [social_media_usage]})
                    # Perform prediction
                    y_pred = clf.predict(prediction_data)
                    prediction_result = y_pred[0]
                    associated_illness = ""
                    precautions = ""
            elif age > 25:
                prediction_result = "more harmful"
                associated_illness = "Potential mental illnesses associated with excessive social media usage: addiction, depression, anxiety, poor sleep quality."
                precautions = "Basic precautions:\n1. Seek professional help.\n2. Limit social media usage.\n3. Engage in offline activities."
            else:
                prediction_result = "Age not within specified range (16-25)"
                associated_illness = ""
                precautions = ""

            if social_media_usage > 2:
                messagebox.showinfo("Prediction Result", "You are in harmful range.\n\nAssociated illnesses:\n\n{associated_illness}\n\n{precautions}")
            else:
                messagebox.showinfo("Prediction Result", f"Predicted {target_column}: {prediction_result}\n\nAssociated illnesses:\n{associated_illness}\n\n{precautions}")
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")

    def check_illness(self):
        # Create a new window for checking illness type
        if not hasattr(self, 'data'):
            messagebox.showerror("Error", "Data not loaded!")
            return
        
        self.illness_window = tk.Toplevel(self.master)
        self.illness_window.title("Check Illness Type")

        self.question1_label = tk.Label(self.illness_window, text="What is the average time you spend on social media every day?")
        self.question1_label.pack()
        self.question1_entry = tk.Entry(self.illness_window)
        self.question1_entry.pack()

        self.question2_label = tk.Label(self.illness_window, text="How often do you find yourself using Social media without a specific purpose?")
        self.question2_label.pack()
        self.question2_entry = tk.Entry(self.illness_window)
        self.question2_entry.pack()

        self.question3_label = tk.Label(self.illness_window, text="On a scale of 1 to 5, how easily distracted are you?")
        self.question3_label.pack()
        self.question3_entry = tk.Entry(self.illness_window)
        self.question3_entry.pack()

        self.question4_label = tk.Label(self.illness_window, text="On a scale of 1 to 5, how often do you face issues regarding sleep?")
        self.question4_label.pack()
        self.question4_entry = tk.Entry(self.illness_window)
        self.question4_entry.pack()

        self.check_button = tk.Button(self.illness_window, text="Check", command=self.check_illness_type)
        self.check_button.pack()

    def check_illness_type(self):
        # Get answers to questions
        answer1 = float(self.question1_entry.get())
        answer2 = self.question2_entry.get().lower()
        answer3 = int(self.question3_entry.get())
        answer4 = int(self.question4_entry.get())

        # Determine potential illness based on answers
        if answer1 > 2 or answer3 > 3 or answer4 > 3:
            associated_illness = "Potential mental illnesses associated with harmful social media usage: depression, anxiety, low self-esteem, social isolation."
            precautions = "Basic precautions:\n1. Limit social media usage.\n2. Engage in offline activities.\n3. Seek professional help if needed."
            messagebox.showinfo("Illness Type", f"You are in harmful range.\n\nAssociated illnesses:\n\n{associated_illness}\n\n{precautions}")

            # Plot bar graph
            labels = ['Social Media Usage', 'Distraction Level', 'Sleep Issues']
            values = [answer1, answer3, answer4]
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.bar(labels, values)
            plt.xlabel('Factors')
            plt.ylabel('Scores')
            plt.title('Bar Graph of Mental Health Factors')

            # Plot line graph
            plt.subplot(1, 2, 2)
            plt.plot(labels, values, marker='o', linestyle='-')
            plt.xlabel('Factors')
            plt.ylabel('Scores')
            plt.title('Line Graph of Mental Health Factors')

            plt.tight_layout()
            plt.show()
        elif "often" in answer2 or "frequently" in answer2:
            messagebox.showinfo("Illness Type", "You may be experiencing symptoms of depression or anxiety. It is recommended to seek professional help.")
        else:
            messagebox.showinfo("Illness Type", "Your responses do not indicate a high likelihood of mental illness. However, if you have concerns, consider consulting a healthcare professional.")

def main():
    root = tk.Tk()
    app = SocialMediaMentalHealthPredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
