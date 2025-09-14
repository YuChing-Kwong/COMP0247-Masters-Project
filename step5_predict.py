import os
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import cv2
import matplotlib.pyplot as plt
import random
import multiprocessing


# Visualisation function
def visualize_predictions(df, num_per_condition=3):
    conditions = df['condition'].unique()
    for cond in conditions:
        cond_df = df[df['condition'] == cond]
        sample_df = cond_df.sample(n=min(num_per_condition, len(cond_df)), random_state=42)
        for _, row in sample_df.iterrows():
            img_path = row['frame_file']
            true_label = row['true_label']
            pred_label = row['pred_label']

            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(4,4))
            plt.imshow(img_rgb)
            plt.title(f"Condition: {cond}\nTrue: {true_label} | Pred: {pred_label}")
            plt.axis('off')
            plt.show()


def main():
    # Paths
    base_dir = r"D:/UCL R&AI Notes/Masters Project/Recordings"
    participants = ["002", "003", "004", "005", "006", "007", "008", "009", "010"]  # Participants to predict
    model_dir = os.path.join(base_dir, "AutoGluonAllFinal")  # The trained model directory

    # Load model
    predictor = MultiModalPredictor.load(model_dir)
    print(f"Model loaded: {model_dir}")

    for pid in participants:
        participant_dir = os.path.join(base_dir, pid)
        csv_file = os.path.join(participant_dir, f"{pid}_frames_labels.csv")

        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} does not exist, skipping {pid}")
            continue

        df = pd.read_csv(csv_file)
        df['frame_file'] = df['frame_file'].apply(
            lambda x: os.path.join(participant_dir, x) if not os.path.isabs(x) else x
        )

        # Keep true labels
        df['true_label'] = df['label']

        # Only pass the columns needed by the model to avoid overwriting true labels
        df_input = df[['frame_file', 'condition']]

        # Model prediction
        df['pred_label'] = predictor.predict(df_input)

        # Save predictions CSV
        pred_csv = os.path.join(participant_dir, f"{pid}_frames_predictions.csv")
        df.to_csv(pred_csv, index=False, encoding='utf-8')
        print(f"{pid} CSV saved: {pred_csv}")

        # condition-level accuracy
        summary = df.groupby('condition').apply(
            lambda x: (x['pred_label'] == x['true_label']).mean()
        ).reset_index().rename(columns={0: 'accuracy'})
        summary_csv = os.path.join(participant_dir, f"{pid}_condition_accuracy.csv")
        summary.to_csv(summary_csv, index=False, encoding='utf-8')
        print(f"{pid} condition-level accuracy CSV saved: {summary_csv}")

        # Visualise some frames
        visualize_predictions(df, num_per_condition=1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()