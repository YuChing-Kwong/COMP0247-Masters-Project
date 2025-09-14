import os
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import shutil
import multiprocessing
import matplotlib.pyplot as plt
import cv2

# Paths
base_dir = r"./Recordings"
participants_train = ["001", "002", "003", "004", "005", "006", "007", "008"]
# participants_tune = ["008"]
participants_eval = ["009", "010"]

temp_model_dir = os.path.join(base_dir, "AutoGluonTemp")
final_model_dir = os.path.join(base_dir, "AutoGluonAllFinal")


# Load CSV files and combine
def load_participants(participant_list):
    all_frames = []
    for pid in participant_list:
        csv_file = os.path.join(base_dir, pid, f"{pid}_frames_labels.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            # Only keep necessary columns and fix paths
            # df['frame_file'] =   "./Recordings/" + df['frame_file'].apply(lambda x: x.replace("\\", "/"))
            df['frame_file'] = df['frame_file'].apply(lambda x: x.replace("\\", "/"))
            df = df[['frame_file', 'label']]

            all_frames.append(df)
        else:
            print(f"Warning: {csv_file} does not exist", flush=True)

    if not all_frames:
        return None
    return pd.concat(all_frames, ignore_index=True)


def main():
    # Load data
    train_data = load_participants(participants_train)
    eval_data = load_participants(participants_eval)

    if train_data is None or eval_data is None:
        print("Some dataset resulted being empty", flush=True)
        return

    # Save CSVs for checking
    train_data.to_csv(os.path.join(base_dir, "train_frames_check.csv"), index=False)
    eval_data.to_csv(os.path.join(base_dir, "eval_frames_check.csv"), index=False)

    # Delete temp_model_dir
    if os.path.exists(temp_model_dir):
        print("Deleting old temporary training folder...", flush=True)
        shutil.rmtree(temp_model_dir, ignore_errors=True)
    os.makedirs(temp_model_dir, exist_ok=True)

    orig_cwd = os.getcwd()
    # os.chdir(temp_model_dir)
    os.makedirs(temp_model_dir, exist_ok=True)

    # Initialize predictor
    predictor = MultiModalPredictor(
        label='label',
        problem_type='multiclass',
        eval_metric='accuracy'
    )
    print(train_data.head())

    # Health check: paths exist & labels are trainable
    def _all_exist(rel_paths):
        return rel_paths.apply(lambda rp: os.path.exists(os.path.join(os.getcwd(), rp))).all()

    assert train_data['label'].nunique() >= 2
    assert _all_exist(train_data['frame_file'])
    assert _all_exist(eval_data['frame_file'])


    predictor.fit(
        train_data=train_data,
        time_limit=120,
        presets='best_quality',
    )

        # Evaluate
    scores = predictor.evaluate(data = eval_data)
    scores 

    # Visualise
    sample_df = train_data.sample(n=5, random_state=42)
    for _, row in sample_df.iterrows():
        img_path = row['frame_file']
        label = row['label']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

    os.chdir(orig_cwd)
    predictor.save(final_model_dir)
    print(f"Final model has been saved to: {final_model_dir}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
