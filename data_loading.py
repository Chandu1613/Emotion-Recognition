import os
import pandas as pd


class EmotionDatasetPreparer:
    def __init__(self, datasets: dict):
        """
        Args:
            datasets (dict): Mapping of dataset_name -> dataset_path
        """
        self.datasets = datasets

        # Emotion label maps for standardization
        self.ravdess_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        self.crema_map = {
            'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
            'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
        }
        self.emodb_map = {
            'W': 'angry', 'L': 'bored', 'E': 'disgust',
            'A': 'fear', 'F': 'happy', 'T': 'sad', 'N': 'neutral'
        }

    def generate_csvs(self):
        """Generate individual DataFrames for each dataset."""
        dfs = {}

        for dataset_name, dataset_path in self.datasets.items():
            data = []
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if not file.endswith(".wav"):
                        continue
                    filepath = os.path.join(root, file)

                    # RAVDESS
                    if dataset_name == "RAVDESS":
                        parts = file.split('-')
                        if len(parts) == 7:
                            emotion = self.ravdess_map.get(parts[2], "unknown")

                    # CREMA-D
                    elif dataset_name == "CREMA-D":
                        parts = file.split('_')
                        emotion = self.crema_map.get(parts[2], "unknown") if len(parts) >= 3 else "unknown"

                    # TESS
                    elif dataset_name == "TESS":
                        emotion = os.path.basename(root).lower()

                    # EmoDB
                    elif dataset_name == "EmoDB":
                        emotion = self.emodb_map.get(file[5], "unknown")

                    # IESC
                    elif dataset_name == "IESC":
                        emotion = os.path.basename(root).lower()

                    else:
                        continue

                    data.append([filepath, emotion])

            if data:
                dfs[dataset_name] = pd.DataFrame(data, columns=['file_path', 'emotion'])

        return dfs

    def merge_and_clean(self, dfs: dict):
        """Merge all DataFrames and clean emotion labels."""
        df = pd.concat(dfs.values(), ignore_index=True)

        # Remove prefixes like oaf_ or yaf_
        df['emotion'] = df['emotion'].str.replace(r'^(oaf_|yaf_)', '', regex=True)

        # Normalize emotion labels
        df['emotion'] = df['emotion'].replace({
            'pleasant_surprised': 'surprised',
            'pleasant_surprise': 'surprised',
            'surprise': 'surprised',
            'calm': 'neutral',
            'bored': 'neutral',
            'fearful': 'fear',
            'anger': 'angry'
        })

        return df