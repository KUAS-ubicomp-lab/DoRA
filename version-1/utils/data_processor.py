from datasets import load_dataset
from openprompt.data_utils.utils import InputExample


class DEPRESSION:
    """
    paper: https://doi.org/10.1057/s41599-022-01313-2
    """
    labels = ["control", "depression"]
    label_words = {
        "control": ["happy", "elation", "joy", "happiness", "cheerfulness", "contentment", "ecstasy", "delight",
                    "euphoria",
                    "bliss", "satisfaction", "well-being", "optimism", "hopefulness", "positivity", "high spirits",
                    "enthusiasm",
                    "excitement", "thrill", "jubilation"],
        "depression": ["sadness", "anger", "bored", "hurt", "nervousness", "moody", "emotional", "discomfort",
                       "unhappy",
                       "no focus", "disturbed", "loss of appetite", "overeating", "weight loss", "light sleep",
                       "waking up early",
                       "short sleep", "dreamy", "bored", "useless", "isolation", "low interaction", "fatigue",
                       "uneasiness",
                       "instability", "stress", "pain", "useless", "hopeless", "helpless", "lonely", "low self-esteem",
                       "hesitate",
                       "tired", "long-term grief", "hopelessness", "end-of-my-life", "die", "suicide", "kill myself",
                       "suicidal",
                       "broken", "worthless", "self-harm"]
    }
    label_mapping = {"control": "control", "depression": "depression"}

    def __init__(self):
        data = load_dataset('depression')
        self.train_dataset = [
            InputExample(guid=i, text_a=e['premise'], text_b=e['hypothesis'], label=self.label_mapping[e['label']])
            for i, e in enumerate(data['train']) if e['label'] != -1]

        self.eval_dataset = [
            InputExample(guid=i, text_a=e['premise'], text_b=e['hypothesis'], label=self.label_mapping[e['label']])
            for i, e in enumerate(data['validation']) if e['label'] != -1]


data_processor_list = {
    'depression': DEPRESSION
}
