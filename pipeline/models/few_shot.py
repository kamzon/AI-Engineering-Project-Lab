import os
import glob
import shutil
import numpy as np
from PIL import Image as PILImage
from sklearn.metrics import accuracy_score
from datasets import Dataset, Image as HFImage
from transformers import (
    AutoImageProcessor,
    ResNetForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

class FewShotResNet:
    """
    A class to fine-tune a ResNet model for a binary classification task
    (e.g., 'target_object' vs. 'others') from a list of image paths.
    """
    def __init__(
        self,
        object_type: str,
        model_checkpoint: str = "microsoft/resnet-50",
        final_model_path: str = "pipeline/finetuned",
    ):
        """
        Initializes the fine-tuning pipeline.

        Args:
            object_type (str): The name of the target class. This will be used to
                               identify target images from their parent directory.
            model_checkpoint (str): The name of the pre-trained model from Hugging Face Hub.
            final_model_path (str): The path where the final fine-tuned model will be saved.
        """
        if not object_type or not isinstance(object_type, str):
            raise ValueError("`object_type` must be a non-empty string.")

        self.object_type = object_type
        self.labels = ["others", self.object_type] # Class 0 is 'others', Class 1 is the target
        self.model_checkpoint = model_checkpoint
        self.final_model_path = final_model_path
        self.output_dir = "./checkpoints_resnet50"

        # Initialize attributes
        self.image_processor = None
        self.model = None
        self.train_ds = None
        self.eval_ds = None

        # Create label mappings
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        print(f"✅ FewShotResNet initialized for target: '{self.object_type}'")

    def load_data_from_paths(self, image_paths: list, test_size: float = 0.2):
        """
        Loads images from a list of paths and labels them based on their
        parent directory. It then splits the data into training and evaluation sets.

        Args:
            image_paths (list): A list of strings, where each string is a path to an image.
            test_size (float): The proportion of the dataset to reserve for evaluation.
        """
        print(f"🛠️ Loading data from {len(image_paths)} image paths...")
        
        labels = []
        for path in image_paths:
            # Extract the parent directory name from the path
            parent_dir = os.path.basename(os.path.dirname(path))
            # Assign label 1 if the directory matches the object_type, else 0
            label = self.label2id[self.object_type] if parent_dir == self.object_type else self.label2id["others"]
            labels.append(label)

        # Create a Hugging Face Dataset
        dataset = Dataset.from_dict({"image": image_paths, "label": labels}).cast_column("image", HFImage())
        
        # Split into training and evaluation sets
        split_dataset = dataset.train_test_split(test_size=test_size)
        self.train_ds = split_dataset["train"]
        self.eval_ds = split_dataset["test"]
        
        print(f"✅ Data loaded and split. Training samples: {len(self.train_ds)}, Evaluation samples: {len(self.eval_ds)}")


    def _load_model_and_processor(self):
        """Loads the pre-trained model and its associated image processor."""
        print(f"🛠️ Loading pre-trained model: {self.model_checkpoint}")
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_checkpoint)
        self.model = ResNetForImageClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

    def _transform_data(self, example_batch):
        """Applies the image processor to a batch of examples."""
        inputs = self.image_processor(example_batch['image'], return_tensors='pt')
        inputs['label'] = example_batch['label']
        return inputs

    @staticmethod
    def _compute_metrics(eval_pred):
        """Computes accuracy metric for evaluation."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    def finetune(self, num_train_epochs: int = 3, per_device_batch_size: int = 8):
        """
        Executes the full fine-tuning pipeline.
        """
        if not self.train_ds or not self.eval_ds:
            raise ValueError("Data not loaded. Please call `load_data_from_paths()` first.")
        
        self._load_model_and_processor()
        
        self.train_ds.set_transform(self._transform_data)
        self.eval_ds.set_transform(self._transform_data)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            tokenizer=self.image_processor,
            compute_metrics=self._compute_metrics,
            data_collator=DefaultDataCollator(),
        )

        print("🚀 Starting fine-tuning...")
        trainer.train()

        print(f"✅ Training complete. Saving the best model to {self.final_model_path}")
        os.makedirs(self.final_model_path, exist_ok=True)
        trainer.save_model(self.final_model_path)
        print("✨ Model saved successfully!")
