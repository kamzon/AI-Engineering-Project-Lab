import os
import glob
import shutil
import io
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

from pipeline.config import ModelConstants

class FewShotResNet:
    """
    A class to fine-tune a ResNet model for a binary classification task
    (e.g., 'target_object' vs. 'others') from a list of image paths.
    """
    def __init__(
        self,
        object_type: str,
        model_checkpoint: str = "microsoft/resnet-50",
        final_model_path: str = ModelConstants.FINETUNED_MODEL_DIR,
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
        # Keep trainer checkpoints inside the finetuned dir
        self.output_dir = os.path.join(self.final_model_path, "checkpoints_resnet50")

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

        # Create a Hugging Face Dataset using file paths; we'll open paths in transform
        dataset = Dataset.from_dict({"image_path": image_paths, "label": labels})
        
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
        # Ensure single-label classification behavior
        try:
            self.model.config.problem_type = "single_label_classification"
        except Exception:
            pass

    def _transform_data(self, example_batch):
        """Applies the image processor to a batch of examples."""
        # Determine batch size
        batch_size = None
        if 'label' in example_batch and isinstance(example_batch['label'], (list, tuple)):
            batch_size = len(example_batch['label'])
        elif 'labels' in example_batch and isinstance(example_batch['labels'], (list, tuple)):
            batch_size = len(example_batch['labels'])

        # Resolve images from various possible batch formats
        images = example_batch.get('image')
        if images is None and 'image_path' in example_batch:
            paths = example_batch['image_path']
            opened = []
            for p in paths:
                try:
                    img = PILImage.open(p).convert('RGB')
                except Exception:
                    img = None
                opened.append(img)
            images = opened
        elif images is None and 'pixel_values' in example_batch:
            # Already transformed
            out = {"pixel_values": example_batch['pixel_values']}
            if 'label' in example_batch:
                out['labels'] = example_batch['label']
            elif 'labels' in example_batch:
                out['labels'] = example_batch['labels']
            return out
        elif images is None:
            images = example_batch.get('images') or example_batch.get('file')

        # Ensure list form
        if images is None:
            images = []
        if not isinstance(images, (list, tuple)):
            images = [images]

        # Replace any Nones with dummy images, also handle empty list using batch_size
        def _dummy():
            return PILImage.new('RGB', (256, 256), (0, 0, 0))

        images = [img if isinstance(img, PILImage.Image) else (_dummy() if img is None else img) for img in images]
        if batch_size is not None and len(images) == 0:
            images = [_dummy() for _ in range(batch_size)]

        inputs = self.image_processor(images, return_tensors='pt')
        # Hugging Face Trainer expects 'labels'
        if 'label' in example_batch:
            inputs['labels'] = example_batch['label']
        elif 'labels' in example_batch:
            inputs['labels'] = example_batch['labels']
        return inputs

    @staticmethod
    def _compute_metrics(eval_pred):
        """Computes accuracy metric for evaluation."""
        try:
            # Newer HF passes EvalPrediction
            logits = getattr(eval_pred, 'predictions', None)
            labels = getattr(eval_pred, 'label_ids', None)
            if logits is None or labels is None:
                logits, labels = eval_pred
        except Exception:
            logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        # Debug: show label mapping to catch label inversion issues
        try:
            print({"id2label": getattr(getattr(eval_pred, 'model', None), 'config', None).id2label})
        except Exception:
            pass
        return {"accuracy": acc}

    def finetune(self, num_train_epochs: int = 3, per_device_batch_size: int = 8):
        """
        Executes the full fine-tuning pipeline.
        """
        if not self.train_ds or not self.eval_ds:
            raise ValueError("Data not loaded. Please call `load_data_from_paths()` first.")
        
        self._load_model_and_processor()
        
        self.train_ds.set_transform(self._transform_data)
        self.eval_ds.set_transform(self._transform_data)
        
        try:
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
        except TypeError as e:
            print(f"ℹ️ Falling back to legacy TrainingArguments due to: {e}")
            # Older transformers versions don't support evaluation/save strategies
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_batch_size,
                per_device_eval_batch_size=per_device_batch_size,
                logging_steps=10,
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            compute_metrics=self._compute_metrics,
            data_collator=DefaultDataCollator(),
        )

        print("🚀 Starting fine-tuning...")
        trainer.train()

        # For older versions without evaluation during training, run an eval pass now
        try:
            if self.eval_ds is not None:
                eval_metrics = trainer.evaluate()
                print(f"📊 Evaluation metrics: {eval_metrics}")
        except Exception as e:
            print(f"⚠️ Post-training evaluation failed: {e}")

        print(f"✅ Training complete. Saving the best model to {self.final_model_path}")
        os.makedirs(self.final_model_path, exist_ok=True)
        trainer.save_model(self.final_model_path)
        # Persist mapping explicitly
        try:
            if hasattr(self.model, 'config'):
                self.model.config.id2label = self.id2label
                self.model.config.label2id = self.label2id
                self.model.config.save_pretrained(self.final_model_path)
        except Exception:
            pass
        try:
            # Persist the image processor so inference can load it from the same directory
            if self.image_processor is not None:
                self.image_processor.save_pretrained(self.final_model_path)
        except Exception as e:
            print(f"⚠️ Failed to save image processor: {e}")
        print("✨ Model saved successfully!")
