import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS = 15
FINE_TUNE_LAST_LAYERS = 30
LEARNING_RATE = 1e-5
PATIENCE = 4

CLASS_NAMES = ["glass", "metal", "paper", "plastic", "trash"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RAW_DIR = Path("data/raw")
SPLIT_DIR = Path("data/split")
WORK_DIR = Path("data/finetune_work")
OUTPUT_DIR = Path("outputs")

OLD_TRAIN = SPLIT_DIR / "train"
OLD_VAL = SPLIT_DIR / "val"
OLD_TEST = SPLIT_DIR / "test"

BASELINE_PATH = OUTPUT_DIR / "baseline_waste_classifier.keras"
CURRENT_PATH = OUTPUT_DIR / "waste_classifier.keras"
CANDIDATE_PATH = OUTPUT_DIR / "finetuned_candidate.keras"
CHECKPOINT_PATH = OUTPUT_DIR / "best_finetune.keras"

FINAL_TFLITE = OUTPUT_DIR / "waste_classifier.tflite"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def image_files(directory):
    if not directory.exists():
        return []

    return sorted(
        p for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def copy_into_split(source, digest, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Hash-based names prevent filename collisions.
    destination = (
        destination_dir /
        f"{digest[:16]}_{source.name}"
    )

    if not destination.exists():
        shutil.copy2(source, destination)


def validate_project():
    for directory in [OLD_TRAIN, OLD_VAL, OLD_TEST]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Missing {directory}. "
                "The original split must exist."
            )

    for class_name in CLASS_NAMES:
        if not (RAW_DIR / class_name).exists():
            raise FileNotFoundError(
                f"Missing class directory: {RAW_DIR / class_name}"
            )

    if not CURRENT_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline model: {CURRENT_PATH}"
        )


def backup_baseline():
    # Created once and never overwritten: later runs compare against it.
    if not BASELINE_PATH.exists():
        shutil.copy2(CURRENT_PATH, BASELINE_PATH)
        print("Baseline backup created:", BASELINE_PATH)
    else:
        print("Using preserved baseline:", BASELINE_PATH)


def prepare_data():
    """Build WORK_DIR from the original split plus new images from data/raw.

    data/raw also contains the old images, so anything whose hash is already in
    the original split (including its test set) is skipped.
    """
    print("\n=== PREPARING DATA ===")

    old_splits = [("train", OLD_TRAIN), ("val", OLD_VAL), ("test", OLD_TEST)]
    old_digests = {
        path: sha256_file(path)
        for _, split_dir in old_splits
        for path in image_files(split_dir)
    }
    old_hashes = set(old_digests.values())

    print("Unique old image hashes:", len(old_hashes))

    new_by_class = {name: [] for name in CLASS_NAMES}
    seen_raw_hashes = set()
    duplicate_count = 0
    old_image_count = 0

    for class_name in CLASS_NAMES:
        for path in image_files(RAW_DIR / class_name):
            digest = sha256_file(path)

            if digest in old_hashes:
                old_image_count += 1
                continue

            if digest in seen_raw_hashes:
                duplicate_count += 1
                continue

            seen_raw_hashes.add(digest)
            new_by_class[class_name].append((path, digest))

    print("Raw images matching old splits:", old_image_count)
    print("Duplicate raw images skipped:", duplicate_count)

    # Only the generated workspace is rebuilt; data/split and data/raw stay untouched.
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            (WORK_DIR / split / class_name).mkdir(
                parents=True, exist_ok=True
            )

    for split_name, source_dir in old_splits:
        for class_name in CLASS_NAMES:
            for path in image_files(source_dir / class_name):
                copy_into_split(
                    path,
                    old_digests[path],
                    WORK_DIR / split_name / class_name
                )

    for class_name, items in new_by_class.items():
        rng = random.Random(SEED)
        items = list(items)
        rng.shuffle(items)

        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        # Tiny groups go entirely to train instead of making one-image holdouts.
        if n >= 3:
            train_items = items[:n_train]
            val_items = items[n_train:n_train + n_val]
            test_items = items[n_train + n_val:]
        else:
            train_items = items
            val_items = []
            test_items = []

        for split_name, split_items in [
            ("train", train_items),
            ("val", val_items),
            ("test", test_items),
        ]:
            for path, digest in split_items:
                copy_into_split(path, digest, WORK_DIR / split_name / class_name)

        print(
            f"{class_name:8s}: new={n}, "
            f"new_train={len(train_items)}, "
            f"new_val={len(val_items)}, "
            f"new_test={len(test_items)}"
        )

    print("\nCombined split sizes:")
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}")
        for class_name in CLASS_NAMES:
            count = len(
                image_files(WORK_DIR / split / class_name)
            )
            print(f"  {class_name:8s}: {count}")

    if not any(new_by_class.values()):
        raise ValueError(
            "No new unique images found. "
            "Check that data/raw contains the new photos."
        )


def load_directory(directory, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )


def load_datasets():
    train_ds = load_directory(
        WORK_DIR / "train", shuffle=True
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = load_directory(
        WORK_DIR / "val"
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = load_directory(
        WORK_DIR / "test"
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def get_class_weights():
    counts = np.array([
        len(image_files(WORK_DIR / "train" / name))
        for name in CLASS_NAMES
    ], dtype=np.float64)

    if np.any(counts == 0):
        raise ValueError(
            f"Missing training images for a class: {counts.tolist()}"
        )

    # Balanced weights, capped to avoid extreme gradients.
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    weights = np.clip(weights, 0.5, 3.0)

    result = {
        i: float(weight)
        for i, weight in enumerate(weights)
    }

    print("\nClass weights:", result)
    return result


def find_backbone(model):
    for layer in model.layers:
        if (
            isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        ):
            return layer

    raise ValueError(
        "MobileNetV2 backbone not found in the saved model."
    )


def fine_tune(train_ds, val_ds, class_weights):
    print("\nLoading preserved baseline...")

    model = tf.keras.models.load_model(
        BASELINE_PATH,
        compile=False
    )

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            "Model output does not match the five configured classes."
        )

    backbone = find_backbone(model)

    # Train only the last layers; BatchNormalization stays frozen.
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-FINE_TUNE_LAST_LAYERS:]:
        if not isinstance(
            layer, tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(CHECKPOINT_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    candidate = tf.keras.models.load_model(
        CHECKPOINT_PATH,
        compile=False
    )

    candidate.save(CANDIDATE_PATH)
    print("Candidate saved:", CANDIDATE_PATH)

    return candidate


def evaluate_loss_accuracy(model, dataset):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    loss, accuracy = model.evaluate(dataset, verbose=0)
    return float(loss), float(accuracy)


def select_model(candidate, val_ds):
    """Pick baseline or candidate using validation data only; the test set stays unseen."""
    baseline = tf.keras.models.load_model(
        BASELINE_PATH,
        compile=False
    )

    old_loss, old_acc = evaluate_loss_accuracy(baseline, val_ds)
    new_loss, new_acc = evaluate_loss_accuracy(candidate, val_ds)

    print("\n=== VALIDATION COMPARISON ===")
    print(f"Baseline:  loss={old_loss:.4f}, accuracy={old_acc:.4f}")
    print(f"Candidate: loss={new_loss:.4f}, accuracy={new_acc:.4f}")

    # Replace the baseline only if accuracy improves,
    # or ties while loss improves.
    improved = (
        new_acc > old_acc + 1e-6
        or (
            abs(new_acc - old_acc) <= 1e-6
            and new_loss < old_loss
        )
    )

    if improved:
        shutil.copy2(CANDIDATE_PATH, CURRENT_PATH)
        print("Candidate selected.")
    else:
        shutil.copy2(BASELINE_PATH, CURRENT_PATH)
        print("Baseline retained.")

    return tf.keras.models.load_model(
        CURRENT_PATH,
        compile=False
    )


def evaluate_test(model, test_ds, prefix):
    y_true = np.concatenate([
        y.numpy() for _, y in test_ds
    ])

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = probabilities.argmax(axis=1)

    labels = list(range(len(CLASS_NAMES)))
    report_args = dict(labels=labels, target_names=CLASS_NAMES, zero_division=0)
    report_text = classification_report(y_true, y_pred, **report_args)
    report_dict = classification_report(y_true, y_pred, output_dict=True, **report_args)

    print(f"\n=== {prefix.upper()} TEST REPORT ===")
    print(report_text)

    with open(
        OUTPUT_DIR / f"metrics_{prefix}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report_dict, file, indent=2, ensure_ascii=False)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print("Confusion matrix:")
    print(matrix)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix — {prefix}")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"confusion_matrix_{prefix}.png",
        dpi=160
    )
    plt.close()

    return report_dict


def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    FINAL_TFLITE.write_bytes(tflite_model)

    (OUTPUT_DIR / "labels.json").write_text(
        json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    interpreter = tf.lite.Interpreter(
        model_path=str(FINAL_TFLITE)
    )
    interpreter.allocate_tensors()

    print("\nTFLite export successful:")
    print("Model:", FINAL_TFLITE)
    print("Input:", interpreter.get_input_details()[0]["shape"])
    print("Output:", interpreter.get_output_details()[0]["shape"])


def main():
    print("=== WASTE CLASSIFIER FINE-TUNING ===")

    validate_project()
    backup_baseline()
    prepare_data()

    train_ds, val_ds, test_ds = load_datasets()
    class_weights = get_class_weights()

    candidate = fine_tune(
        train_ds,
        val_ds,
        class_weights
    )

    selected = select_model(candidate, val_ds)

    # Both models are scored on exactly the same test dataset.
    baseline = tf.keras.models.load_model(
        BASELINE_PATH,
        compile=False
    )

    evaluate_test(baseline, test_ds, "baseline")
    evaluate_test(selected, test_ds, "selected")

    export_tflite(selected)

    print("\n=== COMPLETE ===")
    print("Selected Keras model:", CURRENT_PATH)
    print("Baseline backup:", BASELINE_PATH)
    print("TFLite model:", FINAL_TFLITE)
    print("Working datasets:", WORK_DIR)


if __name__ == "__main__":
    main()
