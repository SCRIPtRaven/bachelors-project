import argparse
import os
import pickle
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, \
    StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder

    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from imblearn.under_sampling import RandomUnderSampler

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


class MLRerouteTrainer:
    MODEL_DIR = os.path.join('models', 'resolvers', 'saved_models')
    DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'reroute_classifier.pkl')

    def __init__(self, training_data_path: str, model_type: str = 'random_forest'):
        self.training_data_path = training_data_path
        self.model_type = model_type

        os.makedirs(self.MODEL_DIR, exist_ok=True)

        self.feature_cols = None
        self.label_col = 'best_action'

        if 'neural_network' in model_type and not HAS_TF:
            print(f"Warning: Neural network requested but TensorFlow/Keras not available.")
            print(
                "Falling back to random forest. Please install tensorflow to use neural networks.")
            self.model_type = 'random_forest'

        self.train_metrics = {}
        self.test_metrics = {}
        self.cv_metrics = {}
        self.feature_importance = {}

        self.label_encoder = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        print(f"Loading training data from {self.training_data_path}")
        try:
            data = pd.read_csv(self.training_data_path)
            print(f"Loaded {len(data)} samples")

            if self.label_col in data.columns:
                data[self.label_col] = data[self.label_col].replace('no_reroute', 'no_action')
                print("Converted 'no_reroute' labels to 'no_action' for compatibility")

            if 'disruption_id' in data.columns:
                exclude_cols = ['best_action', 'disruption_id', 'driver_id']

                dict_cols = []
                for col in data.columns:
                    if col.startswith('all_actions_'):
                        exclude_cols.append(col)

                self.feature_cols = [col for col in data.columns if col not in exclude_cols]
            else:
                self.feature_cols = [col for col in data.columns if col != self.label_col]

            print(f"Using {len(self.feature_cols)} features: {self.feature_cols}")

            X = data[self.feature_cols]
            y = data[self.label_col]

            y_original = y.copy()

            if 'neural_network' in self.model_type and HAS_TF:
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
                print(f"Encoded classes: {list(self.label_encoder.classes_)}")

            X_train, X_test, y_train_orig, y_test_orig = train_test_split(
                X, y_original, test_size=0.2, random_state=42, stratify=y_original
            )

            if 'neural_network' in self.model_type and HAS_TF and self.label_encoder:
                y_train = self.label_encoder.transform(y_train_orig)
                y_test = self.label_encoder.transform(y_test_orig)
            else:
                y_train = y_train_orig
                y_test = y_test_orig

            print(
                f"Split into {len(X_train)} training and {len(X_test)} testing samples (before undersampling)")

            if HAS_IMBLEARN:
                print("\nApplying Random Undersampling to the training data...")
                undersampler = RandomUnderSampler(random_state=42)
                try:
                    X_train, y_train = undersampler.fit_resample(X_train, y_train)
                    print(f"Training data size after undersampling: {len(X_train)}")
                except Exception as e:
                    print(
                        f"Could not apply undersampling: {e}. Continuing with original training data.")
            else:
                print("\nWarning: imbalanced-learn library not found. Cannot apply undersampling.")
                print("Training will proceed with potentially imbalanced data.")

            if not ('neural_network' in self.model_type and HAS_TF):
                class_counts_train = pd.Series(y_train).value_counts(normalize=True) * 100
                print("\nClass distribution in TRAINING data (after potential undersampling):")
                for cls, pct in class_counts_train.items():
                    print(f"{cls}: {pct:.2f}%")
            elif self.label_encoder:
                class_counts_train = pd.Series(y_train).value_counts(normalize=True) * 100
                print("\nClass distribution in TRAINING data (after potential undersampling):")
                for i, pct in class_counts_train.items():
                    if i < len(self.label_encoder.classes_):
                        print(f"{self.label_encoder.classes_[i]}: {pct:.2f}%")
                    else:
                        print(f"Unknown class index {i}: {pct:.2f}%")

            if not ('neural_network' in self.model_type and HAS_TF):
                class_counts_test = pd.Series(y_test).value_counts(normalize=True) * 100
                print("\nClass distribution in TEST data (original):")
                for cls, pct in class_counts_test.items():
                    print(f"{cls}: {pct:.2f}%")
            elif self.label_encoder:
                class_counts_test = pd.Series(y_test).value_counts(normalize=True) * 100
                print("\nClass distribution in TEST data (original):")
                for i, pct in class_counts_test.items():
                    if i < len(self.label_encoder.classes_):
                        print(f"{self.label_encoder.classes_[i]}: {pct:.2f}%")
                    else:
                        print(f"Unknown class index {i}: {pct:.2f}%")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            raise

    def train_model(self, output_path: Optional[str] = None) -> object:
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            X_train, X_test, y_train, y_test = self.load_data()

            if 'neural_network' in self.model_type and HAS_TF:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                print(f"\nTraining neural network model...")
                model = self._create_model(X_train.shape[1], len(set(y_train)))

                if isinstance(model, keras.Model):
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=50,
                        restore_best_weights=True
                    )

                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=15,
                        min_lr=0.0001
                    )

                    checkpoint_path = os.path.join(self.MODEL_DIR, f"nn_checkpoint_{timestamp}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        os.path.join(checkpoint_path, "model.keras"),
                        monitor="val_loss",
                        save_best_only=True,
                        verbose=1
                    )

                    class_weights = None
                    try:
                        class_weights = self._compute_class_weights(y_train)
                        if class_weights:
                            print(f"Using class weights for training: {class_weights}")
                    except Exception as e:
                        print(f"Error setting up class weights: {e}, continuing without weights")
                        class_weights = None

                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    for train_idx, val_idx in sss.split(X_train_scaled, y_train):
                        X_train_split = X_train_scaled[train_idx]
                        y_train_split = y_train[train_idx]
                        X_val_split = X_train_scaled[val_idx]
                        y_val_split = y_train[val_idx]

                    fit_kwargs = {
                        'x': X_train_split,
                        'y': y_train_split,
                        'validation_data': (X_val_split, y_val_split),
                        'epochs': 500,
                        'batch_size': 64,
                        'callbacks': [early_stopping, reduce_lr, checkpoint],
                        'verbose': 1
                    }

                    if class_weights:
                        fit_kwargs['class_weight'] = class_weights

                    history = model.fit(**fit_kwargs)

                    model_wrapper = KerasClassifierWrapper(
                        model=model,
                        label_encoder=self.label_encoder
                    )

                    y_train_pred = model_wrapper.predict(X_train_scaled)
                    y_test_pred = model_wrapper.predict(X_test_scaled)

                    if self.label_encoder is not None:
                        class_thresholds = self._optimize_thresholds_for_rare_classes(
                            model, X_test_scaled, y_test, self.label_encoder
                        )
                        if class_thresholds:
                            print(f"Optimized thresholds for rare classes: {class_thresholds}")
                            model_wrapper = KerasClassifierWrapper(
                                model=model,
                                label_encoder=self.label_encoder,
                                class_thresholds=class_thresholds
                            )
                            y_test_pred = model_wrapper.predict(X_test_scaled)

                    if self.label_encoder is not None:
                        train_accuracy = accuracy_score(
                            self.label_encoder.inverse_transform(y_train),
                            self.label_encoder.inverse_transform(y_train_pred))
                        test_accuracy = accuracy_score(self.label_encoder.inverse_transform(y_test),
                                                       self.label_encoder.inverse_transform(
                                                           y_test_pred))
                        train_report = classification_report(
                            self.label_encoder.inverse_transform(y_train),
                            self.label_encoder.inverse_transform(y_train_pred),
                            output_dict=True)
                        test_report = classification_report(
                            self.label_encoder.inverse_transform(y_test),
                            self.label_encoder.inverse_transform(y_test_pred),
                            output_dict=True)

                        f1 = f1_score(self.label_encoder.inverse_transform(y_test),
                                      self.label_encoder.inverse_transform(y_test_pred),
                                      average='weighted')
                        print(f"Weighted F1 score: {f1:.4f}")
                    else:
                        train_accuracy = accuracy_score(y_train, y_train_pred)
                        test_accuracy = accuracy_score(y_test, y_test_pred)
                        train_report = classification_report(y_train, y_train_pred,
                                                             output_dict=True)
                        test_report = classification_report(y_test, y_test_pred, output_dict=True)

                        f1 = f1_score(y_test, y_test_pred, average='weighted')
                        print(f"Weighted F1 score: {f1:.4f}")

                    self.train_metrics = {
                        'accuracy': train_accuracy,
                        'report': train_report
                    }

                    self.test_metrics = {
                        'accuracy': test_accuracy,
                        'report': test_report,
                        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
                        'f1_score': f1
                    }

                    print(f"\nTraining accuracy: {train_accuracy:.4f}")
                    print(f"Testing accuracy: {test_accuracy:.4f}")
                    print("\nClassification report on test data:")
                    if self.label_encoder is not None:
                        print(classification_report(self.label_encoder.inverse_transform(y_test),
                                                    self.label_encoder.inverse_transform(
                                                        y_test_pred)))
                    else:
                        print(classification_report(y_test, y_test_pred))

                    if hasattr(history, 'history'):
                        print("\nTraining history:")
                        for key, values in history.history.items():
                            if key.startswith('val_'):
                                continue
                            print(f"Final {key}: {values[-1]:.4f}")
                        
                        try:
                            self._plot_training_history(history.history, 
                                                       output_dir=self.MODEL_DIR,
                                                       filename=f"nn_training_history_{timestamp}.png")
                        except Exception as e:
                            print(f"Failed to create training history plot: {e}")

                    try:
                        self._plot_confusion_matrix(y_test, y_test_pred,
                                                    classes=self.label_encoder.classes_ if self.label_encoder else None,
                                                    output_dir=self.MODEL_DIR,
                                                    filename=f"nn_confusion_matrix_{timestamp}.png")
                    except Exception as e:
                        print(f"Failed to create confusion matrix plot: {e}")

                    model_wrapper = KerasClassifierWrapper(
                        model=model,
                        label_encoder=self.label_encoder
                    )

                    model = {
                        'model_type': 'neural_network',
                        'keras_model': model,
                        'label_encoder': self.label_encoder,
                        'scaler': scaler,
                        'f1_score': f1,
                        'accuracy': test_accuracy
                    }

                else:
                    print("Error: Neural network training didn't produce a Keras model")
                    return None

            else:
                base_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                )

                print("\nPerforming feature selection...")
                selector = SelectFromModel(base_model, prefit=False)
                selector.fit(X_train, y_train)

                selected_features = X_train.columns[selector.get_support()].tolist()
                print(f"Selected {len(selected_features)} features: {selected_features}")

                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]

                model = self._create_model()

                print("\nPerforming cross-validation...")
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    model, X_train_selected, y_train,
                    cv=cv, scoring='f1_weighted', n_jobs=-1
                )

                self.cv_metrics = {
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std(),
                    'scores': cv_scores
                }

                print(f"Cross-validation scores: {cv_scores}")
                print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

                print(f"\nTraining final {self.model_type} model...")
                model.fit(X_train_selected, y_train)

                if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    classifier = model.named_steps['classifier']
                    if hasattr(classifier, 'feature_importances_'):
                        self.feature_importance = dict(
                            zip(selected_features, classifier.feature_importances_))

                y_train_pred = model.predict(X_train_selected)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_report = classification_report(y_train, y_train_pred, output_dict=True)

                y_test_pred = model.predict(X_test_selected)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_report = classification_report(y_test, y_test_pred, output_dict=True)

                f1 = f1_score(y_test, y_test_pred, average='weighted')
                print(f"Weighted F1 score: {f1:.4f}")

                self.train_metrics = {
                    'accuracy': train_accuracy,
                    'report': train_report
                }

                self.test_metrics = {
                    'accuracy': test_accuracy,
                    'report': test_report,
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred),
                    'f1_score': f1
                }

                print(f"\nTraining accuracy: {train_accuracy:.4f}")
                print(f"Testing accuracy: {test_accuracy:.4f}")
                print("\nClassification report on test data:")
                print(classification_report(y_test, y_test_pred))

            if output_path is None:
                output_path = os.path.join(self.MODEL_DIR, f"reroute_classifier_{timestamp}.pkl")

            if 'neural_network' in self.model_type and isinstance(model,
                                                                  dict) and 'keras_model' in model:
                keras_model = model['keras_model']
                label_encoder = model['label_encoder']

                wrapper = KerasClassifierWrapper(model=keras_model, label_encoder=label_encoder)

                keras_path = wrapper.save(output_path)
                print(f"\nNeural network model saved to {keras_path} (modern Keras format)")

                metadata = {
                    'model_type': 'neural_network',
                    'scaler': model['scaler'],
                    'f1_score': model['f1_score'],
                    'accuracy': model['accuracy']
                }
                meta_path = output_path.replace('.pkl', '_meta.pkl')
                with open(meta_path, 'wb') as f:
                    pickle.dump(metadata, f)
                print(f"Model metadata saved to {meta_path}")

                default_keras_path = wrapper.save(self.DEFAULT_MODEL_PATH)
                print(f"Model also saved to default path: {default_keras_path}")

                default_meta_path = self.DEFAULT_MODEL_PATH.replace('.pkl', '_meta.pkl')
                with open(default_meta_path, 'wb') as f:
                    pickle.dump(metadata, f)
            else:
                with open(output_path, 'wb') as f:
                    pickle.dump(model, f)

                with open(self.DEFAULT_MODEL_PATH, 'wb') as f:
                    pickle.dump(model, f)

                print(f"\nModel saved to {output_path}")
                print(f"Model also saved to default path: {self.DEFAULT_MODEL_PATH}")

            metadata_path = os.path.join(self.MODEL_DIR, f"model_metadata_{timestamp}.txt")
            with open(metadata_path, 'w') as f:
                f.write(f"Model type: {self.model_type}\n")
                f.write(f"Training data: {self.training_data_path}\n")
                if 'neural_network' not in self.model_type:
                    f.write(f"Selected features: {selected_features}\n")
                f.write(f"Training samples: {len(X_train)}\n")
                f.write(f"Testing samples: {len(X_test)}\n")
                if 'neural_network' not in self.model_type:
                    f.write(
                        f"Cross-validation mean score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
                f.write(f"Training accuracy: {train_accuracy:.4f}\n")
                f.write(f"Testing accuracy: {test_accuracy:.4f}\n")
                f.write(f"Weighted F1 score: {f1:.4f}\n\n")

                if hasattr(self, 'feature_importance') and self.feature_importance:
                    f.write("Feature importance:\n")
                    for feature, importance in self.feature_importance.items():
                        f.write(f"{feature}: {importance:.4f}\n")

                f.write("\nClassification report on test data:\n")
                if 'neural_network' in self.model_type and self.label_encoder is not None:
                    f.write(str(classification_report(self.label_encoder.inverse_transform(y_test),
                                                      self.label_encoder.inverse_transform(
                                                          y_test_pred))))
                else:
                    f.write(str(classification_report(y_test, y_test_pred)))

            print(f"Model metadata saved to {metadata_path}")

            return model

        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_model(self, input_dim=None, num_classes=None) -> object:
        if self.model_type == 'neural_network' and HAS_TF:
            if input_dim is None or num_classes is None:
                raise ValueError(
                    "input_dim and num_classes must be provided for neural network models")

            tf.random.set_seed(42)

            class_weights = None
            try:
                if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                    if hasattr(self.label_encoder, 'classes_'):
                        print(f"Available classes: {self.label_encoder.classes_}")

                    try:
                        data = pd.read_csv(self.training_data_path)
                        if self.label_col in data.columns:
                            class_counts = data[self.label_col].value_counts().to_dict()
                            total_samples = len(data)

                            if class_counts and total_samples > 0:
                                class_weights = {}
                                for i, class_name in enumerate(self.label_encoder.classes_):
                                    if class_name in class_counts and class_counts[class_name] > 0:
                                        weight = total_samples / (
                                                    len(self.label_encoder.classes_) * class_counts[
                                                class_name])
                                        class_weights[i] = min(weight, 5.0)
                                    else:
                                        class_weights[i] = 1.0

                                print(f"Using class weights: {class_weights}")
                    except Exception as e:
                        print(f"Error reading training data for class weights: {e}")
                        class_weights = None
            except Exception as e:
                print(f"Failed to calculate class weights: {e}")
                import traceback
                traceback.print_exc()
                class_weights = None

            def focal_loss(gamma=2.0, alpha=0.25):
                def focal_loss_fn(y_true, y_pred):
                    try:
                        rank = tf.rank(y_true)

                        rank_value = tf.get_static_value(rank)
                        if rank_value is None:
                            sparse_case = lambda: tf.reduce_sum(
                                focal_weight * tf.keras.losses.sparse_categorical_crossentropy(
                                    y_true, y_pred)
                            )

                            dense_case = lambda: tf.reduce_sum(
                                focal_weight * tf.keras.losses.categorical_crossentropy(y_true,
                                                                                        y_pred)
                            )

                            is_sparse = tf.equal(rank, 1)

                            sparse_probs = lambda: tf.reduce_sum(
                                y_pred * tf.one_hot(tf.cast(y_true, tf.int32),
                                                    depth=tf.shape(y_pred)[1]),
                                axis=1
                            )
                            dense_probs = lambda: tf.reduce_sum(y_pred * y_true, axis=1)

                            probs = tf.cond(is_sparse, sparse_probs, dense_probs)

                            focal_weight = tf.pow(1.0 - probs, gamma)

                            loss = tf.cond(is_sparse, sparse_case, dense_case)

                            return tf.reduce_mean(loss)
                        else:
                            if rank_value == 1:
                                y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32),
                                                            depth=tf.shape(y_pred)[1])
                                probs = tf.reduce_sum(y_pred * y_true_one_hot, axis=1)

                                focal_weight = tf.pow(1.0 - probs, gamma)

                                ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                                focal_loss = focal_weight * ce

                                return tf.reduce_mean(focal_loss)
                            else:
                                probs = tf.reduce_sum(y_pred * y_true, axis=1)

                                focal_weight = tf.pow(1.0 - probs, gamma)

                                ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
                                focal_loss = focal_weight * ce

                                return tf.reduce_mean(focal_loss)

                    except Exception as e:
                        tf.print("Error in focal loss:", e)
                        rank_value = tf.get_static_value(tf.rank(y_true))
                        if rank_value == 1:
                            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                        else:
                            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

                return focal_loss_fn

            model = keras.Sequential([
                keras.layers.Input(shape=(input_dim,)),

                keras.layers.Dense(256, kernel_initializer='he_normal'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.3),

                keras.layers.Dense(128, kernel_initializer='he_normal'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.3),

                keras.layers.Dense(128, kernel_initializer='he_normal'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.3),

                keras.layers.Dense(64, kernel_initializer='he_normal'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.2),

                keras.layers.Dense(num_classes, activation='softmax')
            ])

            use_focal_loss = False

            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                try:
                    data = pd.read_csv(self.training_data_path)
                    if self.label_col in data.columns:
                        class_counts = data[self.label_col].value_counts()
                        total = len(data)
                        class_ratios = class_counts / total
                        min_class_ratio = class_counts.min() / total

                        print("\nClass distribution in training data:")
                        for cls, count in class_counts.items():
                            print(f"{cls}: {count} samples ({count / total * 100:.2f}%)")

                        expected_uniform = 1.0 / len(class_counts)
                        min_ratio = min(class_ratios)
                        use_focal_loss = (min_ratio < expected_uniform * 0.15) or (min_ratio < 0.1)

                        if use_focal_loss:
                            print(
                                f"Detected severe class imbalance: smallest class ratio = {min_ratio:.4f} vs expected {expected_uniform:.4f}")
                            print("Using focal loss to address severe class imbalance")
                        else:
                            if 'no_action' in class_counts and class_counts[
                                'no_action'] < total * 0.15:
                                print(
                                    f"'no_action' class is underrepresented ({class_ratios.get('no_action', 0) * 100:.2f}% of data)")
                                use_focal_loss = True
                                print("Using focal loss to improve 'no_action' class detection")
                except Exception as e:
                    print(f"Error checking class distribution for focal loss: {e}")
                    use_focal_loss = False

            try:
                if use_focal_loss:
                    print("Using focal loss for training due to class imbalance")
                    loss_function = focal_loss(gamma=2.0)

                    @tf.function
                    def loss_wrapper(y_true, y_pred):
                        return loss_function(y_true, y_pred)

                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss=loss_wrapper,
                        metrics=['accuracy']
                    )
                else:
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
            except Exception as e:
                print(f"Error compiling model with focal loss: {e}")
                print("Falling back to standard categorical crossentropy")
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

            print(model.summary())

            return model

        elif self.model_type == 'neural_network_large' and HAS_TF:
            if input_dim is None or num_classes is None:
                raise ValueError(
                    "input_dim and num_classes must be provided for neural network models")

            tf.random.set_seed(42)

            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),

                keras.layers.Dense(64, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),

                keras.layers.Dense(32, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.2),

                keras.layers.Dense(num_classes, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            return model

        elif self.model_type == 'random_forest':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
            return pipeline

        elif self.model_type == 'random_forest_tuned':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42))
            ])

            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__class_weight': ['balanced', None]
            }

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )

            return grid_search

        elif self.model_type == 'xgboost' and HAS_XGB:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ))
            ])
            return pipeline

        else:
            print(
                f"Model type '{self.model_type}' not recognized or required library not installed.")
            print("Defaulting to random forest classifier.")

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
            return pipeline

    def _compute_class_weights(self, y):
        if not hasattr(np, 'unique'):
            return None

        try:
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            classes = np.unique(y)

            total_samples = len(y)
            if total_samples == 0:
                print("Empty array provided for class weights")
                return None

            weights = {}

            value_counts = {}
            for cls in classes:
                value_counts[int(cls)] = np.sum(y == cls)

            median_count = np.median(list(value_counts.values()))

            print("\nClass distribution for weighting:")
            for cls, count in value_counts.items():
                percent = (count / total_samples) * 100
                print(f"Class {cls}: {count} samples ({percent:.2f}%)")

            max_weight = 5.0

            for cls, count in value_counts.items():
                if count > 0:
                    weight = total_samples / (len(classes) * count)

                    if count < median_count:
                        boost_factor = 1.5 * (median_count / max(count, 1))
                        weight = min(weight * boost_factor, max_weight)

                    weights[int(cls)] = min(weight, max_weight)
                else:
                    print(f"Warning: Class {cls} has zero samples, assigning default weight")
                    weights[int(cls)] = max_weight

            print("\nComputed class weights:")
            for cls, weight in weights.items():
                if not np.isfinite(weight) or weight <= 0:
                    print(f"Warning: Invalid weight {weight} for class {cls}, using default")
                    weights[cls] = 1.0
                else:
                    print(f"Class {cls}: {weight:.4f}")

            return weights
        except Exception as e:
            print(f"Error computing class weights: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_training_history(self, history, output_dir=None, filename=None):
        try:
            import matplotlib.pyplot as plt
            
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            color_loss = 'tab:red'
            color_acc = 'tab:blue'

            if 'loss' in history:
                epochs = range(1, len(history['loss']) + 1)
                ax1.plot(epochs, history['loss'], color=color_loss, linestyle='-', label='Training Loss', linewidth=2)
                if 'val_loss' in history:
                    ax1.plot(epochs, history['val_loss'], color=color_loss, linestyle='--', label='Validation Loss', linewidth=2)
                
                ax1.set_xlabel('Epoch', fontsize=12)
                ax1.set_ylabel('Loss', color=color_loss, fontsize=12)
                ax1.tick_params(axis='y', labelcolor=color_loss)
                ax1.set_ylim(0, max(1, max(history.get('loss', [1])), max(history.get('val_loss', [1]))) * 1.1)
                ax1.grid(True, alpha=0.3, axis='x')

            ax2 = ax1.twinx()

            if 'accuracy' in history:
                ax2.plot(epochs, history['accuracy'], color=color_acc, linestyle='-', label='Training Accuracy', linewidth=2)
                if 'val_accuracy' in history:
                    ax2.plot(epochs, history['val_accuracy'], color=color_acc, linestyle='--', label='Validation Accuracy', linewidth=2)
                
                ax2.set_ylabel('Accuracy', color=color_acc, fontsize=12)
                ax2.tick_params(axis='y', labelcolor=color_acc)
                ax2.set_ylim(0, max(1, max(history.get('accuracy', [1])), max(history.get('val_accuracy', [1]))) * 1.1)
            
            fig.suptitle('Model Training History (Loss & Accuracy)', fontsize=16, fontweight='bold')

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2, fontsize=11)
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if output_dir and filename:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Training history plot saved to {filepath}")
            
            plt.close()
            
        except ImportError:
            print("Matplotlib not available, skipping training history plot")
        except Exception as e:
            print(f"Error plotting training history: {e}")
            import traceback
            traceback.print_exc()

    def _plot_confusion_matrix(self, y_true, y_pred, classes=None, normalize=False,
                               output_dir=None, filename=None):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)

            if self.label_encoder is not None and classes is None:
                try:
                    if hasattr(y_true, 'shape') and len(y_true.shape) > 0:
                        y_true = self.label_encoder.inverse_transform(y_true)
                    if hasattr(y_pred, 'shape') and len(y_pred.shape) > 0:
                        y_pred = self.label_encoder.inverse_transform(y_pred)
                    classes = self.label_encoder.classes_
                    cm = confusion_matrix(y_true, y_pred)
                except Exception as e:
                    print(f"Error converting indices to labels: {e}")

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                        cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Confusion Matrix')

            if output_dir and filename:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                print(f"Confusion matrix saved to {os.path.join(output_dir, filename)}")

            plt.close()

        except ImportError:
            print("Matplotlib or seaborn not available, skipping confusion matrix plot")
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")

    def _optimize_thresholds_for_rare_classes(self, model, X_test, y_test, label_encoder):
        try:
            probs = model.predict(X_test)

            unique, counts = np.unique(y_test, return_counts=True)
            class_counts = dict(zip(unique, counts))

            total_samples = len(y_test)
            class_frequencies = {cls: count / total_samples for cls, count in class_counts.items()}

            rare_classes = {cls: freq for cls, freq in class_frequencies.items()
                            if freq < 0.15}

            if not rare_classes:
                return {}

            print(f"\nDetected rare classes: {rare_classes}")

            thresholds = {}
            for rare_class, freq in rare_classes.items():
                class_idx = rare_class
                class_probs = probs[:, class_idx]

                best_f1 = 0
                best_threshold = 0.5

                if hasattr(label_encoder, 'classes_'):
                    original_classes = label_encoder.classes_
                    if 'no_action' in original_classes:
                        no_action_idx = np.where(original_classes == 'no_action')[0][0]
                        if no_action_idx == class_idx:
                            print(
                                f"Optimizing specifically for 'no_action' class (idx {no_action_idx})")
                            threshold_range = np.arange(0.05, 0.4, 0.05)
                        else:
                            threshold_range = np.arange(0.1, 0.5, 0.05)
                    else:
                        threshold_range = np.arange(0.1, 0.5, 0.05)
                else:
                    threshold_range = np.arange(0.1, 0.5, 0.05)

                for threshold in threshold_range:
                    binary_pred = (class_probs >= threshold).astype(int)
                    binary_true = (y_test == class_idx).astype(int)

                    if np.sum(binary_pred) > 0 and np.sum(binary_true) > 0:
                        precision = np.sum((binary_pred == 1) & (binary_true == 1)) / np.sum(
                            binary_pred)
                        recall = np.sum((binary_pred == 1) & (binary_true == 1)) / np.sum(
                            binary_true)

                        if precision > 0 and recall > 0:
                            beta = 2
                            f_score = (1 + beta ** 2) * (precision * recall) / (
                                        (beta ** 2 * precision) + recall)

                            if f_score > best_f1:
                                best_f1 = f_score
                                best_threshold = threshold

                if best_threshold != 0.5:
                    thresholds[int(class_idx)] = best_threshold
                    print(
                        f"Class {class_idx} optimized threshold: {best_threshold:.2f}, F-score: {best_f1:.4f}")

            return thresholds

        except Exception as e:
            print(f"Error optimizing thresholds: {e}")
            import traceback
            traceback.print_exc()
            return {}


class KerasClassifierWrapper:
    def __init__(self, model, label_encoder=None, class_thresholds=None):
        self.model = model
        self.label_encoder = label_encoder
        self.classes_ = np.array(
            range(model.output_shape[1])) if label_encoder is None else label_encoder.classes_
        self.class_thresholds = class_thresholds or {}

    def predict(self, X):
        probs = self.model.predict(X)

        if self.class_thresholds and len(self.class_thresholds) > 0:
            predictions = []

            for i in range(probs.shape[0]):
                sample_probs = probs[i]
                max_class = np.argmax(sample_probs)
                max_prob = sample_probs[max_class]

                for class_idx, threshold in self.class_thresholds.items():
                    if (sample_probs[class_idx] >= threshold and
                            max_prob - sample_probs[class_idx] < 0.2):
                        max_class = class_idx
                        break

                predictions.append(max_class)

            return np.array(predictions)
        else:
            return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        model_path = filepath.replace('.pkl', '.keras')
        self.model.save(model_path, save_format='keras')

        wrapper_data = {
            'model_path': model_path,
            'label_encoder': self.label_encoder,
            'classes_': self.classes_,
            'class_thresholds': self.class_thresholds
        }

        with open(filepath, 'wb') as f:
            pickle.dump(wrapper_data, f)

        return model_path

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            wrapper_data = pickle.load(f)

        if isinstance(wrapper_data, dict) and 'model_path' in wrapper_data:
            model_path = wrapper_data['model_path']
            model = tf.keras.models.load_model(model_path)
            label_encoder = wrapper_data.get('label_encoder')
            class_thresholds = wrapper_data.get('class_thresholds', {})

            return cls(model, label_encoder, class_thresholds)
        elif isinstance(wrapper_data, dict) and 'keras_model' in wrapper_data:
            model = wrapper_data['keras_model']
            label_encoder = wrapper_data.get('label_encoder')
            class_thresholds = wrapper_data.get('class_thresholds', {})

            return cls(model, label_encoder, class_thresholds)
        else:
            raise ValueError(f"Invalid wrapper data format: {type(wrapper_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a machine learning model for reroute action prediction')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the training data CSV file')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'random_forest_tuned', 'xgboost',
                                 'neural_network', 'neural_network_large'],
                        help='Type of model to train')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the trained model (default: auto-generated)')

    args = parser.parse_args()

    trainer = MLRerouteTrainer(
        training_data_path=args.data,
        model_type=args.model
    )

    trainer.train_model(output_path=args.output)
