import numpy as np
import tensorflow as tf


class PatientAwareTrainer:
    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn,
        ckpt_path: str,
        threshold: float = 0.5,
        vote_tie_break: str = "soft",  # "soft" or "hard" if equal acc
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.ckpt_path = ckpt_path
        self.threshold = float(threshold)
        self.vote_tie_break = vote_tie_break

        self.best_patient_acc = -np.inf
        self.best_epoch = -1
        self.best_vote_mode = None

        # Frame-level visibility metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc_metric = tf.keras.metrics.BinaryAccuracy(
            threshold=self.threshold, name="train_frame_acc"
        )
        self.val_frame_acc_metric = tf.keras.metrics.BinaryAccuracy(
            threshold=self.threshold, name="val_frame_acc"
        )

    def _reset_epoch_metrics(self):
        self.train_loss_metric.reset_state()
        self.train_acc_metric.reset_state()
        self.val_frame_acc_metric.reset_state()

    @tf.function
    def train_step(self, x, y):
        y = tf.cast(y, tf.float32)

        with tf.GradientTape() as tape:
            y_prob = self.model(x, training=True)  # sigmoid probs expected
            y_prob = tf.cast(y_prob, tf.float32)
            loss = self.loss_fn(y, y_prob)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss_metric.update_state(loss)
        self.train_acc_metric.update_state(y, y_prob)
        return loss

    @tf.function
    def val_step(self, x, y):
        y = tf.cast(y, tf.float32)
        y_prob = self.model(x, training=False)
        y_prob = tf.cast(y_prob, tf.float32)
        self.val_frame_acc_metric.update_state(y, y_prob)
        return y_prob

    def _predict_full_validation(self, val_ds):
        probs_list = []
        labels_list = []

        for xb, yb in val_ds:
            y_prob = self.val_step(xb, yb)
            probs_list.append(tf.reshape(y_prob, [-1]).numpy())
            labels_list.append(tf.reshape(tf.cast(yb, tf.float32), [-1]).numpy())

        probs = np.concatenate(probs_list, axis=0).astype("float32")
        labels = np.concatenate(labels_list, axis=0).astype("float32")
        return probs, labels

    def aggregate_patient_level(self, probs, labels, patient_ids):
        if len(patient_ids) != len(probs) or len(labels) != len(probs):
            raise ValueError(
                f"Length mismatch: patient_ids={len(patient_ids)} probs={len(probs)} labels={len(labels)}"
            )

        patient_to_probs = {}
        patient_to_true = {}

        for pid, p, t in zip(patient_ids, probs, labels):
            patient_to_probs.setdefault(pid, []).append(float(p))
            patient_to_true[pid] = float(t)  # assume constant per patient

        pids = list(patient_to_probs.keys())

        soft_preds = []
        hard_preds = []
        true_labels = []
        avg_probs = []

        thr = self.threshold

        for pid in pids:
            p = np.asarray(patient_to_probs[pid], dtype="float32")
            t = patient_to_true[pid]

            ap = float(p.mean())
            avg_probs.append(ap)
            true_labels.append(t)

            soft_preds.append(1.0 if ap >= thr else 0.0)
            hard_preds.append(1.0 if float((p >= thr).sum()) >= (len(p) / 2.0) else 0.0)

        soft_preds = np.asarray(soft_preds, dtype="float32")
        true_labels = np.asarray(true_labels, dtype="float32")
        avg_probs = np.asarray(avg_probs, dtype="float32")

        soft_acc = float(np.mean(soft_preds == true_labels))

        
        chosen = "soft"
        final_preds = soft_preds
        patient_acc = soft_acc

        return patient_acc, chosen, avg_probs, true_labels, final_preds

    def _filter_by_channels(
        self,
        probs,
        labels,
        patient_ids,
        channels,
        good_channels,
        n_channels,
    ):
        probs = np.asarray(probs)
        labels = np.asarray(labels)
        patient_ids = np.asarray(patient_ids)
        channels = np.asarray(channels)

        # Preserve order: good_channels is a ranked preference list
        good_channels = [int(c) for c in good_channels]

        keep_mask = np.zeros(len(probs), dtype=bool)

        for pid in np.unique(patient_ids):
            idx = np.where(patient_ids == pid)[0]

            patient_channels = channels[idx]
            patient_channel_set = set(int(c) for c in patient_channels)

            # Select top-n preferred channels that this patient actually has
            selected_channels = [
                ch for ch in good_channels
                if ch in patient_channel_set
            ][:n_channels]

            if not selected_channels:
                continue  # patient contributes nothing

            selected_channels = set(selected_channels)

            keep_mask[idx] = np.array(
                [int(ch) in selected_channels for ch in patient_channels],
                dtype=bool
            )

        return probs[keep_mask], labels[keep_mask], patient_ids[keep_mask]


    def fit(
        self,
        train_ds,
        val_ds,
        val_patient_ids,
        epochs: int,
        val_channels=None,      # per-frame channel id for validation set (aligned with X_test)
        good_channels=None,     # list[int] of selected "good" channels
        n_channels=None,
        verbose: int = 1,
        ):
        """
        train_ds: tf.data.Dataset yielding (x_batch, y_batch)
        val_ds: tf.data.Dataset yielding (x_batch, y_batch) in deterministic order, no shuffle/repeat
        val_patient_ids: length == number of validation frames (same order as val_ds iteration)
        val_channels: length == number of validation frames (same order as val_ds iteration)
        good_channels: list[int] channels used to compute the "good-channels-only" patient accuracy
        """

        history = {
            "train_loss": [],
            "train_frame_acc": [],
            "val_frame_acc": [],
            "val_patient_acc_all": [],
            "val_patient_vote_all": [],
            "val_patient_acc_good": [],
            "val_patient_vote_good": [],
        }

        val_patient_ids = list(val_patient_ids)
        if val_channels is not None:
            val_channels = list(val_channels)

        for epoch in range(epochs):
            self._reset_epoch_metrics()

            # ---- Train (frame-level) ----
            for xb, yb in train_ds:
                self.train_step(xb, yb)

            # ---- Validate (frame-level + collect probs for aggregation) ----
            probs, labels = self._predict_full_validation(val_ds)

            if len(val_patient_ids) != len(probs):
                raise ValueError(
                    "val_patient_ids length must match number of validation frames. "
                    f"Got val_patient_ids={len(val_patient_ids)} vs val_frames={len(probs)}. "
                    "Ensure val_ds is deterministic (no shuffle/repeat) and aligned with arrays."
                )

            if val_channels is not None and len(val_channels) != len(probs):
                raise ValueError(
                    "val_channels length must match number of validation frames. "
                    f"Got val_channels={len(val_channels)} vs val_frames={len(probs)}."
                )

            # ---- Patient-level (all frames) ----
            patient_acc_all, chosen_all, _, _, _ = self.aggregate_patient_level(
                probs=probs,
                labels=labels,
                patient_ids=val_patient_ids,
            )

            # ---- Patient-level (good channels only) ----
            patient_acc_good = None
            chosen_good = None

            if good_channels is not None and val_channels is not None:
                probs_g, labels_g, pids_g = self._filter_by_channels(
                    probs=probs,
                    labels=labels,
                    patient_ids=val_patient_ids,
                    channels=val_channels,
                    good_channels=good_channels,
                    n_channels=n_channels,
                )

                # If no frames survive the mask, keep as None (explicit signal)
                if len(probs_g) > 0:
                    patient_acc_good, chosen_good, _, _, _ = self.aggregate_patient_level(
                        probs=probs_g,
                        labels=labels_g,
                        patient_ids=pids_g,
                    )

            # ---- Checkpoint logic: consider both metrics if available ----

            if good_channels is None or val_channels is None or patient_acc_good is None:
                # No channel selection -> use all-frame accuracy
                score_for_selection = patient_acc_all
            else:
                # Channel selection active -> use good channel accuracy
                score_for_selection = patient_acc_good

            is_best = score_for_selection > self.best_patient_acc

            if is_best:
                self.best_patient_acc = score_for_selection
                self.best_epoch = epoch
                self.best_vote_mode = chosen_all  # or chosen_good, but chosen_all is more stable
                self.model.save_weights(self.ckpt_path)  # full model save (.keras)


            # ---- Record history ----
            train_loss = float(self.train_loss_metric.result().numpy())
            train_frame_acc = float(self.train_acc_metric.result().numpy())
            val_frame_acc = float(self.val_frame_acc_metric.result().numpy())

            history["train_loss"].append(train_loss)
            history["train_frame_acc"].append(train_frame_acc)
            history["val_frame_acc"].append(val_frame_acc)

            history["val_patient_acc_all"].append(float(patient_acc_all))
            history["val_patient_vote_all"].append(chosen_all)

            history["val_patient_acc_good"].append(None if patient_acc_good is None else float(patient_acc_good))
            history["val_patient_vote_good"].append(chosen_good)

            if verbose:
                star = " *best*" if is_best else ""
                msg = (
                    f"Epoch {epoch+1}/{epochs}: "
                    f"loss={train_loss:.4f} "
                    f"train_frame_acc={train_frame_acc:.4f} "
                    f"val_frame_acc={val_frame_acc:.4f} "
                    f"patient_all={patient_acc_all:.4f}({chosen_all})"
                )
                if patient_acc_good is not None:
                    msg += f" patient_good={patient_acc_good:.4f}({chosen_good})"
                else:
                    if good_channels is not None and val_channels is not None:
                        msg += " patient_good=None(no frames)"
                msg += star
                print(msg)

        return history
