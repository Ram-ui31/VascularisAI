"""Serializable inference wrapper for Vascularis probability blends."""
from __future__ import annotations

import numpy as np


class VesselBlend:
    def __init__(self, extra_trees, catboost, stack, cat_columns, weights, threshold):
        self.extra_trees = extra_trees
        self.catboost = catboost
        self.stack = stack
        self.cat_columns = list(cat_columns)
        self.weights = np.asarray(weights, dtype=float)
        self.threshold = float(threshold)

    def predict_proba(self, frame):
        cat_frame = frame.copy()
        cat_frame[self.cat_columns] = cat_frame[self.cat_columns].astype(str)
        component_probabilities = np.column_stack([
            self.extra_trees.predict_proba(frame)[:, 1],
            self.catboost.predict_proba(cat_frame)[:, 1],
            self.stack.predict_proba(frame)[:, 1],
        ])
        positive = component_probabilities @ self.weights
        return np.column_stack([1 - positive, positive])

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] >= self.threshold).astype(int)
