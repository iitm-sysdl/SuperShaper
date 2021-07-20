# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" seqeval metric. """

import importlib
from typing import List, Optional, Union

from seqeval.metrics import accuracy_score, classification_report

import datasets
import numpy as np
from utils import flatten_list


_CITATION = ""
_DESCRIPTION = ""
_KWARGS_DESCRIPTION = ""


class mlm_accuracy(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(
                        datasets.Value("string", id="label"), id="sequence"
                    ),
                    "references": datasets.Sequence(
                        datasets.Value("string", id="label"), id="sequence"
                    ),
                }
            ),
            codebase_urls=[
                "https://github.com/deepset-ai/FARM/blob/master/farm/evaluation/metrics.py"
            ],
            reference_urls=[
                "https://github.com/deepset-ai/FARM/blob/master/farm/evaluation/metrics.py"
            ],
        )

    def _compute(
        self,
        predictions,
        references,
    ):
        if type(predictions) == type(references) == list:
            preds = np.array(list(flatten_list(predictions)))
            labels = np.array(list(flatten_list(references)))
        assert type(preds) == type(labels) == np.ndarray
        correct = preds == labels
        return {"accuracy": correct.mean()}
