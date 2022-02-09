# use pca/svd to reduce the embedding dimension and save the reduce matrix
import numpy as np
import transformers
import os
import joblib
from custom_layers import custom_bert, custom_mobile_bert

# taken from https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
def svd(X, k=None):

    is_expanded = False
    # Data matrix X, X doesn't need to be 0-centered
    try:
        n, m = X.shape
    except ValueError:
        # expand dims if X is a vector
        X = np.expand_dims(X, axis=0)
        n, m = X.shape
        is_expanded = True

    # print("before svd: X.shape = {}".format(X.shape))
    # Compute full SVD
    U, Sigma, Vh = np.linalg.svd(
        X,
        full_matrices=False,  # It's not necessary to compute the full matrix of U or V
        compute_uv=True,
    )
    # Transform X with SVD components
    X_svd = np.dot(U, np.diag(Sigma))
    top_k_eigenvectors = X_svd[:, :k]
    if is_expanded:
        top_k_eigenvectors = np.squeeze(top_k_eigenvectors, axis=0)
    return top_k_eigenvectors


# cache this function  output
def reduce_emb_dim(model_name, emb_dim, save_path):
    # make sure the model_name is not a directory
    if os.path.isdir(model_name):
        raise ValueError(
            "model_name should not be a directory but should be a huggingface model name"
        )
    elif os.path.isfile(model_name):
        raise ValueError(
            "model_name should not be a file but should be a huggingface model name"
        )

    assert isinstance(emb_dim, int)
    assert isinstance(model_name, str)

    # check if save path already exists
    if os.path.exists(save_path):
        raise ValueError(
            f"save_path {save_path} already exists. Please delete before running this script again"
        )

    model = transformers.BertForMaskedLM.from_pretrained(
        model_name,
    )
    # for key in model.state_dict():
    #     print(key, ":", model.state_dict()[key].shape)

    keys_to_reduce = [
        "bert.embeddings.word_embeddings.weight",
        "bert.embeddings.position_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight",
        "bert.embeddings.LayerNorm.weight",
        "bert.embeddings.LayerNorm.bias",
    ]

    reduced_emb_matrix = {}

    for key in model.state_dict():
        if key in keys_to_reduce:
            print(key, ":", model.state_dict()[key].shape)
            emb = model.state_dict()[key]
            emb_pca = svd(emb, k=emb_dim)
            print(key, ":", emb_pca.shape)
            reduced_emb_matrix[key] = emb_pca

    # encoder.layer.11.attention.self.query.weight
    # encoder.layer.11.attention.self.query.bias
    # encoder.layer.11.attention.self.key.weight
    # encoder.layer.11.attention.self.key.bias
    # encoder.layer.11.attention.self.value.weight
    # encoder.layer.11.attention.self.value.bias
    # encoder.layer.11.attention.output.dense.weight
    # encoder.layer.11.attention.output.dense.bias
    # encoder.layer.11.attention.output.LayerNorm.weight
    # encoder.layer.11.attention.output.LayerNorm.bias
    # encoder.layer.11.intermediate.dense.weight
    # encoder.layer.11.intermediate.dense.bias
    # encoder.layer.11.output.dense.weight
    # encoder.layer.11.output.dense.bias
    # encoder.layer.11.output.LayerNorm.weight
    # encoder.layer.11.output.LayerNorm.bias

    for i in range(12):
        keys_to_slice = [
            (f"bert.encoder.layer.{i}.attention.self.query.weight", "col"),
            (f"bert.encoder.layer.{i}.attention.self.query.bias", "col"),
            (f"bert.encoder.layer.{i}.attention.self.key.weight", "col"),
            (f"bert.encoder.layer.{i}.attention.self.key.bias", "col"),
            (f"bert.encoder.layer.{i}.attention.self.value.weight", "col"),
            (f"bert.encoder.layer.{i}.attention.self.value.bias", "col"),
            (
                f"bert.encoder.layer.{i}.attention.output.dense.weight",
                "row",
            ),
            (
                f"bert.encoder.layer.{i}.attention.output.dense.bias",
                "row",
            ),
            (
                f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight",
                "row",
            ),
            (
                f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias",
                "row",
            ),
            (f"bert.encoder.layer.{i}.intermediate.dense.weight", "col"),
            (f"bert.encoder.layer.{i}.intermediate.dense.bias", "col"),
            (f"bert.encoder.layer.{i}.output.dense.weight", "row"),
            (f"bert.encoder.layer.{i}.output.dense.bias", "row"),
            (f"bert.encoder.layer.{i}.output.LayerNorm.weight", "row"),
            (f"bert.encoder.layer.{i}.output.LayerNorm.bias", "row"),
        ]
        for key, axis in keys_to_slice:
            weights = model.state_dict()[key]
            try:
                if axis == "col":
                    reduced_emb_matrix[key] = weights[:, :emb_dim]
                elif axis == "row":
                    reduced_emb_matrix[key] = weights[:emb_dim, :]
            except (ValueError, IndexError):
                reduced_emb_matrix[key] = weights[:emb_dim]
    keys_to_slice = [
        (f"cls.predictions.bias", "none"),
        (f"cls.predictions.transform.dense.weight", "both"),
        (f"cls.predictions.transform.dense.bias", "col"),
        (f"cls.predictions.transform.LayerNorm.weight", "col"),
        (f"cls.predictions.transform.LayerNorm.bias", "col"),
        (f"cls.predictions.decoder.weight", "col"),
        (f"cls.predictions.decoder.bias", "none"),
    ]
    for key, axis in keys_to_slice:
        weights = model.state_dict()[key]
        try:
            if axis == "col":
                reduced_emb_matrix[key] = weights[:, :emb_dim]
            elif axis == "row":
                reduced_emb_matrix[key] = weights[:emb_dim, :]
            elif axis == "both":
                reduced_emb_matrix[key] = weights[:emb_dim, :emb_dim]
            elif axis == "none":
                reduced_emb_matrix[key] = weights
        except (ValueError, IndexError):
            reduced_emb_matrix[key] = weights[:emb_dim]

    del model

    joblib.dump(reduced_emb_matrix, save_path)
    return reduced_emb_matrix
