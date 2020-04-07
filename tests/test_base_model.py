from finetuning import BaseModel


def test_load_embedding_models():
    """
    Tests whether no exceptions are thrown when we try to load the embedding
    models for each defined BaseModel.
    """
    for base_model in BaseModel:
        base_model.load_embedding_model()


def test_load_finetune_models():
    """
    Tests whether no exceptions are thrown when we try to load the finetune
    models for each defined BaseModel.
    """
    for base_model in BaseModel:
        base_model.load_finetune_model()
