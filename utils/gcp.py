from typing import List, Optional
import json
from google.cloud import secretmanager
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


def get_creds(secret_name='enterprise_chat_ai_creds'):
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        name="projects/aa-mlops-dev-inm5/secrets/{}/versions/latest".format(secret_name))

    return json.loads(response.payload.data.decode("UTF-8"))


def embed_text(
    texts: List[str] = ["banana muffins? ", "banana bread? banana muffins?"],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = "text-embedding-004",
    dimensionality: Optional[int] = 256,
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]