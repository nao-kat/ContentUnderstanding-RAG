"""Generate embeddings using Azure OpenAI."""
from typing import List
from openai import AzureOpenAI
from .config import settings


class EmbeddingGenerator:
    """Generate embeddings for text using Azure OpenAI."""
    
    def __init__(
        self,
        endpoint: str = None,
        api_key: str = None,
        api_version: str = "2024-08-01-preview",
        deployment_name: str = None,
        model: str = None
    ):
        """Initialize the embedding generator.
        
        Args:
            endpoint: Azure OpenAI endpoint
            api_key: Azure OpenAI API key
            api_version: API version
            deployment_name: Deployment name for embeddings
            model: Model name (for reference)
        """
        self.endpoint = endpoint or settings.embedding_endpoint
        self.api_key = api_key or settings.embedding_key
        self.deployment_name = deployment_name or settings.embedding_deployment_name
        self.model = model or settings.embedding_model
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Embedding endpoint and key must be provided either as arguments "
                "or via EMBEDDING_ENDPOINT and EMBEDDING_KEY environment variables"
            )
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=api_version
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors (floats)
        """
        if not texts:
            return []
        
        # Filter out empty texts
        non_empty_texts = [t if t else " " for t in texts]
        
        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(non_empty_texts), batch_size):
            batch = non_empty_texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment_name
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text string to embed
        
        Returns:
            Embedding vector (list of floats)
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
