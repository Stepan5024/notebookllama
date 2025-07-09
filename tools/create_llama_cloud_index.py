import os
from dotenv import load_dotenv
from openai import OpenAI

from llama_cloud import (
    PipelineCreateEmbeddingConfig_OpenaiEmbedding,
    PipelineTransformConfig_Advanced,
    AdvancedModeTransformConfigChunkingConfig_Sentence,
    AdvancedModeTransformConfigSegmentationConfig_Page,
    PipelineCreate,
)
from llama_cloud.client import LlamaCloud
from llama_index.embeddings.openai import OpenAIEmbedding


def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY exists:", bool(os.getenv("OPENAI_API_KEY")))
    print("LLAMACLOUD_API_KEY exists:", bool(os.getenv("LLAMACLOUD_API_KEY")))

    if not openai_key:
        print("❌ Missing required API keys in .env file")
        return 1

    try:
        client = OpenAI()
        # Initialize OpenAI client with API key
        openai_client = OpenAI(api_key=openai_key)

        # Test OpenAI connection
        test_embed = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_key).get_text_embedding("test")
        print("✅ OpenAI connection test successful")

        # List available models
        models = openai_client.models.list()
        print("📦 Available models for your token:")
        for model in models.data:
            print(f"  - {model.id}")
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return 1

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
    )

    client = LlamaCloud(token=os.getenv("LLAMACLOUD_API_KEY"))

    embedding_config = PipelineCreateEmbeddingConfig_OpenaiEmbedding(
        type="OPENAI_EMBEDDING",
        component=embed_model,
    )

    segm_config = AdvancedModeTransformConfigSegmentationConfig_Page(mode="page")
    chunk_config = AdvancedModeTransformConfigChunkingConfig_Sentence(
        chunk_size=1024,
        chunk_overlap=200,
        separator="<whitespace>",
        paragraph_separator="\n\n\n",
        mode="sentence",
    )

    transform_config = PipelineTransformConfig_Advanced(
        segmentation_config=segm_config,
        chunking_config=chunk_config,
        mode="advanced",
    )

    pipeline_request = PipelineCreate(
        name="notebooklm_pipeline",
        embedding_config=embedding_config,
        transform_config=transform_config,
    )

    pipeline = client.pipelines.upsert_pipeline(request=pipeline_request)

    with open(".env", "a") as f:
        f.write(f'\nLLAMACLOUD_PIPELINE_ID="{pipeline.id}"')

    return 0


if __name__ == "__main__":
    main()
