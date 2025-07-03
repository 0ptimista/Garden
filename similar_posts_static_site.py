"""
A similarity index for a quarto website, using ChromaDB and Ollama and the "mxbai-embed-large" model.
The only quarto-specific parts are the presumption that the filename ends with `.qmd` and that the full-text search data is stored in `_site/search.json`.
You could easily modify this for Jekyll, or Hugo, or other formats that use YAML frontmatter.

There is much caching, batching and all attempts are made to de-duplicate effort.

You will additionally need to modify your templates to include the `similar_posts` metadata from the YAML frontmatter of your posts.

I overrode the following template partial:

    template-partials:
    - /_theme/title-block.html

This is what I added to the default `title-block.html`:

```/_theme/title-block.html
$if(similar_posts)$
<div class="similar-posts">
    <h3>Suspiciously similar content</h3>
    <ul class="related-list">
        $for(similar_posts)$
        <li class="related-item">
            <a href="$it.path$" class="related-link">$it.title$</a>
        </li>
        $endfor$
    </ul>
</div>
$endif$
```

Additional requirements

1. ollama
2. chromadb

I installed them using [homebrew](https://brew.sh/)::

```brew
brew install chromadb ollama
ollama pull mxbai-embed-large
```
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # suppress tokenizer parallelism warning

import datetime
import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional

import click
import numpy as np
import yaml
import frontmatter
from frontmatter import YAMLHandler
from transformers import AutoTokenizer

import chromadb
import ollama

# Global constants and initialization
BLOG_ROOT = Path("posts")
EMBED_MODEL = "Q78KG/gte-qwen2-1.5B-instruct"  # Direct model name
CHROMA_PATH = Path("chroma_data")
MAX_TOKENS = 512
TOKENIZER = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct")

MAX_DISTANCE = 0.45
MAX_SIMILAR = 8

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize ChromaDB collection
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    "livingthing_posts",
    metadata={"hnsw:space": "cosine"},  # Force cosine distance
)

class CleanYAMLHandler(YAMLHandler):
    """YAML handler that preserves original delimiters and list formatting"""

    def export(self, metadata, **kwargs):
        return yaml.safe_dump(
            metadata,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            width=80,
        )


def clean_metadata(metadata: Dict) -> Dict:
    """Ensure all metadata values are Chroma-compatible types"""
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            cleaned[key] = ", ".join(map(str, value))
        elif value is None:
            cleaned[key] = ""
        else:
            try:
                cleaned[key] = str(value)
            except Exception as e:
                cleaned[key] = ""
                logging.warning(f"Could not convert {key} value because {e}")
    return cleaned


def truncate_to_tokens(text: str) -> str:
    tokens = TOKENIZER.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TOKENS,
    )
    return TOKENIZER.decode(tokens)


@lru_cache(maxsize=1)
def load_search_json() -> dict:
    path = Path("_site/search.json")
    with open(path) as f:
        records = json.load(f)
    documents = {}
    for record in records:
        base_url = record["href"].split("#")[0]
        documents.setdefault(base_url, []).append(record["text"])
    return {url: "\n".join(texts) for url, texts in documents.items()}


def build_document_text(metadata: Dict, cleaned_text: str) -> str:
    # Concatenate without truncation first:
    full_text = "\n".join(
        [
            f"Title: {metadata.get('title', '')}",
            f"Categories: {', '.join(metadata.get('categories', []))}",
            cleaned_text,
        ]
    )
    # Now truncate the entire text to MAX_TOKENS tokens:
    tokens = TOKENIZER.encode(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TOKENS,
    )
    return TOKENIZER.decode(tokens)


def build_query_text(post: Dict) -> str:
    # Extract the cleaned text (assumed to be the last line).
    lines = post["content"].split("\n")
    cleaned = lines[-1]
    query = f"Represent this sentence for searching relevant passages: {cleaned}"
    tokens = TOKENIZER.encode(
        query,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TOKENS,
    )
    return TOKENIZER.decode(tokens)


def extract_content(post_path: Path, search_data: Dict = None) -> Optional[Dict]:
    """Extract metadata from .qmd and cleaned text from search.json data"""
    post = frontmatter.load(post_path)
    metadata = post.metadata
    if search_data is None:
        search_data = load_search_json()
    if metadata.get("draft", False) or metadata.get("type") == "about":
        return None
    base_url = str(post_path.with_suffix(".html"))
    cleaned_text = search_data.get(base_url, "")
    if not cleaned_text:
        logging.warning(f"Missing search data for {base_url}")
        return None
    content = build_document_text(metadata, cleaned_text)
    return {
        "path": f"/{post_path.relative_to(BLOG_ROOT).as_posix()}",
        "title": metadata.get("title", ""),
        "content": content,
        "date_modified": metadata.get("date-modified", metadata.get("date")),
        "categories": metadata.get("categories", []),
        "length": len(cleaned_text),
    }


def build_index() -> None:
    search_data = load_search_json()
    existing_metadata = {}
    batch_size = 1000
    total = collection.count()
    for offset in range(0, total, batch_size):
        batch = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
        for meta in batch["metadatas"]:
            existing_metadata[meta["path"]] = {
                "date_modified": meta.get("date_modified"),
                "last_indexed": meta.get("last_indexed"),
            }
    current_batch = []
    for post_path in BLOG_ROOT.glob("**/*.qmd"):
        if post_path.name.startswith("_"):
            continue
        post = extract_content(post_path, search_data)
        if not post:
            logging.debug(f"Skipping empty post: {post_path}")
            continue
        existing = existing_metadata.get(post["path"])
        if existing and existing.get("date_modified") == post["date_modified"]:
            logging.debug(f"Skipping unchanged: {post['path']}")
            continue
        logging.info(f"Indexing: {post['path']}")
        current_batch.append(post)
        if len(current_batch) >= 100:
            _batch_process_embedding(current_batch)
            current_batch = []
    if current_batch:
        _batch_process_embedding(current_batch)


def _batch_process_embedding(batch: List[Dict]):
    embeddings, metadatas, ids = [], [], []
    for post in batch:
        try:
            embeddings.append(generate_embeddings(post["content"]))
            metadatas.append(
                clean_metadata(
                    {
                        "path": post["path"],
                        "title": post["title"],
                        "date_modified": post["date_modified"],
                        "last_indexed": datetime.datetime.now().isoformat(),
                        "length": post["length"],
                    }
                )
            )
            ids.append(post["path"])
        except Exception as e:
            logging.error(f"Failed processing {post['path']}: {str(e)}")
    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        logging.info(f"Indexed batch of {len(ids)} documents")


@lru_cache(maxsize=128)
def generate_embeddings(content: str) -> List[float]:
    response = ollama.embed(
        model=EMBED_MODEL, input=content, options={"num_ctx": MAX_TOKENS}
    )
    return response["embeddings"][0]


def find_similar(
    target_path: Path, n: int = 3, max_distance: float = MAX_DISTANCE
) -> List[Dict]:
    search_data = load_search_json()
    target_post = extract_content(target_path, search_data)
    if not target_post:
        return []
    query_content = build_query_text(target_post)
    query_embedding = generate_embeddings(query_content)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n * 3,
        include=["metadatas", "distances"],
    )
    logging.info(f"\nSimilarity analysis for: {target_post['path']}")
    for i, (meta, dist) in enumerate(
        zip(results["metadatas"][0], results["distances"][0])
    ):
        if meta["path"] != str(target_path):
            logging.info(f"Match {i + 1}: {meta['path']} | Distance: {dist:.4f}")

    distances = [d for d in results["distances"][0] if d > 0]
    if distances:
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        logging.info(f"Distance stats - Mean: {avg_dist:.4f} | StdDev: {std_dist:.4f}")

    return [
        {"title": m["title"], "path": m["path"]}
        for m, d in zip(results["metadatas"][0], results["distances"][0])
        if d <= max_distance and m["path"] != target_post["path"]
    ][:n]


def update_yaml_metadata(max_similar:int  = MAX_SIMILAR, max_distance: float = MAX_DISTANCE) -> None:
    all_posts = collection.get(include=["metadatas"])["metadatas"]
    yaml_handler = CleanYAMLHandler()
    for post_md in all_posts:
        post_path = BLOG_ROOT / Path(post_md["path"].lstrip("/"))
        logging.info(f"Finding similar entries for {post_path}")
        if not post_path.exists():
            continue
        post = frontmatter.load(post_path)
        if post.metadata.get("type") == "about":
            continue
        similar_posts = find_similar(
            post_path, n=max_similar, max_distance=max_distance
        )
        logging.info(f"Updating metadata for {post_path}")
        if similar_posts:  # Only add if similar_posts is nonempty
            post.metadata["similar_posts"] = similar_posts
        elif "similar_posts" in post.metadata:
            # Remove the key if it exists but there are no similar posts
            del post.metadata["similar_posts"]
        with open(post_path, "w") as f:
            f.write(
                frontmatter.dumps(
                    post, handler=yaml_handler, sort_keys=False, default_flow_style=None
                )
            )



def inspect_embeddings(post_path: Path):
    """Debug tool to compare raw text vs embeddings"""
    post = extract_content(post_path)
    print(f"\n=== Content for {post['path']} ===\n")
    print(post["content"][:2000] + "...")
    embedding = generate_embeddings(post["content"])
    print(f"\nEmbedding norm: {np.linalg.norm(embedding):.2f}")
    print(f"Embedding sample: {embedding[:5]}\n")


def compare_posts(path1: Path, path2: Path):
    """Direct comparison of two posts' embeddings"""
    e1 = generate_embeddings(extract_content(path1)["content"])
    e2 = generate_embeddings(extract_content(path2)["content"])
    distance = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    print(f"Cosine similarity: {1 - distance:.4f}")


def similarity_options(func):
    """Decorator to add common similarity CLI options to a command."""
    opts = [
        click.option(
            "--max-distance",
            type=float,
            default=MAX_DISTANCE,
            show_default=True,
            help="Maximum similarity distance for similar posts.",
        ),
        click.option(
            "--max-similar",
            type=int,
            default=MAX_SIMILAR,
            show_default=True,
            help="Maximum number of similar posts to record.",
        ),
    ]
    for opt in reversed(opts):
        func = opt(func)
    return func


@click.group()
def cli():
    """Blog similarity index and topic map generator."""
    pass


@cli.command("all")
@similarity_options
@click.option(
    "--plot-diagnostics", is_flag=True, help="Plot diagnostics for UMAP projections."
)
def run_all(
    max_distance,
    max_similar,
):
    """Perform indexing, update YAML metadata, and generate topic map."""
    build_index()
    update_yaml_metadata(max_similar=max_similar, max_distance=max_distance)


@cli.command("index")
def index_cmd():
    """Index blog posts."""
    build_index()


@cli.command("update")
@similarity_options
def update_cmd(max_distance, max_similar):
    """Update YAML metadata for posts."""
    update_yaml_metadata(max_similar=max_similar, max_distance=max_distance)



if __name__ == "__main__":
    cli()
