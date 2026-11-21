# Multimodal Vector Search with CLIP, Pinecone and MongoDB Atlas

Image and text search over a 500-image animal dataset using CLIP embeddings stored in two managed
vector databases. The same three tasks run on both **Pinecone** (serverless, AWS us-east-1) and
**MongoDB Atlas Vector Search**, so the two can be compared side by side.

**Stack:** Python · PyTorch · Hugging Face Transformers (CLIP ViT-B/32) · Pinecone · MongoDB Atlas · pymongo

## What it does

| Task | How |
|---|---|
| Image similarity search | Each image becomes a 512-dimensional, L2-normalised CLIP embedding; a query image's top-5 neighbours are retrieved by cosine similarity |
| Classification without training | k-nearest-neighbour voting (k = 5) over the labels of retrieved neighbours |
| Text-to-image search | CLIP's text encoder embeds a query such as "a colorful butterfly on a flower" into the same space, and the nearest images are returned |
| Comparison | Average query latency over five queries and a feature comparison of Pinecone vs. MongoDB Atlas |

Both databases store each vector with its class label and file path as metadata. The notebook
keeps its outputs, so the retrieved images and scores can be seen without running it.

## Dataset

500 images in 10 classes (Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider,
Squirrel), 50 per class, provided with the course lab. The images are not included in this
repository. To run it, put them in `animal_dataset/<Class>/<image>.jpg`.

## Running

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your own Pinecone key and MongoDB Atlas URI
```

Export the two variables (or load `.env`), then run `vector_search.ipynb`. Before the MongoDB
part, create a Vector Search index on the collection; the JSON definition is in the notebook
(section B1). The notebook also runs in Google Colab.

## Limitations

- Classification is demonstrated on sample images; there is no held-out accuracy evaluation, and
  the query image is part of the index.
- The latency comparison averages five queries from a single client on free tiers, so it is
  indicative only.

> **Context.** Lab 14 (Vector Databases) of the Database Systems Lab, FAST-NUCES (Spring 2026).

## Licence

MIT; see `LICENSE`. The images belong to their original owners.
