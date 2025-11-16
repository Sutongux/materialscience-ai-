import { embedQuery } from "./embedder";
import { chromaClient } from "./chromaClient";

// Main function: takes the user's goal and returns 10 compounds
export async function getTopCompounds(goal) {
  // 1. Convert text goal → embedding vector
  const embedding = await embedQuery(goal);

  // 2. Query ChromaDB (using placeholder method for now)
  const results = await chromaClient.query({
    queryEmbedding: embedding,
    nResults: 10,
  });

  // 3. Extract clean compound list
  const topCompounds = results.map((item) => ({
    name: item.metadata.compound || "Unknown Compound",
    summary: item.metadata.summary || "",
  }));

  return topCompounds;
}

