// Placeholder Chroma client until your actual endpoint is known.
// This file only defines the structure — you'll plug in your actual DB later.

export const chromaClient = {
  async query({ queryEmbedding, nResults }) {
    console.warn("ChromaDB connection not yet configured.");

    // Return dummy data so the UI works until real DB is connected
    return Array.from({ length: nResults }, (_, i) => ({
      metadata: {
        compound: `Compound ${i + 1}`,
        summary: "Sample description — configure ChromaDB to get real results."
      }
    }));
  },
};

