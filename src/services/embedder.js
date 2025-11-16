// Placeholder embedder — replace with your real embedding model or API later.

export async function embedQuery(text) {
  console.warn("Embedder not yet connected to a real model.");

  // Fake embedding: return a predictable numeric vector
  return Array(10).fill(text.length % 10);
}

