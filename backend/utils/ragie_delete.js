// This script deletes all documents from the Ragie API with a specific scope. Change the scope as needed.

import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

// Resolve the directory name for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables from backend/.env
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const apiKey = process.env.RAGIE_API_KEY;

while (true) {
  const url = new URL("https://api.ragie.ai/documents");
  url.searchParams.set("filter", JSON.stringify({ scope: "clone_data" }));

  const response = await fetch(url, {
    headers: { authorization: `Bearer ${apiKey}` },
  });

  if (!response.ok) {
    throw new Error(
      `Failed to retrieve data from Ragie API: ${response.status} ${response.statusText}`
    );
  }
  const payload = await response.json();

  for (const document of payload.documents) {
    const response = await fetch(
      `https://api.ragie.ai/documents/${document.id}`,
      {
        method: "DELETE",
        headers: {
          authorization: `Bearer ${apiKey}`,
        },
      }
    );

    if (!response.ok) {
      throw new Error(
        `Failed to delete document ${document.id}: ${response.status} ${response.statusText}`
      );
    }
    console.log(`Deleted document ${document.id}`);
  }

  if (!payload.pagination.next_cursor) {
    console.warn("No more documents\n");
    break;
  }
}
