// Load environment variables from backend/.env
require('dotenv').config({ path: require('path').join(__dirname, '../.env') });

const apiKey = process.env.RAGIE_API_KEY;

async function deleteAllCoachDataDocuments() {
  if (!apiKey) {
    console.error("Error: RAGIE_API_KEY environment variable is not set.");
    process.exit(1);
  }
  console.log("RAGIE_API_KEY found. Starting deletion process for scope 'coach_data'...");

  let deletedCount = 0;
  let failedCount = 0;
  let pageCount = 0;
  let nextCursor = null; // Ragie uses cursor-based pagination

  try {
    while (true) {
      pageCount++;
      console.log(`Fetching page ${pageCount}...`);
      const url = new URL("https://api.ragie.ai/documents");
      // Filter specifically for 'coach_data' scope
      url.searchParams.set("filter", JSON.stringify({ scope: "coach_data" }));
      url.searchParams.set("page_size", "100"); // Fetch in batches of 100

      // Add cursor for pagination if available from previous response
      if (nextCursor) {
        url.searchParams.set("cursor", nextCursor);
      }

      const listResponse = await fetch(url, {
        headers: { authorization: `Bearer ${apiKey}` },
      });

      if (!listResponse.ok) {
        console.error(
          `Failed to retrieve data from Ragie API (Page ${pageCount}): ${listResponse.status} ${listResponse.statusText}`
        );
        const errorBody = await listResponse.text();
        console.error("Response body:", errorBody);
        // Decide whether to stop or try to continue
        failedCount += 1; // Count this page fetch as a failure point
        break; // Stop processing if listing fails
      }

      const payload = await listResponse.json();

      if (!payload.documents || payload.documents.length === 0) {
         if (pageCount === 1) {
            console.log("No documents found with scope 'coach_data'.");
         } else {
            console.log("No more documents found on subsequent pages.");
         }
         break; // Exit loop if no documents are returned
      }

      console.log(`Found ${payload.documents.length} documents on page ${pageCount}. Attempting deletion...`);

      for (const document of payload.documents) {
        if (!document.id) {
          console.warn("Found document without ID, skipping.");
          failedCount++;
          continue;
        }
        try {
          const deleteResponse = await fetch(
            `https://api.ragie.ai/documents/${document.id}`,
            {
              method: "DELETE",
              headers: {
                authorization: `Bearer ${apiKey}`,
              },
            }
          );

          if (!deleteResponse.ok) {
            console.error(
              `Failed to delete document ${document.id}: ${deleteResponse.status} ${deleteResponse.statusText}`
            );
             const errorBody = await deleteResponse.text();
             console.error("Delete Response body:", errorBody);
            failedCount++;
          } else {
            console.log(`Deleted document ${document.id}`);
            deletedCount++;
          }
        } catch (deleteError) {
           console.error(`Error during deletion request for document ${document.id}:`, deleteError);
           failedCount++;
        }
      } // End loop for documents on current page

      // Check for next page cursor
      if (payload.pagination && payload.pagination.next_cursor) {
        nextCursor = payload.pagination.next_cursor;
        console.log("Moving to next page...");
      } else {
        console.log("No next cursor found. Reached the end of documents.");
        break; // Exit the main while loop
      }
    } // End while(true) loop for pagination

  } catch (error) {
    console.error("An unexpected error occurred during the process:", error);
  } finally {
    console.log("\n--- Deletion Summary ---");
    console.log(`Scope Filter Applied: {"scope": "coach_data"}`);
    console.log(`Documents Successfully Deleted: ${deletedCount}`);
    console.log(`Deletion Failures (includes fetch errors): ${failedCount}`);
    console.log("----------------------\n");
  }
}

// Execute the main function
deleteAllCoachDataDocuments();
