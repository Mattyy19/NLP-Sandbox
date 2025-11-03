public class CleanUpUserQuery {
    /**
     * Preprocess user query before passing to the NLP model.
     *
     * @param query Raw user query
     * @return Cleaned query ready for embedding
     */
    public String cleanUpUserQuery(String query) {
        // Handle null input
        if (query == null) {
            return "";
        }

        // Check if query is only entered whitespace
        if (query.isEmpty()) {
            return "";
        }

        // Convert to lowercase to normalize text
        query = query.toLowerCase();

        // Remove excessive whitespace (multiple spaces, tabs, newlines)
        query = query.replaceAll("\\s+", " ");

        // Remove leading/trailing whitespace
        query = query.strip();

        // Truncate if too long (max tokens for all-MiniLM-L6-v2 is ~256 tokens)
        // Roughly 256 tokens ≈ 200-250 words
        int maxChars = 1000;

        if (query.length() > maxChars) {
            query = query.substring(0, maxChars);

        }

        return query;
    }
}
