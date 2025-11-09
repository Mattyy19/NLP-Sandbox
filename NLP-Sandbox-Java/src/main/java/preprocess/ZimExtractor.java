package preprocess;

import net.kiwix.jzim.ZimFile;
import net.kiwix.jzim.ZimArticle;
import org.json.JSONObject;
import org.jsoup.Jsoup;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;

/**
 * Extracts text and titles from a ZIM file into a JSONL dataset.
 * @author Alain Mignot
 */
public class ZimExtractor {
    /**
     * Cleans HTML content by removing tags and extra whitespace.
     * @param rawHtml the raw HTML content
     * @return the cleaned plain text
     */
    private static String cleanHtml(String rawHtml) {
        // Parse HTML and remove tags
        String text = Jsoup.parse(rawHtml).text();
        // Remove reference markers
        text = text.replaceAll("\\[[^\\]]*\\]", " ");
        // Normalize whitespace
        text = text.replaceAll("\\s+", " ").trim();
        return text;
    }

    /**
     * Extracts articles from a ZIM file and writes them to a JSONL file.
     * @param args command line arguments: zim-file and output-jsonl
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java preprocess.ZimExtractor <zim-file> <output-jsonl>");
            return;
        }

        String zimFilePath = args[0];
        String outputJsonl = args[1];
        int datasetCount = 0;

        try (ZimFile zim = new ZimFile(Paths.get(zimFilePath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputJsonl))) {

            System.out.println("Extracting articles from ZIM file: " + zimFilePath);

            // Iterate through all articles in the ZIM file
            for (ZimArticle article : zim) {
                try {
                    // Skip non-content namespaces
                    if (!"C".equals(article.getNamespace())) continue;
                    // Skip redirects, non-articles, and the main page
                    if (!article.isArticle() || article.isRedirect() || "Main Page".equals(article.getTitle())) continue;
                    // Skip non-text content types
                    String mimetype = article.getMimetype() != null ? article.getMimetype() : "";
                    if (!mimetype.startsWith("text/") && !mimetype.startsWith("application/xhtml")) continue;
                    // Read raw article data
                    byte[] rawBytes = article.getData();
                    if (rawBytes == null || rawBytes.length == 0) continue;

                    // Clean and normalize the article text
                    String cleanedText = cleanHtml(new String(rawBytes, "UTF-8"));

                    // Skip articles that are too short to be useful
                    if (cleanedText.length() < 100) continue;

                    // Create a JSON object for the article
                    JSONObject obj = new JSONObject();
                    obj.put("title", article.getTitle());
                    obj.put("text", cleanedText);

                    // Write the JSON object to the output file (as one line)
                    writer.write(obj.toString());
                    writer.newLine();

                    // Log progress for visibility
                    System.out.println("Title: " + article.getTitle() + ", Length: " + cleanedText.length());
                    datasetCount++;
                } catch (Exception e) {
                    System.err.println("Error processing article: " + article.getTitle() + " - " + e.getMessage());
                }
            }

            // Print completion summary
            System.out.println("Extraction complete. Total entries written: " + datasetCount);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
