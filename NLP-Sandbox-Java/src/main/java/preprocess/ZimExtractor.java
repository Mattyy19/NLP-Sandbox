package preprocess;

import net.kiwix.jzim.ZimFile;
import net.kiwix.jzim.ZimArticle;
import org.json.JSONObject;
import org.jsoup.Jsoup;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;

public class ZimExtractor {
    private static String cleanHtml(String rawHtml) {
        String text = Jsoup.parse(rawHtml).text();
        text = text.replaceAll("\\[[^\\]]*\\]", " ");
        text = text.replaceAll("\\s+", " ").trim();
        return text;
    }

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

            for (ZimArticle article : zim) {
                try {
                    if (!"C".equals(article.getNamespace())) continue;
                    if (!article.isArticle() || article.isRedirect() || "Main Page".equals(article.getTitle())) continue;
                    String mimetype = article.getMimetype() != null ? article.getMimetype() : "";
                    if (!mimetype.startsWith("text/") && !mimetype.startsWith("application/xhtml")) continue;
                    byte[] rawBytes = article.getData();
                    if (rawBytes == null || rawBytes.length == 0) continue;

                    String cleanedText = cleanHtml(new String(rawBytes, "UTF-8"));
                    if (cleanedText.length() < 100) continue;

                    JSONObject obj = new JSONObject();
                    obj.put("title", article.getTitle());
                    obj.put("text", cleanedText);
                    writer.write(obj.toString());
                    writer.newLine();

                    System.out.println("Title: " + article.getTitle() + ", Length: " + cleanedText.length());
                    datasetCount++;
                } catch (Exception e) {
                    System.err.println("Error processing article: " + article.getTitle() + " - " + e.getMessage());
                }
            }

            System.out.println("Extraction complete. Total entries written: " + datasetCount);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
