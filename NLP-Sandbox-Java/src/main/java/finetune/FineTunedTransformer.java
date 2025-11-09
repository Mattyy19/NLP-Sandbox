package fine_tune;

import ai.djl.Application;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.inference.Predictor;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.json.JSONObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Loads a fine-tuned NLP model and computes text similarity scores using embeddings.
 * @author Alain Mignot
 */
public class FineTunedTransformer {

    /**
     * Loads dataset entries from a JSONL file.
     * @param filename the dataset file path
     * @return a list of title-text maps
     * @throws IOException if reading the file fails
     */
    public static List<Map<String, String>> loadDataset(String filename) throws IOException {
        // Read all lines from the JSONL file (each line is a JSON object)
        List<String> lines = Files.readAllLines(Paths.get(filename));
        List<Map<String, String>> dataset = new ArrayList<>();

        // Parse each JSON line and extract title/text fields
        for (String line : lines) {
            JSONObject obj = new JSONObject(line);
            Map<String, String> data = new HashMap<>();
            data.put("title", obj.optString("title", ""));
            data.put("text", obj.optString("text", ""));
            dataset.add(data);
        }
        return dataset;
    }

    /**
     * Generates embeddings for dataset entries and prints cosine similarity scores.
     * @param args command line arguments: dataset-jsonl and fine-tuned-model-dir
     * @throws IOException if reading or model loading fails
     * @throws TranslateException if embedding generation fails
     */
    public static void main(String[] args) throws IOException, TranslateException {
        if (args.length < 2) {
            System.out.println("Usage: java fine_tune.FineTunedSentenceTransformer <dataset-jsonl> <fine-tuned-model-dir>");
            return;
        }

        // Extract file paths from arguments
        String datasetPath = args[0];
        String modelDir = args[1];

        // Load dataset and shuffle it for randomness
        List<Map<String, String>> dataset = loadDataset(datasetPath);
        Collections.shuffle(dataset, new Random());

        // Build criteria for loading the fine-tuned NLP model
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optApplication(Application.NLP.TEXT_EMBEDDING) // Specify NLP embedding task
                .optModelPath(Paths.get(modelDir)) // Set the path to the fine-tuned model
                .optProgress(new ProgressBar()) // Add progress bar for loading
                .build();

        // Load model and create predictor (auto-closes via try-with-resources)
        try (ZooModel<String, float[]> model = ModelZoo.loadModel(criteria);
             Predictor<String, float[]> predictor = model.newPredictor()) {

            // Limit output to first 3 dataset entries for demonstration
            for (int i = 0; i < Math.min(3, dataset.size()); i++) {
                Map<String, String> entry = dataset.get(i);
                String title = entry.get("title");
                String text = entry.get("text");

                // Generate embeddings for title and text
                float[] titleEmbedding = predictor.predict(title);
                float[] textEmbedding = predictor.predict(text);

                // Compute cosine similarity between embeddings
                double similarity = cosineSimilarity(titleEmbedding, textEmbedding);

                // Print similarity result
                System.out.println("Title/Text similarity: " + similarity);
            }
        }
    }

    private static double cosineSimilarity(float[] a, float[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;

        // Compute dot product and magnitude for both vectors
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        // Return normalized cosine similarity score
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
