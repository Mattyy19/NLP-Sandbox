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

public class FineTunedTransformer {

    public static List<Map<String, String>> loadDataset(String filename) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(filename));
        List<Map<String, String>> dataset = new ArrayList<>();
        for (String line : lines) {
            JSONObject obj = new JSONObject(line);
            Map<String, String> data = new HashMap<>();
            data.put("title", obj.optString("title", ""));
            data.put("text", obj.optString("text", ""));
            dataset.add(data);
        }
        return dataset;
    }

    public static void main(String[] args) throws IOException, TranslateException {
        if (args.length < 2) {
            System.out.println("Usage: java fine_tune.FineTunedSentenceTransformer <dataset-jsonl> <fine-tuned-model-dir>");
            return;
        }

        String datasetPath = args[0];
        String modelDir = args[1];

        List<Map<String, String>> dataset = loadDataset(datasetPath);
        Collections.shuffle(dataset, new Random());

        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optApplication(Application.NLP.TEXT_EMBEDDING)
                .optModelPath(Paths.get(modelDir))
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<String, float[]> model = ModelZoo.loadModel(criteria);
             Predictor<String, float[]> predictor = model.newPredictor()) {

            for (int i = 0; i < Math.min(3, dataset.size()); i++) {
                Map<String, String> entry = dataset.get(i);
                String title = entry.get("title");
                String text = entry.get("text");

                float[] titleEmbedding = predictor.predict(title);
                float[] textEmbedding = predictor.predict(text);

                double similarity = cosineSimilarity(titleEmbedding, textEmbedding);
                System.out.println("Title/Text similarity: " + similarity);
            }
        }
    }

    private static double cosineSimilarity(float[] a, float[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
