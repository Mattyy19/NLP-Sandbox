import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class SemanticQueryHelperClient {
    //TODO: URL replaces with embedder endpoint when ready.
    private static final String URL = "http://localhost:8000/search";
    private final HttpClient CLIENT = HttpClient.newHttpClient();

    public String search(String query) {
        try {
            String safeQuery = query.replace("\"", "\\\"");
            String body = "{\"query\":\"" + safeQuery + "\"}";

            HttpRequest request = HttpRequest.newBuilder().
                    uri(URI.create(URL))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            HttpResponse<String> response = CLIENT.send(request, HttpResponse.BodyHandlers.ofString());

            System.out.println("Semantic Search response: " + response.body());
            //return value for now is JSON/TEXT
            return response.body();

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}