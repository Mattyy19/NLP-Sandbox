public class Main {
    public static void main(String[] args) {

        CleanUpUserQuery preprocessor = new CleanUpUserQuery();

        String rawQuery = "   What is     the capital\n\nof France? ";
        String cleanQuery = preprocessor.cleanUpUserQuery(rawQuery);
        System.out.println(cleanQuery); // Output: "What is the capital of France?"
    }
}