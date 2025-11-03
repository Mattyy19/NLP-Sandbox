import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestCleanQuery {

    @Test
    public void testNullInput() {
        CleanUpUserQuery preprocessor = new CleanUpUserQuery();
        String result = preprocessor.cleanUpUserQuery(null);
        assertEquals("", result, "Null input should return empty string");
    }

    @Test
    public void testWhitespaceOnly() {
        CleanUpUserQuery preprocessor = new CleanUpUserQuery();
        String result = preprocessor.cleanUpUserQuery("   ");
        assertEquals("", result, "Whitespace-only input should return empty string");
    }

    @Test
    public void testComplexQuery() {
        CleanUpUserQuery preprocessor = new CleanUpUserQuery();
        String result = preprocessor.cleanUpUserQuery("   What is     the capital\n\nof France? ");
        assertEquals("what is the capital of france?", result, "Complex query with multiple issues should be cleaned properly");
    }

    @Test
    public void testQueryExceedingMaxLength() {
        CleanUpUserQuery preprocessor = new CleanUpUserQuery();
        String longQuery = "a".repeat(1500);
        String result = preprocessor.cleanUpUserQuery(longQuery);
        assertEquals(1000, result.length(), "Query exceeding max length should be truncated to 1000 characters");
    }
}