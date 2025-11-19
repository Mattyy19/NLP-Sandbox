public class NlpServer
{
    private final SemanticQueryHelperClient SC = new SemanticQueryHelperClient();

    public void runSemanticProcessing(String q)
    {
        System.out.println("Received of Kiwix Query: "+ q);

        String result = SC.search(q);

        System.out.println("Result of Kiwix Query: "+ result);
    }

    public static void main (String[] args){
        NlpServer server = new NlpServer();
        server.runSemanticProcessing("test query from main()");
    }
}
