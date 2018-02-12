import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

public class Listener implements IterationListener {

    private boolean isInvoked;

    @Override
    public boolean invoked() {
        return isInvoked;
    }

    @Override
    public void invoke() {
        isInvoked = !isInvoked;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        System.out.println("Iteration " + iteration + " completed");
    }
}
