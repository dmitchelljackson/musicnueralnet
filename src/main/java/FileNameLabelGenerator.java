import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.net.URI;

public class FileNameLabelGenerator implements PathLabelGenerator {

    @Override
    public Writable getLabelForPath(String path) {
        Writable writable;
        if(path.contains("true")) {
            writable = new Text("0");
        } else {
            writable = new Text("1");
        }
        return writable;
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        Writable writable;
        if(uri.getPath().contains("true")) {
            writable = new Text("0");
        } else {
            writable = new Text("1");
        }
        return writable;
    }
}
