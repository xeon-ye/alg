package zju.util.expression;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2011-6-11
 */
public class Variable {
    private String name;
    private Number value;

    public Variable() {
    }

    public Variable(String name) {
        this.name = name;
    }

    public Variable(Number value) {
        this.value = value;
    }

    public Variable(String name, Number value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Number getValue() {
        return value;
    }

    public void setValue(Number value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return name;
    }
}
