package zju.util.expression;

public interface InfixParser {

    /**
     * parse expression in infix form(natural form) to predefined objects
     * @param expr expression in infix form(natural form)
     * @return expression in infix form contains: Operator, Bracket, Number, or Variable.
     * @throws IllegalExpressionException runtime exception
     */
    Object[] parse(String expr) throws IllegalExpressionException;
}