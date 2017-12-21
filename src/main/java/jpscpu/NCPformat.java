package jpscpu;

/**
 * Created by IntelliJ IDEA.
 * Author: Fang Rui
 * Date: 17-12-9
 * Time: 下午5:37
 */
public class NCPformat {
    int nnz;	  /* number of nonzeros in the matrix */
    double[] nzval;  /* pointer to array of nonzero values, packed by column */
    int[] rowind;/* pointer to array of row indices of the nonzeros */
    /* Note: nzval[]/rowind[] always have the same length */
    int[] colbeg;/* colbeg[j] points to the beginning of column j in nzval[]
                     and rowind[]  */
    int[] colend;/* colend[j] points to one past the last element of column
             j in nzval[] and rowind[]  */
		  /* Note:
		     Zero-based indexing is used;
		     The consecutive columns of the nonzeros may not be
		     contiguous in storage, because the matrix has been
		     postmultiplied by a column permutation matrix. */

    public int getNnz() {
        return nnz;
    }

    public void setNnz(int nnz) {
        this.nnz = nnz;
    }

    public double[] getNzval() {
        return nzval;
    }

    public void setNzval(double[] nzval) {
        this.nzval = nzval;
    }

    public int[] getRowind() {
        return rowind;
    }

    public void setRowind(int[] rowind) {
        this.rowind = rowind;
    }

    public int[] getColbeg() {
        return colbeg;
    }

    public void setColbeg(int[] colbeg) {
        this.colbeg = colbeg;
    }

    public int[] getColend() {
        return colend;
    }

    public void setColend(int[] colend) {
        this.colend = colend;
    }
}
