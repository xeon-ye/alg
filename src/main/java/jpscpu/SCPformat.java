package jpscpu;

/**
 * Created by IntelliJ IDEA.
 * Author: Fang Rui
 * Date: 17-12-9
 * Time: 下午5:33
 */
public class SCPformat {
    int nnz;	     /* number of nonzeros in the matrix */
    int nsuper;     /* number of supernodes */
    double[] nzval;       /* pointer to array of nonzero values, packed by column */
    int[] nzval_colbeg;/* nzval_colbeg[j] points to beginning of column j
              in nzval[] */
    int[] nzval_colend;/* nzval_colend[j] points to one past the last element
              of column j in nzval[] */
    int[] rowind;      /* pointer to array of compressed row indices of
              rectangular supernodes */
    int[] rowind_colbeg;/* rowind_colbeg[j] points to beginning of column j
              in rowind[] */
    int[] rowind_colend;/* rowind_colend[j] points to one past the last element
			  of column j in rowind[] */
    int[] col_to_sup;   /* col_to_sup[j] is the supernode number to which column
			  j belongs; mapping from column to supernode. */
    int[] sup_to_colbeg; /* sup_to_colbeg[s] points to the start of the s-th
			   supernode; mapping from supernode to column.*/
    int[] sup_to_colend; /* sup_to_colend[s] points to one past the end of the
			   s-th supernode; mapping from supernode number to
			   column.
		        e.g.: col_to_sup: 0 1 2 2 3 3 3 4 4 4 4 4 4 (ncol=12)
		              sup_to_colbeg: 0 1 2 4 7              (nsuper=4)
			      sup_to_colend: 1 2 4 7 12                    */
                     /* Note:
		        Zero-based indexing is used;
		        nzval_colptr[], rowind_colptr[], col_to_sup and
		        sup_to_col[] have ncol+1 entries, the last one
		        pointing beyond the last column.         */

    public int getNnz() {
        return nnz;
    }

    public void setNnz(int nnz) {
        this.nnz = nnz;
    }

    public int getNsuper() {
        return nsuper;
    }

    public void setNsuper(int nsuper) {
        this.nsuper = nsuper;
    }

    public double[] getNzval() {
        return nzval;
    }

    public void setNzval(double[] nzval) {
        this.nzval = nzval;
    }

    public int[] getNzval_colbeg() {
        return nzval_colbeg;
    }

    public void setNzval_colbeg(int[] nzval_colbeg) {
        this.nzval_colbeg = nzval_colbeg;
    }

    public int[] getNzval_colend() {
        return nzval_colend;
    }

    public void setNzval_colend(int[] nzval_colend) {
        this.nzval_colend = nzval_colend;
    }

    public int[] getRowind() {
        return rowind;
    }

    public void setRowind(int[] rowind) {
        this.rowind = rowind;
    }

    public int[] getRowind_colbeg() {
        return rowind_colbeg;
    }

    public void setRowind_colbeg(int[] rowind_colbeg) {
        this.rowind_colbeg = rowind_colbeg;
    }

    public int[] getRowind_colend() {
        return rowind_colend;
    }

    public void setRowind_colend(int[] rowind_colend) {
        this.rowind_colend = rowind_colend;
    }

    public int[] getCol_to_sup() {
        return col_to_sup;
    }

    public void setCol_to_sup(int[] col_to_sup) {
        this.col_to_sup = col_to_sup;
    }

    public int[] getSup_to_colbeg() {
        return sup_to_colbeg;
    }

    public void setSup_to_colbeg(int[] sup_to_colbeg) {
        this.sup_to_colbeg = sup_to_colbeg;
    }

    public int[] getSup_to_colend() {
        return sup_to_colend;
    }

    public void setSup_to_colend(int[] sup_to_colend) {
        this.sup_to_colend = sup_to_colend;
    }
}
