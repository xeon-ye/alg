/*
 * -- SuperLU MT for JNI --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * and Xerox Palo Alto Research Center.
 * This is a revised version of JNI, a C++ interface of SuperLU, revised by Rui Fang of Zhejiang University.
 * Rui Fang, Zhejiang University,
 * and Smart Grid Operation and Optimization Laboratory.
 * November 30, 2017
 *
 */

#include "slu_mt_ddefs.h"
#include "sstream"
#include <iostream>
#include "jpscpu_LinearSolverMT.h"

using namespace std;

template <class T>
int length(T &data)
{
    return sizeof(data) / sizeof(data[0]);
}

// use simple driver PDGSSV to solve a linear system one time.
double *solve0_mt(int m, int n, int nnz, double *a, int *asub, int *xa, double *b, int nprocs)
{
    SuperMatrix A;
    int_t *perm_r; /* row permutations from partial pivoting */
    int_t *perm_c; /* column permutation vector */
    SuperMatrix L; /* factor L */
    SuperMatrix U; /* factor U */
    SuperMatrix B;
    int_t nrhs, info;
    int_t permc_spec;
    double *rhs, *result;

    nrhs = 1;

    dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

    if (!(rhs = doubleMalloc(m * nrhs)))
        SUPERLU_ABORT("Malloc fails for rhs[].");
    if (!(result = doubleMalloc(m * nrhs)))
        SUPERLU_ABORT("Malloc fails for result[].");
    dCreate_Dense_Matrix(&B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);

    for (int i = 0; i < m; ++i)
        rhs[i] = b[i];

    if (!(perm_r = intMalloc(m)))
        SUPERLU_ABORT("Malloc fails for perm_r[].");
    if (!(perm_c = intMalloc(n)))
        SUPERLU_ABORT("Malloc fails for perm_c[].");

    /*
     * Get column permutation vector perm_c[], according to permc_spec:
     *   permc_spec = 0: natural ordering 
     *   permc_spec = 1: minimum degree ordering on structure of A'*A
     *   permc_spec = 2: minimum degree ordering on structure of A'+A
     *   permc_spec = 3: approximate minimum degree for unsymmetric matrices
     */
    permc_spec = 1;
    get_perm_c(permc_spec, &A, perm_c);

    pdgssv(nprocs, &A, perm_c, perm_r, &L, &U, &B, &info);

    if (info == 0)
    {
        for (int i = 0; i < n; i++)
            result[i] = ((double *)((DNformat *)B.Store)->nzval)[i];
    }
    else
    {
        printf("pdgssv() error returns INFO= %d\n", info);
    }

    SUPERLU_FREE(rhs);
    SUPERLU_FREE(perm_r);
    SUPERLU_FREE(perm_c);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_SCP(&L);
    Destroy_CompCol_NCP(&U);

    return result;
}

//use PDGSSVX to solve a linear system.
double *solve1_mt(int m, int n, int nnz, double *a, int *asub, int *xa, double *b, int nprocs)
{
    SuperMatrix A, L, U;
    SuperMatrix B, X;
    fact_t fact;
    trans_t trans;
    yes_no_t refact, usepr;
    equed_t equed;
    int_t *perm_c; /* column permutation vector */
    int_t *perm_r; /* row permutations from partial pivoting */
    void *work;
    superlumt_options_t superlumt_options;
    int_t info, lwork, nrhs, ldx, panel_size, relax;
    int_t permc_spec, i;
    double *rhsb, *rhsx, *result;
    double *R, *C;
    double *ferr, *berr;
    double u, drop_tol, rpg, rcond;
    superlu_memusage_t superlu_memusage;

    /* Default parameters to control factorization. */
    fact = EQUILIBRATE;
    trans = NOTRANS;
    equed = NOEQUIL;
    refact = NO;
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    u = 1.0;
    usepr = NO;
    drop_tol = 0.0;
    lwork = 0;
    nrhs = 1;

    if (lwork > 0)
    {
        work = SUPERLU_MALLOC(lwork);
        if (!work)
        {
            SUPERLU_ABORT("DLINSOLX: cannot allocate work[]");
        }
    }

    dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

    if (!(rhsb = doubleMalloc(m * nrhs)))
        SUPERLU_ABORT("Malloc fails for rhsb[].");
    if (!(rhsx = doubleMalloc(m * nrhs)))
        SUPERLU_ABORT("Malloc fails for rhsx[].");
    if (!(result = doubleMalloc(n * nrhs)))
        SUPERLU_ABORT("Malloc fails for result[].");
    for (i = 0; i < m; ++i)
    {
        rhsb[i] = b[i];
        rhsx[i] = b[i];
    }
    dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m)))
        SUPERLU_ABORT("Malloc fails for perm_r[].");
    if (!(perm_c = intMalloc(n)))
        SUPERLU_ABORT("Malloc fails for perm_c[].");
    if (!(R = (double *)SUPERLU_MALLOC(A.nrow * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for R[].");
    if (!(C = (double *)SUPERLU_MALLOC(A.ncol * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for C[].");
    if (!(ferr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for ferr[].");
    if (!(berr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for berr[].");

    /*
     * Get column permutation vector perm_c[], according to permc_spec:
     *   permc_spec = 0: natural ordering 
     *   permc_spec = 1: minimum degree ordering on structure of A'*A
     *   permc_spec = 2: minimum degree ordering on structure of A'+A
     *   permc_spec = 3: approximate minimum degree for unsymmetric matrices
     */
    permc_spec = 1;
    get_perm_c(permc_spec, &A, perm_c);

    superlumt_options.nprocs = nprocs;
    superlumt_options.fact = fact;
    superlumt_options.trans = trans;
    superlumt_options.refact = refact;
    superlumt_options.panel_size = panel_size;
    superlumt_options.relax = relax;
    superlumt_options.usepr = usepr;
    superlumt_options.drop_tol = drop_tol;
    superlumt_options.diag_pivot_thresh = u;
    superlumt_options.SymmetricMode = NO;
    superlumt_options.PrintStat = NO;
    superlumt_options.perm_c = perm_c;
    superlumt_options.perm_r = perm_r;
    superlumt_options.work = work;
    superlumt_options.lwork = lwork;
    if (!(superlumt_options.etree = intMalloc(n)))
        SUPERLU_ABORT("Malloc fails for etree[].");
    if (!(superlumt_options.colcnt_h = intMalloc(n)))
        SUPERLU_ABORT("Malloc fails for colcnt_h[].");
    if (!(superlumt_options.part_super_h = intMalloc(n)))
        SUPERLU_ABORT("Malloc fails for colcnt_h[].");

    /* 
     * Solve the system and compute the condition number
     * and error bounds using pdgssvx.
     */
    pdgssvx(nprocs, &superlumt_options, &A, perm_c, perm_r,
            &equed, R, C, &L, &U, &B, &X, &rpg, &rcond,
            ferr, berr, &superlu_memusage, &info);

    if (info == 0 || info == n + 1)
    {
        for (int i = 0; i < n; i++)
            result[i] = ((double *)((DNformat *)X.Store)->nzval)[i];
    }
    else if (info > 0 && lwork == -1)
    {
        printf("** Estimated memory: " IFMT " bytes\n", info - n);
    }

    SUPERLU_FREE(rhsb);
    SUPERLU_FREE(rhsx);
    SUPERLU_FREE(perm_r);
    SUPERLU_FREE(perm_c);
    SUPERLU_FREE(R);
    SUPERLU_FREE(C);
    SUPERLU_FREE(ferr);
    SUPERLU_FREE(berr);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    SUPERLU_FREE(superlumt_options.etree);
    SUPERLU_FREE(superlumt_options.colcnt_h);
    SUPERLU_FREE(superlumt_options.part_super_h);
    if (lwork == 0)
    {
        Destroy_SuperNode_SCP(&L);
        Destroy_CompCol_NCP(&U);
    }
    else if (lwork > 0)
    {
        SUPERLU_FREE(work);
    }
    return result;
}

/* use PDGSSVX to solve systems repeatedly
 * the column permutation vector perm_c is computed once. 
 * The following data structures will be reused in the subsequent call to PDGSSVX: 
 * perm_c, etree, colcnt_h, part_super_h
 * with the same sparsity pattern of matrix A.
 */
double *solve2_mt(int m, int n, int nnz, double *a, int *asub, int *xa, double *b, int nprocs, int *perm_c, int *etree, int *colcnt_h, int *part_super_h, SuperMatrix *L, SuperMatrix *U)
{
    SuperMatrix A;
    SuperMatrix B, X;
    fact_t fact;
    trans_t trans;
    yes_no_t refact, usepr;
    equed_t equed;
    int_t *perm_r; /* row permutations from partial pivoting */
    void *work;
    superlumt_options_t superlumt_options;
    int_t info, lwork, nrhs, ldx, panel_size, relax;
    int_t permc_spec, i;
    double *rhsb, *rhsx, *result;
    double *R, *C;
    double *ferr, *berr;
    double u, drop_tol, rpg, rcond;
    superlu_memusage_t superlu_memusage;

    /* Default parameters to control factorization. */
    fact = EQUILIBRATE;
    trans = NOTRANS;
    equed = NOEQUIL;
    refact = NO;
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    u = 1.0;
    usepr = NO;
    drop_tol = 0.0;
    lwork = 0;
    nrhs = 1;

    if (lwork > 0)
    {
        work = SUPERLU_MALLOC(lwork);
        if (!work)
        {
            SUPERLU_ABORT("DLINSOLX: cannot allocate work[]");
        }
    }

    dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

    if (!(rhsb = doubleMalloc(m * nrhs)))
        SUPERLU_ABORT("Malloc fails for rhsb[].");
    if (!(rhsx = doubleMalloc(m * nrhs)))
        SUPERLU_ABORT("Malloc fails for rhsx[].");
    if (!(result = doubleMalloc(n * nrhs)))
        SUPERLU_ABORT("Malloc fails for result[].");
    for (i = 0; i < m; ++i)
    {
        rhsb[i] = b[i];
        rhsx[i] = b[i];
    }
    dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m)))
        SUPERLU_ABORT("Malloc fails for perm_r[].");
    if (!(R = (double *)SUPERLU_MALLOC(A.nrow * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for R[].");
    if (!(C = (double *)SUPERLU_MALLOC(A.ncol * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for C[].");
    if (!(ferr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for ferr[].");
    if (!(berr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for berr[].");

    /*
     * Get column permutation vector perm_c[], according to permc_spec:
     *   permc_spec = 0: natural ordering 
     *   permc_spec = 1: minimum degree ordering on structure of A'*A
     *   permc_spec = 2: minimum degree ordering on structure of A'+A
     *   permc_spec = 3: approximate minimum degree for unsymmetric matrices
     */
    permc_spec = 1;
    get_perm_c(permc_spec, &A, perm_c);

    superlumt_options.nprocs = nprocs;
    superlumt_options.fact = fact;
    superlumt_options.trans = trans;
    superlumt_options.refact = refact;
    superlumt_options.panel_size = panel_size;
    superlumt_options.relax = relax;
    superlumt_options.usepr = usepr;
    superlumt_options.drop_tol = drop_tol;
    superlumt_options.diag_pivot_thresh = u;
    superlumt_options.SymmetricMode = NO;
    superlumt_options.PrintStat = NO;
    superlumt_options.perm_c = perm_c;
    superlumt_options.perm_r = perm_r;
    superlumt_options.work = work;
    superlumt_options.lwork = lwork;
    superlumt_options.etree = etree;
    superlumt_options.colcnt_h = colcnt_h;
    superlumt_options.part_super_h = part_super_h;

    /* ------------------------------------------------------------
       WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME: AX = B
       ------------------------------------------------------------*/

    pdgssvx(nprocs, &superlumt_options, &A, perm_c, perm_r,
            &equed, R, C, L, U, &B, &X, &rpg, &rcond,
            ferr, berr, &superlu_memusage, &info);

    if (info == 0 || info == n + 1)
    {
        for (int i = 0; i < n; i++)
            result[i] = ((double *)((DNformat *)X.Store)->nzval)[i];
    }
    else if (info > 0 && lwork == -1)
    {
        printf("** Estimated memory: " IFMT " bytes\n", info - n);
    }

    SUPERLU_FREE(rhsb);
    SUPERLU_FREE(rhsx);
    SUPERLU_FREE(perm_r);
    SUPERLU_FREE(R);
    SUPERLU_FREE(C);
    SUPERLU_FREE(ferr);
    SUPERLU_FREE(berr);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);

    if (lwork > 0)
    {
        SUPERLU_FREE(work);
    }
    return result;
}

/* use PDGSSVX to solve systems repeatedly
 * the column permutation vector perm_c is computed once. 
 * The following data structures will be reused in the subsequent call to PDGSSVX: 
 * perm_c, etree, colcnt_h, part_super_h
 * with the same sparsity pattern of matrix A.
 */
double *solve3_mt(int m, int n, int nnz, double *a, int *asub, int *xa, double *b, int nprocs, int *perm_c, int *etree, int *colcnt_h, int *part_super_h, SuperMatrix *L, SuperMatrix *U)
{
    SuperMatrix A;
    SuperMatrix B, X;
    fact_t fact;
    trans_t trans;
    yes_no_t refact, usepr;
    equed_t equed;
    int_t *perm_r; /* row permutations from partial pivoting */
    void *work;
    superlumt_options_t superlumt_options;
    int_t info, lwork, nrhs, ldx, panel_size, relax;
    int_t permc_spec, i;
    double *rhsb, *rhsx, *result;
    double *R, *C;
    double *ferr, *berr;
    double u, drop_tol, rpg, rcond;
    superlu_memusage_t superlu_memusage;

    /* Default parameters to control factorization. */
    fact = EQUILIBRATE;
    trans = NOTRANS;
    equed = NOEQUIL;
    refact = NO;
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    u = 1.0;
    usepr = NO;
    drop_tol = 0.0;
    lwork = 0;
    nrhs = 1;

    if (lwork > 0)
    {
        work = SUPERLU_MALLOC(lwork);
        if (!work)
        {
            SUPERLU_ABORT("DLINSOLX: cannot allocate work[]");
        }
    }

    dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

    if (!(rhsb = doubleMalloc(m * nrhs)))
        SUPERLU_ABORT("Malloc fails for rhsb[].");
    if (!(rhsx = doubleMalloc(m * nrhs)))
        SUPERLU_ABORT("Malloc fails for rhsx[].");
    if (!(result = doubleMalloc(n * nrhs)))
        SUPERLU_ABORT("Malloc fails for result[].");
    for (i = 0; i < m; ++i)
    {
        rhsb[i] = b[i];
        rhsx[i] = b[i];
    }
    dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m)))
        SUPERLU_ABORT("Malloc fails for perm_r[].");
    if (!(R = (double *)SUPERLU_MALLOC(A.nrow * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for R[].");
    if (!(C = (double *)SUPERLU_MALLOC(A.ncol * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for C[].");
    if (!(ferr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for ferr[].");
    if (!(berr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
        SUPERLU_ABORT("SUPERLU_MALLOC fails for berr[].");

    /*
    * Get column permutation vector perm_c[], according to permc_spec:
    *   permc_spec = 0: natural ordering
    *   permc_spec = 1: minimum degree ordering on structure of A'*A
    *   permc_spec = 2: minimum degree ordering on structure of A'+A
    *   permc_spec = 3: approximate minimum degree for unsymmetric matrices
    */
    //    permc_spec = 1;
    //    get_perm_c(permc_spec, &A, perm_c);

    superlumt_options.nprocs = nprocs;
    superlumt_options.fact = fact;
    superlumt_options.trans = trans;
    superlumt_options.refact = refact;
    superlumt_options.panel_size = panel_size;
    superlumt_options.relax = relax;
    superlumt_options.usepr = usepr;
    superlumt_options.drop_tol = drop_tol;
    superlumt_options.diag_pivot_thresh = u;
    superlumt_options.SymmetricMode = NO;
    superlumt_options.PrintStat = YES;
    superlumt_options.perm_c = perm_c;
    superlumt_options.perm_r = perm_r;
    superlumt_options.work = work;
    superlumt_options.lwork = lwork;
    superlumt_options.etree = etree;
    superlumt_options.colcnt_h = colcnt_h;
    superlumt_options.part_super_h = part_super_h;

    /* ------------------------------------------------------------
       NOW WE SOLVE ANOTHER LINEAR SYSTEM: A*X = B
       ONLY THE SPARSITY PATTERN OF A IS THE SAME AS THAT OF THE FIRST TIME.
       ------------------------------------------------------------*/

    superlumt_options.refact = YES;
    pdgssvx(nprocs, &superlumt_options, &A, perm_c, perm_r,
            &equed, R, C, L, U, &B, &X, &rpg, &rcond,
            ferr, berr, &superlu_memusage, &info);

    if (info == 0 || info == n + 1)
    {
        for (int i = 0; i < n; i++)
            result[i] = ((double *)((DNformat *)X.Store)->nzval)[i];
    }
    else if (info > 0 && lwork == -1)
    {
        printf("** Estimated memory: " IFMT " bytes\n", info - n);
    }
    SUPERLU_FREE(rhsb);
    SUPERLU_FREE(rhsx);
    SUPERLU_FREE(perm_r);
    SUPERLU_FREE(R);
    SUPERLU_FREE(C);
    SUPERLU_FREE(ferr);
    SUPERLU_FREE(berr);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    // if (lwork == 0)
    // {
    //     Destroy_SuperNode_SCP(L);
    //     Destroy_CompCol_NCP(U);
    // }
    if (lwork > 0)
    {
        SUPERLU_FREE(work);
    }
    return result;
}
//======================== jni方法开始 ===========================
JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolverMT_solve0_1mt(JNIEnv *env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, jdoubleArray b, jint nprocs)
{
    jint *jasub, *jxa;
    jdouble *ja, *jb;
    double sb[n];
    double *sa;
    int *sasub, *sxa;

    if (!(sa = doubleMalloc(nnz)))
        SUPERLU_ABORT("Malloc fails for a[].");
    if (!(sasub = intMalloc(nnz)))
        SUPERLU_ABORT("Malloc fails for asub[].");
    if (!(sxa = intMalloc(n + 1)))
        SUPERLU_ABORT("Malloc fails for xa[].");

    jasub = env->GetIntArrayElements(asub, 0);
    jxa = env->GetIntArrayElements(xa, 0);
    ja = env->GetDoubleArrayElements(a, 0);
    jb = env->GetDoubleArrayElements(b, 0);

    for (int i = 0; i < nnz; i++)
    {
        sa[i] = (double)ja[i];
        sasub[i] = (int)jasub[i];
    }
    for (int j = 0; j < n; j++)
    {
        sb[j] = (double)jb[j];
    }
    for (int k = 0; k < n + 1; k++)
    {
        sxa[k] = (int)jxa[k];
    }
    double *res = solve0_mt(m, n, nnz, sa, sasub, sxa, sb, nprocs);

    env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
    env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);
    env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);
    env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);

    env->SetDoubleArrayRegion(b, 0, n, (const jdouble *)res);

    SUPERLU_FREE(res);
    return b;
}

JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolverMT_solve1_1mt(JNIEnv *env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, jdoubleArray b, jint nprocs)
{
    jint *jasub, *jxa;
    jdouble *ja, *jb;
    double sb[n];
    double *sa;
    int *sasub, *sxa;

    if (!(sa = doubleMalloc(nnz)))
        SUPERLU_ABORT("Malloc fails for a[].");
    if (!(sasub = intMalloc(nnz)))
        SUPERLU_ABORT("Malloc fails for asub[].");
    if (!(sxa = intMalloc(n + 1)))
        SUPERLU_ABORT("Malloc fails for xa[].");

    jasub = env->GetIntArrayElements(asub, 0);
    jxa = env->GetIntArrayElements(xa, 0);
    ja = env->GetDoubleArrayElements(a, 0);
    jb = env->GetDoubleArrayElements(b, 0);

    for (int i = 0; i < nnz; i++)
    {
        sa[i] = (double)ja[i];
        sasub[i] = (int)jasub[i];
    }
    for (int j = 0; j < n; j++)
    {
        sb[j] = (double)jb[j];
    }
    for (int k = 0; k < n + 1; k++)
    {
        sxa[k] = (int)jxa[k];
    }
    double *res = solve1_mt(m, n, nnz, sa, sasub, sxa, sb, nprocs);

    env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
    env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);
    env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);
    env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);

    env->SetDoubleArrayRegion(b, 0, n, (const jdouble *)res);

    SUPERLU_FREE(res);
    return b;
}

JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolverMT_solve2_1mt(JNIEnv *env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, jdoubleArray b, jint nprocs, jintArray perm_c, jintArray etree, jintArray colcnt_h, jintArray part_super_h, jobject L, jobject U)
{
    jint *jasub, *jxa;
    jdouble *ja, *jb;

    double sb[n];
    int sperm_c[n];
    int setree[n];
    int scolcnt_h[n];
    int spart_super_h[n];

    double *sa;
    int *sasub, *sxa;
    SuperMatrix sL, sU;

    if (!(sa = doubleMalloc(nnz)))
        SUPERLU_ABORT("Malloc fails for a[].");
    if (!(sasub = intMalloc(nnz)))
        SUPERLU_ABORT("Malloc fails for asub[].");
    if (!(sxa = intMalloc(n + 1)))
        SUPERLU_ABORT("Malloc fails for xa[].");

    jasub = env->GetIntArrayElements(asub, 0);
    jxa = env->GetIntArrayElements(xa, 0);
    ja = env->GetDoubleArrayElements(a, 0);
    jb = env->GetDoubleArrayElements(b, 0);

    for (int i = 0; i < nnz; i++)
    {
        sa[i] = (double)ja[i];
        sasub[i] = (int)jasub[i];
    }
    for (int j = 0; j < n; j++)
    {
        sb[j] = (double)jb[j];
    }
    for (int k = 0; k < n + 1; k++)
    {
        sxa[k] = (int)jxa[k];
    }

    double *res = solve2_mt(m, n, nnz, sa, sasub, sxa, sb, nprocs, sperm_c, setree, scolcnt_h, spart_super_h, &sL, &sU);

    env->SetIntArrayRegion(perm_c, 0, n, (const jint *)sperm_c);
    env->SetIntArrayRegion(etree, 0, n, (const jint *)setree);
    env->SetIntArrayRegion(colcnt_h, 0, n, (const jint *)scolcnt_h);
    env->SetIntArrayRegion(part_super_h, 0, n, (const jint *)spart_super_h);

    env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
    env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);
    env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);
    env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);

    SCPformat *Lstore = (SCPformat *)sL.Store;
    NCPformat *Ustore = (NCPformat *)sU.Store;

    jclass clazz = env->FindClass("jpscpu/SCPformat");
    jmethodID methodId = env->GetMethodID(clazz, "setNnz", "(I)V");
    env->CallVoidMethod(L, methodId, Lstore->nnz);

    methodId = env->GetMethodID(clazz, "setNsuper", "(I)V");
    env->CallVoidMethod(L, methodId, Lstore->nsuper);

    jdoubleArray nzval = env->NewDoubleArray(2 * Lstore->nnz);
    env->SetDoubleArrayRegion(nzval, 0, 2 * Lstore->nnz, (const jdouble *)Lstore->nzval);
    methodId = env->GetMethodID(clazz, "setNzval", "([D)V");
    env->CallVoidMethod(L, methodId, nzval);
    env->DeleteLocalRef(nzval);
    // printf("%d",(int)(2 * nnz));

    jintArray nzval_colbeg = env->NewIntArray(n + 1);
    env->SetIntArrayRegion(nzval_colbeg, 0, n + 1, (const jint *)Lstore->nzval_colbeg);
    methodId = env->GetMethodID(clazz, "setNzval_colbeg", "([I)V");
    env->CallVoidMethod(L, methodId, nzval_colbeg);
    env->DeleteLocalRef(nzval_colbeg);

    jintArray nzval_colend = env->NewIntArray(n);
    env->SetIntArrayRegion(nzval_colend, 0, n, (const jint *)Lstore->nzval_colend);
    methodId = env->GetMethodID(clazz, "setNzval_colend", "([I)V");
    env->CallVoidMethod(L, methodId, nzval_colend);
    env->DeleteLocalRef(nzval_colend);

    jintArray rowind = env->NewIntArray(2 * Lstore->nnz);
    env->SetIntArrayRegion(rowind, 0, 2 * Lstore->nnz, (const jint *)Lstore->rowind);
    methodId = env->GetMethodID(clazz, "setRowind", "([I)V");
    env->CallVoidMethod(L, methodId, rowind);
    env->DeleteLocalRef(rowind);

    jintArray rowind_colbeg = env->NewIntArray(n + 1);
    env->SetIntArrayRegion(rowind_colbeg, 0, n + 1, (const jint *)Lstore->rowind_colbeg);
    methodId = env->GetMethodID(clazz, "setRowind_colbeg", "([I)V");
    env->CallVoidMethod(L, methodId, rowind_colbeg);
    env->DeleteLocalRef(rowind_colbeg);

    jintArray rowind_colend = env->NewIntArray(n);
    env->SetIntArrayRegion(rowind_colend, 0, n, (const jint *)Lstore->rowind_colend);
    methodId = env->GetMethodID(clazz, "setRowind_colend", "([I)V");
    env->CallVoidMethod(L, methodId, rowind_colend);
    env->DeleteLocalRef(rowind_colend);

    jintArray col_to_sup = env->NewIntArray(n + 1);
    env->SetIntArrayRegion(col_to_sup, 0, n + 1, (const jint *)Lstore->col_to_sup);
    methodId = env->GetMethodID(clazz, "setCol_to_sup", "([I)V");
    env->CallVoidMethod(L, methodId, col_to_sup);
    env->DeleteLocalRef(col_to_sup);

    jintArray sup_to_colbeg = env->NewIntArray(n + 1);
    env->SetIntArrayRegion(sup_to_colbeg, 0, n + 1, (const jint *)Lstore->sup_to_colbeg);
    methodId = env->GetMethodID(clazz, "setSup_to_colbeg", "([I)V");
    env->CallVoidMethod(L, methodId, sup_to_colbeg);
    env->DeleteLocalRef(sup_to_colbeg);

    jintArray sup_to_colend = env->NewIntArray(n);
    env->SetIntArrayRegion(sup_to_colend, 0, n, (const jint *)Lstore->sup_to_colend);
    methodId = env->GetMethodID(clazz, "setSup_to_colend", "([I)V");
    env->CallVoidMethod(L, methodId, sup_to_colend);
    env->DeleteLocalRef(sup_to_colend);

    clazz = env->FindClass("jpscpu/NCPformat");
    methodId = env->GetMethodID(clazz, "setNnz", "(I)V");
    env->CallVoidMethod(U, methodId, Ustore->nnz);

    jdoubleArray nzval_U = env->NewDoubleArray(2 * Ustore->nnz);
    env->SetDoubleArrayRegion(nzval_U, 0, 2 * Ustore->nnz, (const jdouble *)Ustore->nzval);
    methodId = env->GetMethodID(clazz, "setNzval", "([D)V");
    env->CallVoidMethod(U, methodId, nzval_U);
    env->DeleteLocalRef(nzval_U);

    jintArray rowind_U = env->NewIntArray(2 * Ustore->nnz);
    env->SetIntArrayRegion(rowind_U, 0, 2 * Ustore->nnz, (const jint *)Ustore->rowind);
    methodId = env->GetMethodID(clazz, "setRowind", "([I)V");
    env->CallVoidMethod(U, methodId, rowind_U);
    env->DeleteLocalRef(rowind_U);

    jintArray colbeg = env->NewIntArray(n + 1);
    env->SetIntArrayRegion(colbeg, 0, n + 1, (const jint *)Ustore->colbeg);
    methodId = env->GetMethodID(clazz, "setColbeg", "([I)V");
    env->CallVoidMethod(U, methodId, colbeg);
    env->DeleteLocalRef(colbeg);

    jintArray colend = env->NewIntArray(n);
    env->SetIntArrayRegion(colend, 0, n, (const jint *)Ustore->colend);
    methodId = env->GetMethodID(clazz, "setColend", "([I)V");
    env->CallVoidMethod(U, methodId, colend);
    env->DeleteLocalRef(colend);

    env->SetDoubleArrayRegion(b, 0, n, (const jdouble *)res);

    Destroy_SuperNode_SCP(&sL);
    Destroy_CompCol_NCP(&sU);
    SUPERLU_FREE(res);
    return b;
}

JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolverMT_solve3_1mt(
    JNIEnv *env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, jdoubleArray b,
    jint nprocs, jintArray perm_c, jintArray etree, jintArray colcnt_h, jintArray part_super_h,
    jint nnz_L, jint nsuper_L, jdoubleArray nzval_L, jintArray nzval_colbeg_L, jintArray nzval_colend_L, jintArray rowind_L,
    jintArray rowind_colbeg_L, jintArray rowind_colend_L, jintArray col_to_sup_L, jintArray sup_to_colbeg_L, jintArray sup_to_colend_L,
    jint nnz_U, jdoubleArray nzval_U, jintArray rowind_U, jintArray colbeg_U, jintArray colend_U)
{
    {
        jint *jasub, *jxa, *jperm_c, *jetree, *jcolcnt_h, *jpart_super_h;
        jint *jnzval_colbeg_L, *jnzval_colend_L, *jrowind_L, *jrowind_colbeg_L, *jrowind_colend_L, *jcol_to_sup_L, *jsup_to_colbeg_L, *jsup_to_colend_L;
        jint *jrowind_U, *jcolbeg_U, *jcolend_U;
        jdouble *ja, *jb, *jnzval_L, *jnzval_U;
        double sb[n];
        int sperm_c[n], setree[n], scolcnt_h[n], spart_super_h[n];
        int *snzval_colbeg_L, *snzval_colend_L, *srowind_L, *srowind_colbeg_L, *srowind_colend_L, *scol_to_sup_L, *ssup_to_colbeg_L, *ssup_to_colend_L, *srowind_U, *scolbeg_U, *scolend_U;
        double *sa, *snzval_L, *snzval_U;
        int *sasub, *sxa;

        ja = env->GetDoubleArrayElements(a, 0);
        jb = env->GetDoubleArrayElements(b, 0);
        jnzval_L = env->GetDoubleArrayElements(nzval_L, 0);
        jnzval_U = env->GetDoubleArrayElements(nzval_U, 0);

        jasub = env->GetIntArrayElements(asub, 0);
        jxa = env->GetIntArrayElements(xa, 0);
        jperm_c = env->GetIntArrayElements(perm_c, 0);
        jetree = env->GetIntArrayElements(etree, 0);
        jcolcnt_h = env->GetIntArrayElements(colcnt_h, 0);
        jpart_super_h = env->GetIntArrayElements(part_super_h, 0);

        jnzval_colbeg_L = env->GetIntArrayElements(nzval_colbeg_L, 0);
        jnzval_colend_L = env->GetIntArrayElements(nzval_colend_L, 0);
        jrowind_L = env->GetIntArrayElements(rowind_L, 0);
        jrowind_colbeg_L = env->GetIntArrayElements(rowind_colbeg_L, 0);
        jrowind_colend_L = env->GetIntArrayElements(rowind_colend_L, 0);
        jcol_to_sup_L = env->GetIntArrayElements(col_to_sup_L, 0);
        jsup_to_colbeg_L = env->GetIntArrayElements(sup_to_colbeg_L, 0);
        jsup_to_colend_L = env->GetIntArrayElements(sup_to_colend_L, 0);

        jrowind_U = env->GetIntArrayElements(rowind_U, 0);
        jcolbeg_U = env->GetIntArrayElements(colbeg_U, 0);
        jcolend_U = env->GetIntArrayElements(colend_U, 0);

        if (!(sa = doubleMalloc(nnz)))
            SUPERLU_ABORT("Malloc fails for a[].");
        if (!(sasub = intMalloc(nnz)))
            SUPERLU_ABORT("Malloc fails for asub[].");
        if (!(sxa = intMalloc(n + 1)))
            SUPERLU_ABORT("Malloc fails for xa[].");

        if (!(snzval_L = doubleMalloc(2 * nnz_L)))
            SUPERLU_ABORT("Malloc fails for nzval_L[].");
        if (!(snzval_U = doubleMalloc(2 * nnz_U)))
            SUPERLU_ABORT("Malloc fails for nzval_U[].");

        if (!(snzval_colbeg_L = intMalloc(n + 1)))
            SUPERLU_ABORT("Malloc fails for nzval_colbeg_L[].");
        if (!(snzval_colend_L = intMalloc(n)))
            SUPERLU_ABORT("Malloc fails for nzval_colend_L[].");
        if (!(srowind_L = intMalloc(2 * nnz_L)))
            SUPERLU_ABORT("Malloc fails for rowind_L[].");
        if (!(srowind_colbeg_L = intMalloc(n + 1)))
            SUPERLU_ABORT("Malloc fails for rowind_colbeg_L[].");
        if (!(srowind_colend_L = intMalloc(n)))
            SUPERLU_ABORT("Malloc fails for rowind_colend_L[].");
        if (!(scol_to_sup_L = intMalloc(n + 1)))
            SUPERLU_ABORT("Malloc fails for col_to_sup_L[].");
        if (!(ssup_to_colbeg_L = intMalloc(n + 1)))
            SUPERLU_ABORT("Malloc fails for sup_to_colbeg_L[].");
        if (!(ssup_to_colend_L = intMalloc(n)))
            SUPERLU_ABORT("Malloc fails for sup_to_colend_L[].");

        if (!(srowind_U = intMalloc(2 * nnz_U)))
            SUPERLU_ABORT("Malloc fails for rowind_U[].");
        if (!(scolbeg_U = intMalloc(n + 1)))
            SUPERLU_ABORT("Malloc fails for colbeg_U[].");
        if (!(scolend_U = intMalloc(n)))
            SUPERLU_ABORT("Malloc fails for colend_U[].");

        for (int i = 0; i < nnz; i++)
        {
            sa[i] = (double)ja[i];
            sasub[i] = (int)jasub[i];
        }
        for (int i = 0; i < n; i++)
        {
            sb[i] = (double)jb[i];
            sperm_c[i] = (int)jperm_c[i];
            setree[i] = (int)jetree[i];
            scolcnt_h[i] = (int)jcolcnt_h[i];
            spart_super_h[i] = (int)jpart_super_h[i];
            snzval_colend_L[i] = (int)jnzval_colend_L[i];
            srowind_colend_L[i] = (int)jrowind_colend_L[i];
            ssup_to_colend_L[i] = (int)jsup_to_colend_L[i];
            scolend_U[i] = (int)jcolend_U[i];
        }
        for (int i = 0; i < n + 1; i++)
        {
            sxa[i] = (int)jxa[i];
            snzval_colbeg_L[i] = (int)jnzval_colbeg_L[i];
            srowind_colbeg_L[i] = (int)jrowind_colbeg_L[i];
            scol_to_sup_L[i] = (int)jcol_to_sup_L[i];
            ssup_to_colbeg_L[i] = (int)jsup_to_colbeg_L[i];
            scolbeg_U[i] = (int)jcolbeg_U[i];
        }
        for (int i = 0; i < 2 * nnz_L; i++)
        {
            snzval_L[i] = (double)jnzval_L[i];
            srowind_L[i] = (int)jrowind_L[i];
        }
        for (int i = 0; i < 2 * nnz_U; i++)
        {
            snzval_U[i] = (double)jnzval_U[i];
            srowind_U[i] = (int)jrowind_U[i];
        }

        SuperMatrix sL, sU;
        SCPformat Lstore;
        NCPformat Ustore;
        Lstore.nnz = (int)nnz_L;
        Lstore.nsuper = (int)nsuper_L;
        Lstore.nzval = snzval_L;
        Lstore.nzval_colbeg = snzval_colbeg_L;
        Lstore.nzval_colend = snzval_colend_L;
        Lstore.rowind = srowind_L;
        Lstore.rowind_colbeg = srowind_colbeg_L;
        Lstore.rowind_colend = srowind_colend_L;
        Lstore.col_to_sup = scol_to_sup_L;
        Lstore.sup_to_colbeg = ssup_to_colbeg_L;
        Lstore.sup_to_colend = ssup_to_colend_L;
        sL.Stype = SLU_SCP;
        sL.Dtype = SLU_D;
        sL.Mtype = SLU_TRLU;
        sL.nrow = m;
        sL.ncol = n;
        sL.Store = &Lstore;

        Ustore.nnz = (int)nnz_U;
        Ustore.nzval = snzval_U;
        Ustore.rowind = srowind_U;
        Ustore.colbeg = scolbeg_U;
        Ustore.colend = scolend_U;
        sU.Stype = SLU_NCP;
        sU.Dtype = SLU_D;
        sU.Mtype = SLU_TRU;
        sU.nrow = m;
        sU.ncol = n;
        sU.Store = &Ustore;

        double *res = solve3_mt(m, n, nnz, sa, sasub, sxa, sb, nprocs, sperm_c, setree, scolcnt_h, spart_super_h, &sL, &sU);

        env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
        env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);
        env->ReleaseDoubleArrayElements(nzval_L, jnzval_L, JNI_ABORT);
        env->ReleaseDoubleArrayElements(nzval_U, jnzval_U, JNI_ABORT);

        env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);
        env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);
        env->ReleaseIntArrayElements(perm_c, jperm_c, JNI_ABORT);
        env->ReleaseIntArrayElements(etree, jetree, JNI_ABORT);
        env->ReleaseIntArrayElements(colcnt_h, jcolcnt_h, JNI_ABORT);
        env->ReleaseIntArrayElements(part_super_h, jpart_super_h, JNI_ABORT);

        env->ReleaseIntArrayElements(nzval_colbeg_L, jnzval_colbeg_L, JNI_ABORT);
        env->ReleaseIntArrayElements(nzval_colend_L, jnzval_colend_L, JNI_ABORT);
        env->ReleaseIntArrayElements(rowind_L, jrowind_L, JNI_ABORT);
        env->ReleaseIntArrayElements(rowind_colbeg_L, jrowind_colbeg_L, JNI_ABORT);
        env->ReleaseIntArrayElements(rowind_colend_L, jrowind_colend_L, JNI_ABORT);
        env->ReleaseIntArrayElements(col_to_sup_L, jcol_to_sup_L, JNI_ABORT);
        env->ReleaseIntArrayElements(sup_to_colbeg_L, jsup_to_colbeg_L, JNI_ABORT);
        env->ReleaseIntArrayElements(sup_to_colend_L, jsup_to_colend_L, JNI_ABORT);

        env->ReleaseIntArrayElements(rowind_U, jrowind_U, JNI_ABORT);
        env->ReleaseIntArrayElements(colbeg_U, jcolbeg_U, JNI_ABORT);
        env->ReleaseIntArrayElements(colend_U, jcolend_U, JNI_ABORT);

        env->SetDoubleArrayRegion(b, 0, n, (const jdouble *)res);

        SUPERLU_FREE(res);
        SUPERLU_FREE(Lstore.rowind);
        SUPERLU_FREE(Lstore.rowind_colbeg);
        SUPERLU_FREE(Lstore.rowind_colend);
        SUPERLU_FREE(Lstore.nzval);
        SUPERLU_FREE(Lstore.nzval_colbeg);
        SUPERLU_FREE(Lstore.nzval_colend);
        SUPERLU_FREE(Lstore.col_to_sup);
        SUPERLU_FREE(Lstore.sup_to_colbeg);
        SUPERLU_FREE(Lstore.sup_to_colend);
        SUPERLU_FREE(Ustore.rowind);
        SUPERLU_FREE(Ustore.colbeg);
        SUPERLU_FREE(Ustore.colend);
        SUPERLU_FREE(Ustore.nzval);
        return b;
    }
}
