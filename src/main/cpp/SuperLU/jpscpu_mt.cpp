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
#include "jpscpu_LinearSolverMT.h"

using namespace std;

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
