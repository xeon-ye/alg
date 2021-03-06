/* 
 * Authors: Dong Shufeng
 * 2011-11-21
 */

#include "jipopt.hpp"

using namespace std;
using namespace Ipopt;

//
Jipopt::Jipopt(JNIEnv *env, jobject solver, jint n, jint m, jint nele_jac, jint nele_hess, jint index_style){
	this->env = env;
	this->solver = solver;
	this->n = n;
	this->m = m;
	this->nele_jac = nele_jac;
	this->nele_hess = nele_hess;
	this->index_style = index_style;

	using_LBFGS = false;
	using_scaling_parameters = false;

	xj = fj = grad_fj = gj = jac_gj = hessj = mult_x_Lj = mult_x_Uj = NULL;

	// the solver class
	jclass solverCls = env->GetObjectClass(solver);

	// get the methods
	get_bounds_info_= env->GetMethodID(solverCls,"get_bounds_info","(I[D[DI[D[D)Z");
	get_starting_point_= env->GetMethodID(solverCls,"get_starting_point","(IZ[DZ[D[DIZ[D)Z");
	eval_f_= env->GetMethodID(solverCls, "eval_f", "(I[DZ[D)Z");
	eval_grad_f_= env->GetMethodID(solverCls, "eval_grad_f", "(I[DZ[D)Z");
	eval_g_= env->GetMethodID(solverCls, "eval_g", "(I[DZI[D)Z");
	eval_jac_g_= env->GetMethodID(solverCls, "eval_jac_g", "(I[DZII[I[I[D)Z");
	eval_h_= env->GetMethodID(solverCls, "eval_h", "(I[DZDI[DZI[I[I[D)Z");
	
	get_scaling_parameters_= env->GetMethodID(solverCls,"get_scaling_parameters","([DI[DI[D[Z)Z");
	get_number_of_nonlinear_variables_= env->GetMethodID(solverCls,"get_number_of_nonlinear_variables","()I");
	get_list_of_nonlinear_variables_= env->GetMethodID(solverCls,"get_list_of_nonlinear_variables","(I[I)Z");
	
	if(get_bounds_info_==0 || get_starting_point_==0 ||
		eval_f_==0 || eval_grad_f_==0 || eval_g_==0 || eval_jac_g_==0 ||
		eval_h_==0 || get_number_of_nonlinear_variables_==0 || get_list_of_nonlinear_variables_==0){
		std::cerr << "Expected callback methods missing on JIpopt.java" << std::endl;		
	}
}

Jipopt::~Jipopt(){
}

bool Jipopt::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
						  Index& nnz_h_lag, IndexStyleEnum& index_style){
	n = this->n;
	m = this->m;
	nnz_jac_g = this->nele_jac;
	nnz_h_lag = this->nele_hess;
	
	index_style = (IndexStyleEnum)this->index_style;
	  
	return true;
}

bool Jipopt::get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l, Number *g_u) {

	jdoubleArray x_lj = NULL;
	jdoubleArray x_uj = NULL;
	jdoubleArray g_lj = NULL;
	jdoubleArray g_uj = NULL;

	if(x_l != NULL || x_u != NULL){
		x_lj = env->NewDoubleArray(n);
		x_uj = env->NewDoubleArray(n);
	}
	if(g_l != NULL || g_u != NULL){
		g_lj = env->NewDoubleArray(m);
		g_uj = env->NewDoubleArray(m);
	}

	if(!env->CallBooleanMethod(solver,get_bounds_info_,n,x_lj,x_uj,m,g_lj,g_uj))
		return false;
	
	// Copy from Java to native value 
	if(x_l!=NULL||x_u!=NULL){
		env->GetDoubleArrayRegion(x_lj, 0, n, x_l);
		env->GetDoubleArrayRegion(x_uj, 0, n, x_u);
	}
	if(g_l!=NULL||g_u!=NULL){
		env->GetDoubleArrayRegion(g_lj, 0, m, g_l);//
		env->GetDoubleArrayRegion(g_uj, 0, m, g_u);//
	}		
	return true;		
}

bool Jipopt::get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda, Number* lambda){
	jdoubleArray xj = this->xj;
	jdoubleArray z_lj = this->mult_x_Lj;
	jdoubleArray z_uj = this->mult_x_Uj;
	jdoubleArray lambdaj = this->mult_gj;

	if(!env->CallBooleanMethod(solver,get_starting_point_,
		n,init_x,xj,
		init_z,z_lj,z_uj,
		m,init_lambda,lambdaj)){			
		return false;
	}


	/* Copy from Java to native value */
	if(init_x)
		env->GetDoubleArrayRegion(xj, 0, n, x);//

	if(init_z){
		env->GetDoubleArrayRegion(z_lj, 0, n, z_L);
		env->GetDoubleArrayRegion(z_uj, 0, n, z_U);	
	}
	if(init_lambda)		
		env->GetDoubleArrayRegion(lambdaj, 0, m, lambda);
	return true;
}

bool Jipopt::eval_f(Index n, const Number* x, bool new_x,
                        Number& obj_value){
	if(new_x){
	/* Copy the native double x to the Java double array xj */ 
	  env->SetDoubleArrayRegion(xj, 0, n, const_cast<Number*>(x));
	}
    
	/* Call the java method */
	jboolean new_xj = new_x;
    if(!env->CallBooleanMethod(solver, eval_f_, n, xj, new_xj, fj))
    	return false;
    
	/* Copy from Java to native value */
    env->GetDoubleArrayRegion(fj, 0, 1, &obj_value);//should be a pointer	
	
    return true;
}

bool Jipopt::eval_grad_f(Index n, const Number* x, bool new_x,
                             Number* grad_f){
	if(new_x){
		//Copy the native double x to the Java double array xj 
		env->SetDoubleArrayRegion(xj, 0, n, const_cast<Number*>(x));	
	}

	//Call the java method
	jboolean new_xj = new_x;
	if(!env->CallBooleanMethod(solver, eval_grad_f_, n, xj, new_xj, grad_fj)) 	
		return false;
	
	env->GetDoubleArrayRegion(grad_fj, 0, n, grad_f);	
	return true;
}

bool Jipopt::eval_g(Index n, const Number* x, bool new_x,
                        Index m, Number* g){
	if(new_x){
		//Copy the native double x to the Java double array xj 
	  env->SetDoubleArrayRegion(xj, 0, n, const_cast<Number*>(x));
	}
	//Call the java method
	jboolean new_xj = new_x;
    if(!env->CallBooleanMethod(solver, eval_g_, n, xj, new_xj, m, gj))
    	return false;
	//Copy from Java to native value
    env->GetDoubleArrayRegion(gj, 0, m, g);	
	return true;
}

bool Jipopt::eval_jac_g(Index n, const Number* x, bool new_x,
                            Index m, Index nele_jac, Index* iRow,
                            Index *jCol, Number* jac_g){
  	if( new_x && x != NULL){
		// Copy the native double x to the Java double array xj  
	  env->SetDoubleArrayRegion(xj, 0, n, const_cast<Number*>(x));
  	}
  	
  	/// Create the index arrays if needed
  	jintArray iRowj = NULL;
  	jintArray jColj = NULL;
  	if(iRow != NULL && jCol != NULL){
  		iRowj = env->NewIntArray(nele_jac);
  		jColj = env->NewIntArray(nele_jac);
  	}
	
	//Call the java method 
	jboolean new_xj = new_x;
    if(!env->CallBooleanMethod(solver, eval_jac_g_, n, xj, new_xj,
		m, nele_jac, iRowj, jColj, jac_g == NULL ? NULL : jac_gj)) {
		return false;
	}
	
	// Copy from Java to native value
	if(jac_g != NULL)
    	env->GetDoubleArrayRegion(jac_gj, 0, nele_jac, jac_g);

	if(iRow != NULL && jCol != NULL){    	
		jint *iRow_jint = env->GetIntArrayElements(iRowj, 0);
		jint *jCol_jint = env->GetIntArrayElements(jColj, 0);
		for(int i = 0; i < nele_jac; i++){
			iRow[i] = iRow_jint[i];
			jCol[i] = jCol_jint[i];
		}	
		env->ReleaseIntArrayElements(iRowj, iRow_jint, 0);
		env->ReleaseIntArrayElements(jColj, jCol_jint, 0);		
    }
	return true;
}

bool Jipopt::eval_h(Index n, const Number* x, bool new_x,
                        Number obj_factor, Index m, const Number* lambda,
                        bool new_lambda, Index nele_hess,
                        Index* iRow, Index* jCol, Number* hess) {
  	if(new_x && x != NULL){		
		//Copy the native double x to the Java double array xj
	  env->SetDoubleArrayRegion(xj, 0, n, const_cast<Number*>(x));
  	}
  	if( new_lambda && lambda!=NULL){		
		//Copy the native double lambda to the Java double array lambdaj
	  env->SetDoubleArrayRegion(mult_gj, 0, m, const_cast<Number*>(lambda));//multi_gj <==> lambdaj
  	}
  	
  	/// Create the index arrays if needed
  	jintArray iRowj = NULL;
  	jintArray jColj = NULL;
  	if(iRow != NULL && jCol != NULL){
  		iRowj = env->NewIntArray(nele_hess);
  		jColj = env->NewIntArray(nele_hess);
  	}

	//Call the java method
	jboolean new_xj = new_x;
	jboolean new_lambdaj = new_lambda;
    if(!env->CallBooleanMethod(solver, eval_h_, n, xj, new_xj,
    	obj_factor, m, mult_gj, new_lambdaj,
		nele_hess, iRowj, jColj, hess == NULL ? NULL : hessj)){			
    	return false;
	}
   
	/* Copy from Java to native value */
	if(hess != NULL)
    	env->GetDoubleArrayRegion(hessj, 0, nele_hess, hess);
		
    if(iRow != NULL && jCol != NULL){			
 		jint *iRow_jint = env->GetIntArrayElements(iRowj, 0);
		jint *jCol_jint = env->GetIntArrayElements(jColj, 0);			
		for(int i = 0; i < nele_hess; i++){
			iRow[i] = iRow_jint[i];
			jCol[i] = jCol_jint[i];			
		}
		env->ReleaseIntArrayElements(iRowj, iRow_jint, 0);
		env->ReleaseIntArrayElements(jColj, jCol_jint, 0);
	}	
	return true;
}

void Jipopt::finalize_solution(SolverReturn status, Index n, const Number *x, 
							   const Number *z_L, const Number *z_U, Index m, 
							   const Number *g, const Number *lambda, Number obj_value, 
							   const IpoptData *ip_data, IpoptCalculatedQuantities *ip_cq) {
	//nothing is done in this method now.
}


/** overload this method to return scaling parameters. This is
     *  only called if the options are set to retrieve user scaling.
     *  There, use_x_scaling (or use_g_scaling) should get set to true
     *  only if the variables (or constraints) are to be scaled.  This
     *  method should return true only if the scaling parameters could
     *  be provided.
     */
bool Jipopt::get_scaling_parameters(Number& obj_scaling,
                                        bool& use_x_scaling, Index n,
                                        Number* x_scaling,
                                        bool& use_g_scaling, Index m,
                                        Number* g_scaling) {	
	if(using_scaling_parameters){
		jdoubleArray obj_scaling_j=env->NewDoubleArray(1);
		jdoubleArray x_scaling_j=env->NewDoubleArray(n);
		jdoubleArray g_scaling_j=env->NewDoubleArray(m);

		jbooleanArray use_x_g_scaling_j = env->NewBooleanArray(2);			

		env->CallBooleanMethod(solver,get_scaling_parameters_,
			obj_scaling_j,
			n,x_scaling_j,
			m,g_scaling_j,
			use_x_g_scaling_j);

		jboolean* use_x_g_scaling=env->GetBooleanArrayElements(use_x_g_scaling_j,0);
		//Copy from Java to native value
		env->GetDoubleArrayRegion(obj_scaling_j, 0, 1, &obj_scaling);
		if(use_x_g_scaling[0]) {			
			env->GetDoubleArrayRegion(x_scaling_j, 0, n, x_scaling);
			use_x_scaling = true;
		} else
			use_x_scaling=false;
		
		//Copy from Java to native value
		if(use_x_g_scaling[1]) {			
			env->GetDoubleArrayRegion(g_scaling_j, 0, n, g_scaling);
			use_g_scaling = true;
		} else 
			use_g_scaling = false;
		
		env->ReleaseBooleanArrayElements(use_x_g_scaling_j, use_x_g_scaling, 0);
		return true;
	} else
		return false;		
}



Index Jipopt::get_number_of_nonlinear_variables(){
	if(using_LBFGS) {
		return env->CallIntMethod(solver,get_number_of_nonlinear_variables_);
	} else 
		return -1;	
}
    

bool Jipopt::get_list_of_nonlinear_variables(Index num_nonlin_vars,Index* pos_nonlin_vars){
	if(using_LBFGS) {
		jintArray pos_nonlin_vars_j = env->NewIntArray(num_nonlin_vars);		
		if(!env->CallBooleanMethod(solver,get_list_of_nonlinear_variables_,
			num_nonlin_vars,pos_nonlin_vars_j)) {
			return false;
		}

		if(pos_nonlin_vars != NULL){	
			jint *pos_nonlin_vars_jp = env->GetIntArrayElements(pos_nonlin_vars_j, 0);
			for(int i = 0; i < num_nonlin_vars; i++)
				pos_nonlin_vars[i] = pos_nonlin_vars_jp[i];
			env->ReleaseIntArrayElements(pos_nonlin_vars_j, pos_nonlin_vars_jp, 0);			
		}		
		return true;
	} else
		return false;
}

#ifdef __cplusplus
extern "C" {
#endif


JNIEXPORT jlong JNICALL Java_org_coinor_Ipopt_CreateIpoptProblem 
(JNIEnv *env, jobject obj_this, 
 jint n,  jint m,
 jint nele_jac, jint nele_hess,
 jint index_style) {
	/* create the IpoptProblem */
	Jipopt* problem = new Jipopt(env, obj_this, n, m, nele_jac, nele_hess, index_style);
	if(problem == NULL)
		return 0;
	JIpoptSolver * solver = new JIpoptSolver();
	solver->problem = problem;
	solver->application = new IpoptApplication();
	//return c++ class point
	return (jlong)solver;
}

JNIEXPORT jint JNICALL Java_org_coinor_Ipopt_OptimizeTNLP
  (JNIEnv *env, jobject obj_this, jlong pipopt, jstring outfilename ,
  jdoubleArray xj,
jdoubleArray gj,
jdoubleArray obj_valj,
jdoubleArray mult_gj,
jdoubleArray mult_x_Lj,
jdoubleArray mult_x_Uj,

jdoubleArray callback_grad_f,
jdoubleArray callback_jac_g,
jdoubleArray callback_hess) {
	// cast back our class

	JIpoptSolver *solver = (JIpoptSolver *)pipopt;
	Jipopt * problem = solver->problem;
	problem->env = env;
	problem->solver = obj_this;

	problem->xj = xj;
	problem->gj = gj;
	problem->fj = obj_valj;
	problem->mult_gj = mult_gj;
	problem->mult_x_Lj = mult_x_Lj;
	problem->mult_x_Uj = mult_x_Uj;

	problem->grad_fj = callback_grad_f;
	problem->jac_gj = callback_jac_g;
	problem->hessj = callback_hess;

	 //  (use a SmartPtr, not raw)
	ApplicationReturnStatus status;

	if(outfilename){
		const char *pparameterName = env->GetStringUTFChars(outfilename, 0);
		string outfile=pparameterName;
		status = solver->application->Initialize(outfile);
		env->ReleaseStringUTFChars(outfilename, pparameterName);
	} else
		status = solver->application->Initialize();

	if (status != Solve_Succeeded) {
		printf("\n\n*** Error during initialization!\n");
		return (int) status;
	}

	/* solve the problem */
    //SmartPtr<SparseSymLinearSolverInterface> SolverInterface = new CustomSolverInterface();
    SmartPtr<SparseSymLinearSolverInterface> SolverInterface = new CudaSolverInterface();
    SmartPtr<TSymScalingMethod> ScalingMethod;
//    SmartPtr<TSymScalingMethod> ScalingMethod = new SlackBasedTSymScalingMethod();
    SmartPtr<SymLinearSolver> ScaledSolver = new TSymLinearSolver(SolverInterface, ScalingMethod);
    SmartPtr<AugSystemSolver> AugSolver = new StdAugSystemSolver(*ScaledSolver);
    SmartPtr<AlgorithmBuilder> alg_builder = new AlgorithmBuilder(AugSolver);
    solver->application->Options()->SetStringValue("linear_solver","custom");
    SmartPtr<NLP> nlp_adapter = new TNLPAdapter(problem, NULL);
	status = solver->application->OptimizeNLP(nlp_adapter, alg_builder);
//	status = solver->application->OptimizeTNLP(problem);

	return (jint) status;
}

JNIEXPORT void JNICALL Java_org_coinor_Ipopt_FreeIpoptProblem
(JNIEnv *env, jobject obj_this, jlong pipopt) {
	// cast back our class
	JIpoptSolver *solver = (JIpoptSolver *)pipopt;
	if(solver != NULL) 
		delete solver;	
}

JNIEXPORT jboolean JNICALL Java_org_coinor_Ipopt_AddIpoptIntOption
(JNIEnv * env, jobject obj_this, jlong pipopt, jstring jparname, jint jparvalue) {
	// cast back our class
	JIpoptSolver *solver = (JIpoptSolver *)pipopt;
	
	const char *pparameterName = env->GetStringUTFChars(jparname, 0);
	string parameterName = pparameterName;

	// Try to apply the integer option
	jboolean ret = solver->application->Options()->SetIntegerValue(parameterName, jparvalue);
	
	env->ReleaseStringUTFChars(jparname, pparameterName);	
	return ret;
}

JNIEXPORT jboolean JNICALL Java_org_coinor_Ipopt_AddIpoptNumOption
(JNIEnv * env, jobject obj_this, jlong pipopt, jstring jparname, jdouble jparvalue) {  
	// cast back our class
	JIpoptSolver *solver = (JIpoptSolver *)pipopt;
	
	const char *pparameterName = env->GetStringUTFChars(jparname, 0);
	string parameterName=pparameterName;

	// Try to set the real option
	jboolean ret = solver->application->Options()->SetNumericValue(parameterName,jparvalue);
	
	env->ReleaseStringUTFChars(jparname, pparameterName);
	
	return ret;
}

JNIEXPORT jboolean JNICALL Java_org_coinor_Ipopt_AddIpoptStrOption
(JNIEnv * env, jobject obj_this, jlong pipopt, jstring jparname, jstring jparvalue) {
	// cast back our class
	JIpoptSolver *solver = (JIpoptSolver *)pipopt;
	
	const char *pparameterName = env->GetStringUTFChars(jparname, NULL);
	string parameterName = pparameterName;
	const char *pparameterValue = env->GetStringUTFChars(jparvalue, NULL);
	string parameterValue = pparameterValue;

	//parameterValue has been changed to LowerCase in Java!
	if(parameterName == "hessian_approximation" && parameterValue=="limited-memory") {
		solver->problem->using_LBFGS = true;		
	} else if(parameterName == "nlp_scaling_method" && parameterValue=="user-scaling") 
		solver->problem->using_scaling_parameters = true;			

	// Try to apply the string option
	jboolean ret = solver->application->Options()->SetStringValue(parameterName,parameterValue);

	env->ReleaseStringUTFChars(jparname, pparameterName);
	env->ReleaseStringUTFChars(jparname, pparameterValue);	
	return ret;
}

#ifdef __cplusplus
}
#endif
