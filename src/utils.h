#ifndef _UTILS
#define _UTILS
#include <vector>
#include "globals.h"
#include "stdp.h"

# define PI 3.14159265358979323846 

//
using namespace libconfig;



// ----------------------------------------- //
unsigned int* int_vector(unsigned int n, unsigned int init_val) {
  unsigned int *x = (unsigned int *) malloc(n * sizeof(unsigned int));
  // init to zero
  for(unsigned int i=0; i < n; ++i) {
    x[i] = init_val;
  }
  return x;
}

double* double_vector(unsigned int n, double init_val) {
  double *x = (double *) malloc(n * sizeof(double));
  for(unsigned int i=0; i < n; ++i) {
    x[i] = init_val;
  }
  return x;
}

double* double_vector(unsigned long long n, double init_val) {
  double *x = (double *) malloc(n * sizeof(double));
  for(unsigned int i=0; i < n; ++i) {
    x[i] = init_val;
  }
  return x;
}

void vector_divide(std::vector<double> &a, double z) { 
   for(unsigned int i = 0; i < N; ++i) {
     a[i] /= z;
   }
}

void vector_sum(std::vector<double> &a, std::vector<double> &b) { 
  for(unsigned int i = 0; i < N; ++i) {
    a[i] += b[i];
  }
}

void shift_matrix(int **mat, int rows, int clmns) {
  // shift matrix to the left by one and set the last element to zero
  int row = 0;
  while (row < rows) {
    for(int clmn = 1; clmn < clmns; ++clmn) {
      mat[row][clmn - 1] = mat[row][clmn];
    }
    mat[row][clmns-1] = 0;
    row += 1; 
  }
}

int** create_2d_matrix(int rows, int clmns, int init_val) {
  int **arr = new int* [rows];
  // arr = new int* [rows];
  for(int i = 0; i < rows; ++i) {
    arr[i] = new int [clmns];
  }
  // initialize
  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < clmns; ++j) {
      arr[i][j] = init_val;
    }
  }
  std::cout << "allocation done" << "\n";
  return arr;
}

void clear_2d_matrix(int **arr, int rows, int clmns) {
    for(int i = 0; i < rows; ++i) {
      delete [] arr[i];
    }
  delete [] arr;
}


void get_ff_input(const char *filename) {
  double x;
  std::ifstream inFile;
  inFile.open(filename);
  std::cout << "reading ff input"  << "\n";
  // inFile.open("test.txt");
  if (!inFile || test_stdp) {
    if(!test_stdp) {
      std::cout << "Unable to open file \n";
    }
    std::cout << "setting time varying ff input to zero \n";
    for(size_t i = 0; i < n_steps; ++i){
      I_of_t.push_back(0.0);
    }
    // exit(1); // terminate with error
  }
  while (inFile >> x) {
    I_of_t.push_back(x);
    }
    inFile.close();
}

void GenSparseMat(unsigned int *conVec,  unsigned int rows, unsigned int clms, unsigned int* sparseVec, unsigned int* idxVec, unsigned int* nPostNeurons ) {
  /* generate sparse representation
     conVec       : input vector / flattened matrix 
     sparseVec    : sparse vector
     idxVec       : every element is the starting index in sparseVec for ith row in matrix conVec
     nPostNeurons : number of non-zero elements in ith row 
  */
  // printf("\n MAX Idx of conmat allowed = %u \n", rows * clms);
  unsigned long long int i, j, counter = 0, nPost;
  for(i = 0; i < rows; ++i) {
    nPost = 0;
    for(j = 0; j < clms; ++j) {
      // printf("%llu %llu %llu %llu\n", i, j, i + clms * j, i + rows * j);
      if(conVec[i + rows * j]) { /* i --> j  */
        sparseVec[counter] = j;
        counter += 1;
        nPost += 1;
      }
    }
    nPostNeurons[i] = nPost;
  }
  
  idxVec[0] = 0;
  for(i = 1; i < rows; ++i) {
    idxVec[i] = idxVec[i-1] + nPostNeurons[i-1];
  }
}

void gen_stdp_cons(){
  // stdp only on sqrt_K e-to-e synapses
  // should be called after forward sparse vectors are generated
  const gsl_rng_type * T;
  gsl_rng *r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  //
  double stdp_con_prob = 1.0 / (double)sqrt_K;
  std::cout << "stdp_con prob = " << stdp_con_prob  << "\n";
  //
  unsigned int tmpIdx, cntr, con_idx;
  unsigned int stdp_cntr_i;
  FILE *fp = fopen("./data/is_stdp_con.txt", "w");
  //  
  for(size_t i = 0; i < NE; ++i) {
    stdp_cntr_i = 0;
    cntr = 0;
    tmpIdx = idxVec[i];
    while(cntr < nPostNeurons[i]) {
      con_idx = tmpIdx + cntr;
      if(sparseConVec[con_idx] < NE) {
	if(gsl_rng_uniform(r) <= stdp_con_prob){
	  IS_STDP_CON[con_idx] = 1;

	  stdp_weights[con_idx] = stdp_initial_weights;
	  // * (1.0 + rand() / double(RAND_MAX));

	  n_stdp_connections += 1;
	  stdp_cntr_i += 1;
	  fprintf(fp, "%u\n", con_idx);
	  temp_stdp_con_idx = con_idx;
	  // std::cout << con_idx  << "\n";
	}
      }
      cntr +=1;      
    }
    // fprintf(fp, "%u\n", stdp_cntr_i);    
  }
  std::cout << "n stdp cons = " << n_stdp_connections << "\n";  
  std::cout << "n stdp cons per E neuron = " << (double)n_stdp_connections / (double)NE  << "\n";
  fclose(fp);
  gsl_rng_free(r);
}

void gen_backward_vectors(unsigned int *conVec,  unsigned int rows, unsigned int clms) {
  /* generate sparse representation
     conVec       : input vector / flattened matrix 
     sparseVec    : sparse vector
     idxVec       : every element is the starting index in sparseVec for ith row in matrix conVec
     n_pre_neurons : number of non-zero elements in ith row 
  */
  // printf("\n MAX Idx of conmat allowed = %u \n", rows * clms);
  unsigned long long int i, j, counter = 0, nPre;
  for(j = 0; j < clms; ++j) {
    nPre = 0;
    for(i = 0; i < rows; ++i) {
      if(conVec[i + rows * j]) { /* i --> j  */
        pre_sparseVec[counter] = i;
        counter += 1;
        nPre += 1;
      }
    }
    n_pre_neurons[j] = nPre;
  }

  // for(j =0; j < rows; ++j) {
  //   nPre = n_pre_neurons[j];
  //   std::cout << " pre neurons of neuron # " << j << "\n";    
  //   unsigned int tmp_idx = pre_idxVec[j], tmp_cntr=0;
  //   while(tmp_cntr < nPre) {
  //     std::cout << pre_sparseVec[tmp_idx + tmp_cntr] << std::endl;
  //     tmp_cntr += 1;
  //   }
  // }

}


void init_STDP_vectors() {
  // must be called after gen_conmat functions,
  // depends on the variable n_connections
  std::cout << "init stdp vecs"  << "\n";
  std::cout << "n_connections = " << n_connections << "\n";
  if (n_connections > 0){
      //
      n_pre_neurons = int_vector(N, 0);
      pre_idxVec = int_vector(N, 0);
      pre_sparseVec = int_vector(n_connections, 0);

      //  
      stdp_pre_trace = double_vector(n_connections, 0.0);
      stdp_post_trace = double_vector(n_connections, 0.0);



      if(test_stdp && STDP_ON) {
	stdp_initial_weights = 0.5;
	stdp_max_weight = 1.0;
	IS_STDP_CON = int_vector(n_connections, 1);	
      }
      else {
      // stdp weights are sacaled as J / K so that they result in O(1) net input

	// stdp_initial_weights = 1e-1 * Jee_K;   // pre factor 1e-1 necessary? 
	// stdp_max_weight = Jee_K;

	IS_STDP_CON = int_vector(n_connections, 0);
	stdp_initial_weights = 1e-4 * Jee_K; // / sqrt_K;
	stdp_max_weight = Jee_K / 2.0; // * sqrt_K;             
      }

      
      std::cout << "initial weights = " << stdp_initial_weights << "\n";      
      std::cout << "max weight = " << stdp_max_weight << "\n";      
      
      // stdp_weights = double_vector(n_connections, stdp_initial_weights);

      // set stdp weights sto tdp_initial_weights in method gen_stdp_cons() only for sqrt_K connections else is zero
      stdp_weights = double_vector(n_connections, 0.0); 

      //
      std::cout << "done!"  << "\n";            
    }
}


double connection_prob(unsigned int i, unsigned int j){
  //  double alpha = 8;
  double assymetry = 0.0 * PI / 180.0;
  double d_theta = ((double)i - (double)j - assymetry) * PI / (double)NE;

  // double prob = ((double) K / NE) * (1.0 + 2.0 * alpha * cos(2.0 * (d_theta)) / sqrt_K);
  double prob = ((double) K / NE) * (1.0 + 2.0 * alpha * cos(2.0 * d_theta) / sqrt_K);
  if (prob > 1){
    std::cout << "prob > 1!"  << "\n";
    exit (EXIT_FAILURE);
  }
  return prob;
}

void recip_ei() {
  double pBi, pUni, p;
  unsigned long long i=0, j=0;
  double urand = 0;
  unsigned int *conmat = int_vector(N * N, 0);  
  const gsl_rng_type * T;
  gsl_rng *r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  p = (double)K / (double)NE;
  pBi = con_symmetry * p + (1 - con_symmetry) * p * p;
  pUni = 2 * (1 - con_symmetry) * p * (1 - p);
  for(i = 0; i < N; ++i) {

    if(i < NE){
      for(j = 0; j < NE; ++j) { /* E --> E */
	if(p >= gsl_rng_uniform(r)) {
	  conmat[i + j * N] = 1;
	}
      }
	
      for(j = NE; j < N; ++j) { // E <--> I
	if(pBi > gsl_rng_uniform(r)) {
	  conmat[i + j * N] = 1; // i --> j
	  conmat[j + i * N] = 1; //  j --> i
	}
	else {
	  if(pUni > gsl_rng_uniform(r)) {
	    if(gsl_rng_uniform(r) > 0.5) {
	      conmat[j + i * N] = 1; // i --> j
	    }
	    else {
	      conmat[i + j * N] = 1; //  j --> i
	    }
	  }      
	}
      }
    }
    if(i >= NE) {
      for(j = NE; j < N; ++j) {/* I --> I */
	if(p >= gsl_rng_uniform(r)) {
	  conmat[i + j * N] = 1;
	}
      }
    }
  }


  n_connections = 0;
  for(i = 0; i < N; ++i) {
    for (j=0; j < N; ++j) {
      if (conmat[i + j * N]) {
	++n_connections;
      }
    }
  }
  
  sparseConVec = int_vector(n_connections, 0);
  GenSparseMat(conmat, N, N, sparseConVec, idxVec, nPostNeurons);
  //
  free(conmat);

  
}


void gen_conmat() {
  // connection matrix (i, j) = i + j * n,
  // i is pre,  j is post
  // e.g, 
  // 1 3 6
  // 2 4 7
  // 3 5 8
  
  const gsl_rng_type * T;
  gsl_rng *r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  // T1 = gsl_rng_default;
  // r1 = gsl_rng_alloc (T1);
  
  double u_rand; // uniform random number

  double con_prob = 0; // (double) K / NE;
  double con_prob_E=0, con_prob_I=0;

  


  con_prob_E = (double) K / NE;
  con_prob_I = (double) K / NI;

  if(scaling_type == 2) {
    con_prob_E = (double) K / NE;
    con_prob_I = (double) 0.25 * K / NI;
  }

  // unsigned long long int n_connections = 0;
  n_connections = 0;
  unsigned int *conmat = int_vector(N * N, 0);;

  unsigned int n_ee_cons = 0;

  if(test_stdp) {
    // all neurons project to neuron 0
    //    size_t i=0;
    for(size_t j=1; j< NE; ++j){
      conmat[j] = 1;
      // std::cout << N*j << "\n";
      n_connections += 1;
    }
  }

  else{
    for(size_t i=0; i < N; ++i) {
      if (i < NE) {
	con_prob = con_prob_E;
      }
      else {
	con_prob = con_prob_I;
      }
      for(size_t j=0; j < N; ++j) {
	if(j < NE) {
	  if(IS_RANDOM){
	    con_prob = con_prob_E;	    
	  }
	  else {
	    con_prob = connection_prob(i, j);
	  }
	}
	u_rand = gsl_rng_uniform(r);
	if(u_rand <= con_prob){
	  conmat[i + N * j] = 1;
	  n_connections += 1;
	  
	  
	  if(i < NE && j < NE) {n_ee_cons += 1;}

	  //   if(gsl_rng_uniform(r) <= stdp_con_prob) {
	  //     IS_STDP_CON[i ] = 1;
	  //     n_stdp_connections += 1;
	  //   }
	  // }
	}
      }
    }
  }

  std::cout << " connections done!  "  << "\n"; 
  std::cout << "n cons = " << (double)n_connections  << "\n";
  std::cout << "c prob = " << con_prob  << "\n";

  std::cout << "n ee cons = " << n_ee_cons << "\n";
  std::cout << "n ee cons per cell = " << (double)n_ee_cons / NE << "\n";      

  /* debug code
  if(NE == 2) {
    conmat[0] = 0;
    conmat[1] = 0;
    conmat[3] = 0;
    n_connections = 1;
    printf(" ----  conmat --> \n");
    for(size_t i=0; i < N; ++i) {
      for(size_t j=0; j < N; ++j) {
	printf("%u ", conmat[i + j * N]);      
      }
      printf("\n");
    }
  }
  */

  // printf("conmat --> \n");
  // for(size_t i=0; i < N; ++i) {
  //   for(size_t j=0; j < N; ++j) {
      
  //     printf("[(%lu %lu)::%lu =  %u]  ", i, j, i + j * N, conmat[i + j * N]);      
  //   }
  //   printf("\n");
  // }

  //

  sparseConVec = int_vector(n_connections, 0);

  //
  GenSparseMat(conmat, N, N, sparseConVec, idxVec, nPostNeurons);


  if(STDP_ON) {
    init_STDP_vectors();    
    gen_backward_vectors(conmat, N, N);
    gen_stdp_cons();
  }
  //
  free(conmat);
  gsl_rng_free(r);
}


int read_params() {
  Config cfg;

  // Read the file. If there is an error, report it and exit.
  try {
    cfg.readFile("params.cfg");
  }
  catch(const FileIOException &fioex)
  {
    std::cerr << "I/O error while reading file." << std::endl;
    return(EXIT_FAILURE);
  }
  catch(const ParseException &pex)
  {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    return(EXIT_FAILURE);
  }

  const Setting& root = cfg.getRoot();

  // set params
  NE = root["NE"];
  NI = root["NI"];
  N = NE + NI;
  K = root["K"];

  // symmetry parameter 
  con_symmetry = root["con_symmetry"];
  //
  tau_membrane = root["tau_mem"];
  tau_e = root["tau_e"];
  tau_i = root["tau_i"];
  tau_threshold = root["tau_thresh"];

  // Brunel parameters
  delay_syn = root["delay_syn"];
  std::cout << "hello here 5"  << "\n";  
  g_brunel = root["g_brunel"];
  std::cout << "hello here 7"  << "\n";  
  //
  Je0 = root["Je0"];
  Ji0 = root["Ji0"];
  //
  Jee = root["Jee"];
  Jei = root["Jei"];
  Jie = root["Jie"]; 
  Jii = root["Jii"];
  //
  sqrt_K = sqrt((double)K);

  //
  V_rest = root["V_rest"];
  V_threshold_initial = root["V_threshold_initial"];
  d_threshold = root["d_threshold"];
  // V_e = root["V_e"];
  // V_i = root["V_i"];
  V_reset = root["V_reset"];

  // input rates
  v_ext = root["v_ext"];

  // simulation time
  // std::cout << root["t_stop"] << "\n";
  
  t_stop = root["t_stop"];
  dt = root["dt"];
  discard_time = root["discard_time"];
  t_stop += discard_time;

  // connectivity type
  IS_RANDOM = root["IS_RANDOM"];
  alpha = root["alpha"];

  // // stp
  STP_ON = root["STP_ON"];
  stp_U = root["stp_U"];
  stp_A = root["stp_A"];
  stp_tau_d = root["stp_tau_d"];
  stp_tau_f = root["stp_tau_f"];

  // STDP params
  test_stdp = root["test_stdp"];
  STDP_ON = root["STDP_ON"];
  if(test_stdp == 1) {
    STDP_ON = 1;
  }
  stdp_tau_pre = root["stdp_tau_pre"];
  stdp_tau_post = root["stdp_tau_post"];
  stdp_lr_pre = root["stdp_lr_pre"];
  stdp_lr_post = root["stdp_lr_post"];  
  
  scaling_type = 0; // root["scaling_type"];
  
  // scaling
  if (scaling_type == 1) {
    Je0 = Je0 * (V_threshold_initial - V_rest);
    Ji0 = Ji0 * (V_threshold_initial - V_rest);
    Jee_K = (Jee / (double)K) * (V_threshold_initial - V_rest) / tau_e;
    Jei_K = (Jei / (double)K) * (V_threshold_initial - V_rest) / tau_i;
    Jie_K = (Jie / (double)K) * (V_threshold_initial - V_rest) / tau_e;
    Jii_K = (Jii / (double)K) * (V_threshold_initial - V_rest) / tau_i;  
  }
  else if (scaling_type == 2){

    Jee = Jee * tau_membrane;
    Jee_K = Jee;
    Jei_K = -1.0 * g_brunel * Jee;
    Jie_K = Jee;
    Jii_K = -1.0 * g_brunel * Jee;
    Je0 = Jee;
    Ji0 = Jee;

    // Je0 = Jee * (V_threshold_initial - V_rest);
    // Ji0 = Jee * (V_threshold_initial - V_rest);
    // Jee_K = Jee * (V_threshold_initial - V_rest) / tau_e;
    // Jei_K = -1.0 * Jee * (V_threshold_initial - V_rest) / tau_i;
    // Jie_K = Jee * (V_threshold_initial - V_rest) / tau_e;
    // Jii_K = -1.0 * Jee * (V_threshold_initial - V_rest) / tau_i;  


  }
  else {
    Je0 = Je0 * (V_threshold_initial - V_rest);
    Ji0 = Ji0 * (V_threshold_initial - V_rest);
    Jee_K = (Jee / sqrt_K) * (V_threshold_initial - V_rest) / tau_e;
    Jei_K = (Jei / sqrt_K) * (V_threshold_initial - V_rest) / tau_i;
    Jie_K = (Jie / sqrt_K) * (V_threshold_initial - V_rest) / tau_e;
    Jii_K = (Jii / sqrt_K) * (V_threshold_initial - V_rest) / tau_i;  
  }
  

  // Jii_K = (Jii / sqrt_K) * (V_rest - V_i) / tau_i;  
  std::cout << "-- -- -- -- -- -- -- -- -- -- --" << "\n";
  std::cout << " recurrent interactions " << "\n";
  std::cout << Jee_K << " " << Jei_K << "\n";
  std::cout << Jie_K << " " << Jii_K << "\n";
  std::cout << " " << "\n";
  std::cout << "-- -- -- -- -- -- -- -- -- -- --" << "\n";  
  std::cout << " FF strength " << "\n";
  std::cout << Je0 << "\n" << Ji0 << "\n";
  std::cout << "-- -- -- -- -- -- -- -- -- -- --" << "\n";
  // std::cout << "- - - - - - - -"  << "\n";
  // float mf_rate = tau_membrane * v_ext * Ji0 / (-Jii * (V_threshold_initial - V_rest) * 1e-3);
  // // float mf_rate = tau_membrane * v_ext / (-Jii * (V_rest - V_i) * 1e-3);
  // std::cout << "mf rate = " <<  mf_rate   << "\n";		
  // std::cout << "- - - - - - - -"  << "\n";
  
  // float mf_rate_3 = tau_membrane * input_ext / (-Jii * (V_rest - V_i));  
  
  // display params 
  std::cout << "sim params:"  << "\n";
  std::cout << "NE = " << NE << "\n";
  std::cout << "NI = " << NI << "\n";
  std::cout << "v_ext = " << v_ext * 1e3 << "Hz" << "\n";  
  std::cout << "t_stop = " << t_stop  << "\n";
  std::cout << "V_rest = " << V_rest  << "\n";  
  std::cout << "V_threshold_init = " << V_threshold_initial  << "\n";
  std::cout << "V_reset = " << V_reset  << "\n";

  std::cout << "- - - - - " << "\n";
  std::cout << "STP_ON " << (bool)STP_ON  << "\n";
  std::cout << "STDP_ON " << (bool)STDP_ON  << "\n";  
  std::cout << "- - - - - " << "\n";  
  // std::cout << "V_i = " << V_i  << "\n";
  // std::cout << "V_e = " << V_e  << "\n";  
  return 0;
}

FILE* push_spike_init() {
  FILE *spk_fp = fopen("./data/spikes.txt", "w");
  return spk_fp;
}



void init_state_vectors() {
  // this function must be called after calling read_params
  Vm = double_vector(N, V_rest);
  V_threshold = double_vector(N, V_threshold_initial);  
  g_e = double_vector(N, 0); 
  g_i = double_vector(N, 0);
  nPostNeurons = int_vector(N, 0);
  idxVec = int_vector(N, 0);
  spk_fp = push_spike_init();
  // STP
  if (STP_ON) {
    stp_u = double_vector(N, 0.0);
    stp_x = double_vector(N, 1.0);
    stp_x_old = double_vector(N, 1.0);
  }
  // stp_x_old = double_vector(n_connections, 1.0);  
  // valid_sim_time = (t_stop - discard_time);

  // delay buffer
  if(scaling_type == 2) {  
    n_delay_bins = (int) (delay_syn / dt);
    std::cout << "# delay bins" << n_delay_bins << "\n";
    syn_delay_buffer = create_2d_matrix(N, n_delay_bins, 0);
    
    
  }
}


void delete_state_vectors() {
  //
  free(Vm);
  free(V_threshold);
  free(g_i);
  free(g_e);
  free(nPostNeurons);
  free(idxVec);
  free(sparseConVec);
  //
  if(scaling_type == 2) {
    clear_2d_matrix(syn_delay_buffer, N, n_delay_bins);
  }
  //
  fflush(spk_fp);
  fclose(spk_fp);
  //
  if (STP_ON) {
    free(stp_u);
    free(stp_x);
  }
  //
  if(STDP_ON) {
    free(n_pre_neurons);
    free(pre_idxVec);
    free(pre_sparseVec);
    //  
    free(stdp_pre_trace);
    free(stdp_post_trace);
    free(stdp_weights);
  }
}


void push_spike(double spike_time, unsigned int neuron_idx) {
  // save one excitatory spikes 
  if(neuron_idx < NE) {
    fprintf(spk_fp, "%f %u\n", spike_time, neuron_idx);
  }
}

void propagate_spikes(unsigned int pre_neuron_idx) {
  // update all the post synaptic neurons to presynaptic neuron
  unsigned int tmpIdx, cntr, post_neuron_idx;
  // unsigned int n_spikes_e = 0, n_spikes_i = 0;
  // n_spikes[0] = 0;
  // n_spikes[1] = 0;
  cntr = 0;
  tmpIdx = idxVec[pre_neuron_idx];
  if (pre_neuron_idx < NE){
    n_spikes[0] += 1;    
  }
  else {
    n_spikes[1] += 1;
  }
  while(cntr < nPostNeurons[pre_neuron_idx]) {
    post_neuron_idx = sparseConVec[tmpIdx + cntr];
    cntr += 1;
    if(pre_neuron_idx < NE) {
      /* --    E-to-E    -- */      
      if(post_neuron_idx < NE) {
	  g_e[post_neuron_idx] += Jee_K;
      }
      /* --    E-to-I    -- */      
      else {
	g_e[post_neuron_idx] += Jie_K;		
      }
    }
    else {
      /* --    I-to-E    -- */      
      if(post_neuron_idx < NE) {
	g_i[post_neuron_idx] += Jei_K;	
      }
      /* --    I-to-I    -- */      
      else {
	g_i[post_neuron_idx] += Jii_K;		
      }
    }
  }
}

void propagate_spikes_brunel(unsigned int pre_neuron_idx) {
  // update all the post synaptic neurons to presynaptic neuron
  unsigned int tmpIdx, cntr, post_neuron_idx;
  // unsigned int n_spikes_e = 0, n_spikes_i = 0;
  // n_spikes[0] = 0;
  // n_spikes[1] = 0;
  cntr = 0;
  tmpIdx = idxVec[pre_neuron_idx];
  if (pre_neuron_idx < NE){
    n_spikes[0] += 1;    
  }
  else {
    n_spikes[1] += 1;
  }
  while(cntr < nPostNeurons[pre_neuron_idx]) {
    post_neuron_idx = sparseConVec[tmpIdx + cntr];
    cntr += 1;
    if(pre_neuron_idx < NE) {
      /* --    E-to-E    -- */      
      if(post_neuron_idx < NE) {
	g_e[post_neuron_idx] += Jee; // * syn_delay_buffer[pre_neuron_idx][0];
      }
      /* --    E-to-I    -- */      
      else {
	g_e[post_neuron_idx] += Jie; // * syn_delay_buffer[pre_neuron_idx][0];		
      }
    }
    else {
      /* --    I-to-E    -- */      
      if(post_neuron_idx < NE) {
	g_i[post_neuron_idx] += Jei; //  * syn_delay_buffer[pre_neuron_idx][0];	
      }
      /* --    I-to-I    -- */      
      else {
	g_i[post_neuron_idx] += Jii; //  * syn_delay_buffer[pre_neuron_idx][0];		
      }
    }
  }
}





void propagate_spikes_stp(unsigned int pre_neuron_idx) {
  // update all the post synaptic neurons to presynaptic neuron
  unsigned int tmpIdx, cntr, post_neuron_idx;
  double stp_factor = 1.0;
  cntr = 0;
  tmpIdx = idxVec[pre_neuron_idx];

  if(pre_neuron_idx < NE) {
    double stp_u_old = stp_u[pre_neuron_idx];
    stp_u[pre_neuron_idx] += stp_U * (1.0 - stp_u_old);
    stp_x[pre_neuron_idx] -= stp_u[pre_neuron_idx] * stp_x_old[pre_neuron_idx];
    stp_factor = stp_A * stp_u[pre_neuron_idx] * stp_x_old[pre_neuron_idx];
    //    std::cout << "oh yeah!" << stp_factor << "\n";
  }
  
  if (pre_neuron_idx < NE){
    n_spikes[0] += 1;    
  }
  else {
    n_spikes[1] += 1;
  }

  while(cntr < nPostNeurons[pre_neuron_idx]) {
    post_neuron_idx = sparseConVec[tmpIdx + cntr];
    cntr += 1;
    if(pre_neuron_idx < NE) {
      /* --    E-to-E    -- */      
      if(post_neuron_idx < NE) {
	// g_e[post_neuron_idx] += Jee_K;
	g_e[post_neuron_idx] += Jee_K * stp_factor;
	// std::cout << "#" << pre_neuron_idx << "->" << post_neuron_idx << ": " << Jee_K * stp_factor << "\n";
      }
      /* --    E-to-I    -- */      
      else {
	g_e[post_neuron_idx] += Jie_K;		
      }
    }
    else {
      /* --    I-to-E    -- */      
      if(post_neuron_idx < NE) {
	g_i[post_neuron_idx] += Jei_K;	
      }
      /* --    I-to-I    -- */      
      else {
	// g_i[post_neuron_idx] += Jii_K;
	g_i[post_neuron_idx] += Jii_K; //  * stp_factor;
      }
    }
  }
}


void update_synaptic_traces(unsigned int spiked_neuron) {
  // update synaptic traces where spiked_neuron is presynaptic
  // currently called only if spiked_neuron < NE
  unsigned int tmpIdx, cntr, con_idx;
  unsigned int pre_tmpIdx, pre_cntr;
  cntr = 0;
  pre_cntr = 0;
  tmpIdx = idxVec[spiked_neuron];
  pre_tmpIdx = pre_idxVec[spiked_neuron];

  /* - - - - - - - - - - - - - - - - - - - - - */
  while(cntr < nPostNeurons[spiked_neuron]) { // spiked neuron is presynaptic to 
    con_idx = tmpIdx + cntr;
    stdp_pre_trace[con_idx] += 1.0;
    // std::cout << "dw = " << stdp_a_minus(stdp_weights[con_idx]) * stdp_post_trace[con_idx] << "\n";
    // std::cout << "scientific:\n" << std::scientific;
    // std::cout << "dw = " << stdp_a_minus(stdp_weights[con_idx]) * stdp_post_trace[con_idx] << "\n";
    // std::cout << "w old = " << stdp_weights[con_idx] << "\n";
    stdp_weights[con_idx] += stdp_a_minus(stdp_weights[con_idx]) * stdp_post_trace[con_idx];

    cntr += 1;
  }

  /* - - - - - - - - - - - - - - - - - - - - - */
  while(pre_cntr < n_pre_neurons[spiked_neuron]) { // spiked neuron is postsynaptic to
    con_idx = pre_tmpIdx + pre_cntr;
    stdp_post_trace[con_idx] += 1.0;
    
    // std::cout << "scientific:\n" << std::scientific;
    // std::cout << "dw in post = " << stdp_a_plus(stdp_weights[con_idx]) * stdp_pre_trace[con_idx] << "\n";


    stdp_weights[con_idx] += stdp_a_plus(stdp_weights[con_idx]) * stdp_pre_trace[con_idx];
    pre_cntr += 1;
  }

  for(con_idx =0; con_idx < n_connections; ++con_idx) {
    if(stdp_weights[con_idx] < 0) {
      stdp_weights[con_idx] = 0;
    }

    //    std::cout << stdp_weights[0] << "\n";
    // if(stdp_weights[con_idx] > stdp_max_weight + 0.2) {
    //   std::cout << "con #" << con_idx << " exceeds w_max"  << "\n";
    // }
  }
  
  
}

void propagate_spikes_stp_stdp(unsigned int spiked_neuron) {
  // update all the post synaptic neurons to presynaptic neuron
  unsigned int tmpIdx, cntr, post_neuron_idx, post_con_idx;
  double stp_factor = 1.0;
  cntr = 0;
  tmpIdx = idxVec[spiked_neuron];

  /* STP */
  if(spiked_neuron < NE) {
    double stp_u_old = stp_u[spiked_neuron];
    stp_u[spiked_neuron] += stp_U * (1.0 - stp_u_old);
    stp_x[spiked_neuron] -= stp_u[spiked_neuron] * stp_x_old[spiked_neuron];
    stp_factor = stp_A * stp_u[spiked_neuron] * stp_x_old[spiked_neuron];
  }

  /* STDP */
  if(spiked_neuron < NE){
    update_synaptic_traces(spiked_neuron);
  }
  
  
  if (spiked_neuron < NE){
    n_spikes[0] += 1;    
  }
  else {
    n_spikes[1] += 1;
  }

  while(cntr < nPostNeurons[spiked_neuron]) {
    post_con_idx = tmpIdx + cntr;    
    post_neuron_idx = sparseConVec[post_con_idx];
    cntr += 1;
    if(spiked_neuron < NE) {
      /* --    E-to-E    -- */      
      if(post_neuron_idx < NE) {
	// g_e[post_neuron_idx] += Jee_K;
	g_e[post_neuron_idx] += Jee_K * stp_factor + stdp_weights[post_con_idx] * IS_STDP_CON[post_con_idx];
      }
      /* --    E-to-I    -- */      
      else {
	g_e[post_neuron_idx] += Jie_K;		
      }
    }
    else {
      /* --    I-to-E    -- */      
      if(post_neuron_idx < NE) {
	g_i[post_neuron_idx] += Jei_K;	
      }
      /* --    I-to-I    -- */      
      else {
	// g_i[post_neuron_idx] += Jii_K;
	g_i[post_neuron_idx] += Jii_K; //  * stp_factor;
      }
    }
  }
}



// STDP 
void propagate_spikes_stdp(unsigned int spiked_neuron) {
  // update all the post synaptic neurons to presynaptic neuron
  unsigned int tmpIdx, cntr, post_neuron, post_con_idx;
  //  unsigned int pre_tmpIdx, pre_cntr, pre_neuron, pre_con_idx;  
  cntr = 0;
  //  pre_cntr = 0;
  tmpIdx = idxVec[spiked_neuron];
  //  pre_tmpIdx = pre_idxVec[spiked_neuron];

  if (spiked_neuron < NE){
    n_spikes[0] += 1;    
  }
  else {
    n_spikes[1] += 1;
  }

  if(spiked_neuron < NE){
    update_synaptic_traces(spiked_neuron);
  }

  while(cntr < nPostNeurons[spiked_neuron]) {
    post_con_idx = tmpIdx + cntr;
    post_neuron = sparseConVec[post_con_idx];

    // - - - - //
    cntr += 1;
    if(spiked_neuron < NE) {
      /* --    E-to-E    -- */      
      if(post_neuron < NE) {
	g_e[post_neuron] += Jee_K + stdp_weights[post_con_idx] * IS_STDP_CON[post_con_idx];
      }
      /* --    E-to-I    -- */      
      else {
	g_e[post_neuron] += Jie_K;		
      }
    }
    else {

      /* --    I-to-E    -- */      
      if(post_neuron < NE) {
	g_i[post_neuron] += Jei_K;	
      }
      /* --    I-to-I    -- */      
      else {
	g_i[post_neuron] += Jii_K;		
      }
    }
  }
}


void get_pop_rates(double time_interval) {
  /* if(cur_t > discard_time) { */
  pop_rate_e = (double)n_spikes[0] / (time_interval * NE);
  pop_rate_i = (double)n_spikes[1] / (time_interval * NI);
  /* } */
  n_spikes[0] = 0;
  n_spikes[1] = 0;
}

void detect_spikes(double t) {
  // detect if Vm > V_threshold and add 1 to g_x vector

  for (unsigned int neuron_idx = 0; neuron_idx < N; ++neuron_idx) {
    if(Vm[neuron_idx] >= V_threshold[neuron_idx]) {
      Vm[neuron_idx] = V_reset;

      // std::cout << "spk detected :: spkd neuron :: " << neuron_idx << "\n";

      if(t > discard_time) {      
	if(STDP_ON && STP_ON) {
	  propagate_spikes_stp_stdp(neuron_idx);
	}
	else if(STP_ON) {
	  propagate_spikes_stp(neuron_idx);
	}
	else if(STDP_ON){
	  propagate_spikes_stdp(neuron_idx);	  
	}
	else {
	  propagate_spikes(neuron_idx);
	}
      }
      else {
	propagate_spikes(neuron_idx);
      }

      // update threshold
      //      V_threshold[neuron_idx] += d_threshold;
      
      if (t > discard_time) {
	push_spike((t - discard_time) * 1e-3, neuron_idx);
      }
      // n_spikes_now += 1;
    }
  }
  // get_pop_rates(t);
}

void detect_spikes_brunel(double t) {
  // detect if Vm > V_threshold and add 1 to g_x vector
  for (unsigned int neuron_idx = 0; neuron_idx < N; ++neuron_idx) {
    //
    if(syn_delay_buffer[neuron_idx][0]) {
      // update threshold
      V_threshold[neuron_idx] += d_threshold;
      // 
      propagate_spikes_brunel(neuron_idx);
    }
    //
    if(Vm[neuron_idx] >= V_threshold[neuron_idx]) {
      Vm[neuron_idx] = V_reset;
      /* propagate_spikes_brunel(neuron_idx); */
      /* int k = 0; */
      syn_delay_buffer[neuron_idx][n_delay_bins - 1] += 1;

      if (t > discard_time) {
	push_spike((t - discard_time) * 1e-3, neuron_idx);
      }
    }
  }
}

// void record_state(double *x, const char *filename) {
//   FIlE *fp = fopen(filename, "w");
// }

void ProgressBar(float progress, float me, float mi) {
    int barWidth = 31;
    std::cout << "Progress: [";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)  std::cout << "\u25a0"; //std::cout << "=";
      else std::cout << " ";
    }
    std::cout << std::fixed;    
    std::cout << "] " << int(progress * 100.0) << "% done | << t = " << std::setprecision(2) << cur_t * 1e-3 << " mE = " << std::setprecision(2) << me << " mI = " << std::setprecision(2) << mi << "\r";
    std::cout.flush();
    if(progress == 1.) {
      //      std::cout << std::endl;
      std::cout << std::fixed;    
      std::cout << "] " << int(progress * 100.0) << "% done | << t = " << std::setprecision(2) << cur_t * 1e-3 << " mE = " << std::setprecision(2) << me << " mI = " << std::setprecision(2) << mi << std::endl;
      std::cout.flush();
    }
}

bool AreSame(float a, float b) {
    return fabs(a - b) < dt;
}


void integrate_test() {

  discard_time = 0.0;
  
  double Vm_new, dt_over_tau, exp_dt_ovet_tau, exp_tau_e, exp_tau_i, exp_tau_thresh, dt_over_tau_th;
  //
  //  std::vector<double> rates_in_window(N);

  // stp
  double exp_tau_stp_d, exp_tau_stp_f;
  exp_tau_stp_f = exp(-dt / stp_tau_f);
  exp_tau_stp_d = exp(-dt / stp_tau_d);  
  
  /*  double V_e_tilde, V_i_tilde; */
  
  dt_over_tau = dt / tau_membrane;
  double dt_tau_times_rest = dt_over_tau * V_rest;
  dt_over_tau_th = dt / tau_threshold;
  exp_dt_ovet_tau = exp(-dt_over_tau);
  exp_tau_e = exp(-dt / tau_e);
  exp_tau_i = exp(-dt / tau_i);
  exp_tau_thresh = exp(-dt / tau_threshold);
  double dt_over_stp_tau_d = dt / stp_tau_d;

  // STDP
  double exp_decay_stdp_pre, exp_decay_stdp_post;
  exp_decay_stdp_pre = exp(-dt / stdp_tau_pre);
  exp_decay_stdp_post = exp(-dt / stdp_tau_post);

  //
  FILE *fpw = fopen("./data/all_weights.txt", "w");  
  FILE *fpvm = fopen("./data/weights.txt", "w");
  FILE *fpvt = fopen("./data/pre_trace.txt", "w");
  FILE *fpge = fopen("./data/post_trace.txt", "w");
  FILE *fp_rates_i = fopen("./data/pop_rates_i.txt", "w");
  FILE *fp_rates_e = fopen("./data/pop_rates_e.txt", "w");  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - */
  float input_ext;
  float input_ext_e, input_ext_i;


  input_ext_e = Je0 * v_ext * sqrt_K; // * tau_membrane;
  input_ext_i = Ji0 * v_ext * sqrt_K; // * tau_membrane;


  std::cout << "I_ext = " << input_ext_e << "\n";

  //
  float firing_period = 145.49;
  float delta_t = 1; // (ms) time difference between the firing times of neighbouring neurons
  t_stop = 6  * firing_period + K * delta_t; 
  std::cout << "t_stop = " << t_stop << "\n";
  n_steps = (size_t) (t_stop / dt);  
  // size_t
  std::cout << "n steps = " << n_steps  << "\n";  
  double time_interval = 1e-3 * dt * n_steps / 500;  // in seconds
  std::cout << "time win used to estimate pop avg rates= " << time_interval  << "\n";


  const size_t n_spikes_per_neuron = (size_t) t_stop / firing_period;
  std::vector<std::vector<int> > spk_bins(NE, std::vector<int>(n_spikes_per_neuron));
  std::vector<int> last_spk_bin_idx(NE);
  std::vector<int> last_spk_bin_cntr(NE);  

  for(size_t j = 1; j < NE; ++j){
    for(size_t i = 0; i < n_spikes_per_neuron; ++i) {
      spk_bins[j][i] =  j * delta_t + i * firing_period + firing_period - (NE - 2) / 2.0; //+ 139.4 
      // spk_bins[j][i] =  firing_period + j * delta_t + i * firing_period;
    }
  }

  for(size_t j = 0; j < 2; ++j){
    for(size_t i = 0; i < n_spikes_per_neuron; ++i) {
      std::cout << spk_bins[j][i] << " ";
    }
    std::cout << std::endl;
  }
  
  for(size_t t_i = 0; t_i < n_steps; ++t_i) {
    /* - - - */
    cur_t = dt * t_i;

    //
    if(STDP_ON) {
      for(size_t k = 0; k < n_connections; ++k) {
	stdp_pre_trace[k] *= exp_decay_stdp_pre;
	stdp_post_trace[k] *= exp_decay_stdp_post;
      }
    }
    for(size_t neuron_idx=0; neuron_idx < N; ++neuron_idx) {
      
      // g_e[neuron_idx] *= exp_tau_e;
      // g_i[neuron_idx] *= exp_tau_i;

      g_e[neuron_idx] = 0.0;
      // adaptive threshold 
      // V_threshold[neuron_idx] = V_threshold_initial;
      //      V_threshold[neuron_idx] += dt_over_tau_th * (V_threshold_initial - V_threshold[neuron_idx]);

      if(neuron_idx < NE) {
	if(neuron_idx == 0) {
	  if(cur_t < 0){
	    input_ext = 0.0;
	  }
	  else {
	    input_ext = 2.000001;
	  }

	}
	else {
	  last_spk_bin_idx[neuron_idx] = spk_bins[neuron_idx][last_spk_bin_cntr[neuron_idx]];
	  if(last_spk_bin_idx[neuron_idx] == cur_t) {
	    Vm[neuron_idx] = 0.0;
	    last_spk_bin_cntr[neuron_idx] += 1;
	  }
	}
      }
      
      else {
	input_ext = input_ext_i;
      }

      if(neuron_idx == 0) {
	Vm_new = (1.0 - dt_over_tau) * Vm[neuron_idx] + dt * (input_ext
      							 + g_e[neuron_idx]
      							 + g_i[neuron_idx])
	  + dt_tau_times_rest;
	Vm[neuron_idx] = Vm_new;
      }

      if(STP_ON) {
	if(neuron_idx < NE) {
	  stp_u[neuron_idx] *= exp_tau_stp_f;
	  stp_x_old[neuron_idx] = stp_x[neuron_idx];
	  stp_x[neuron_idx] += dt_over_stp_tau_d * (1 - stp_x[neuron_idx]);
	}
      }
    }
    // n_spikes =
    detect_spikes(cur_t);
    if (test_stdp && STDP_ON) {
      fprintf(fpvm, "%f %f %f\n", cur_t, Vm[0], g_e[1]);
      fprintf(fpw, "%f ", cur_t);
      for(size_t lll = 0; lll < n_connections; ++lll) {
	fprintf(fpw, "%f ", stdp_weights[lll]);
      }
      fprintf(fpw, "\n");
    }
    else {
      fprintf(fpvm, "%f %f %f %f\n", cur_t, Vm[0], Vm[1], g_e[1]);
    }
    /* - - - */
    if(t_i > 0 && t_i % (unsigned int)((unsigned int)n_steps / 500) == 0) {
      get_pop_rates(time_interval);
      ProgressBar((float)t_i / (float)n_steps, pop_rate_e, pop_rate_i);
      if(cur_t > discard_time) {
	fprintf(fp_rates_e, "%f %f\n", cur_t - discard_time, pop_rate_e);
	fprintf(fp_rates_i, "%f\n", pop_rate_i);
	if (STDP_ON && !test_stdp) {
	  // fprintf(fpvm, "%f %f %f\n", cur_t, stdp_weights[0], Vm[0]);
	  // // fprintf(fpw, "%f ", cur_t);

	  for(size_t lll = 0; lll < n_connections; ++lll) {
	    fprintf(fpw, "%f ", stdp_weights[lll]);
	  }
	  fprintf(fpw, "\n");

	  // fprintf(fpvt, "%f\n", stdp_pre_trace[0]);
	  // fprintf(fpge, "%f\n", stdp_post_trace[0]);
	  
	}
      }
    }
  }

  /* - - - */
  fflush(fpw);
  fclose(fpw);
  fflush(fpvm);
  fclose(fpvm);
  fflush(fpvt);
  fclose(fpvt);
  fflush(fpge);
  fclose(fpge);
  fflush(fp_rates_e);
  fclose(fp_rates_e);
  fflush(fp_rates_i);
  fclose(fp_rates_i);  
}



void integrate() {
  std::cout << "INTEGRATING"  << "\n";
  n_steps = (size_t) (t_stop / dt); // t_stop is set in read_params func to tstop + discard_time
  size_t n_discard_steps = (size_t) discard_time / dt;

  get_ff_input("./data/ff_input.txt");
  
  double Vm_new, dt_over_tau, exp_dt_ovet_tau, exp_tau_e, exp_tau_i, exp_tau_thresh, dt_over_tau_th;
  // stp
  double exp_tau_stp_d, exp_tau_stp_f;
  exp_tau_stp_f = exp(-dt / stp_tau_f);
  exp_tau_stp_d = exp(-dt / stp_tau_d);  
  dt_over_tau = dt / tau_membrane;
  double dt_tau_times_rest = dt_over_tau * V_rest;
  dt_over_tau_th = dt / tau_threshold;
  exp_dt_ovet_tau = exp(-dt_over_tau);
  exp_tau_e = exp(-dt / tau_e);
  exp_tau_i = exp(-dt / tau_i);
  exp_tau_thresh = exp(-dt / tau_threshold);
  double dt_over_stp_tau_d = dt / stp_tau_d;

  // STDP
  double exp_decay_stdp_pre, exp_decay_stdp_post;
  exp_decay_stdp_pre = exp(-dt / stdp_tau_pre);
  exp_decay_stdp_post = exp(-dt / stdp_tau_post);

  // 
  std::cout << "n steps = " << n_steps  << "\n";  

  // store results in text files
  FILE *fpw = fopen("./data/all_weights.txt", "w");  
  FILE *fpvm = fopen("./data/weights.txt", "w");
  FILE *fpvt = fopen("./data/pre_trace.txt", "w");
  FILE *fpge = fopen("./data/post_trace.txt", "w");
  FILE *fp_rates_i = fopen("./data/pop_rates_i.txt", "w");
  FILE *fp_rates_e = fopen("./data/pop_rates_e.txt", "w");  
  // double cur_t;
  double time_interval = 1e-3 * dt * n_steps / 100;  // in seconds
  std::cout << "tim win = " << time_interval << "s" << "\n";
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - */
  float input_ext;
  float input_ext_e, input_ext_i;
  
  if (scaling_type) {
    input_ext_e = Je0 * v_ext; 
    input_ext_i = Ji0 * v_ext;
      
    }
  else {
    input_ext_e = Je0 * v_ext * sqrt_K; 
    input_ext_i = Ji0 * v_ext * sqrt_K; 
  }

  std::cout << "I_ext = " << input_ext_e << "\n";
  // std::cout << "Je0 = " << Je0 << "\n";    
  // std::cout << "sqrt_K = " << sqrt_K << "\n";
  // std::cout << "v_ext = " << v_ext << "\n";    

  for(size_t t_i = 0; t_i < n_steps; ++t_i) {
    //
    if(STDP_ON) {
      for(size_t k = 0; k < n_connections; ++k) { // n_stdp_cons
	stdp_pre_trace[k] *= exp_decay_stdp_pre;
	stdp_post_trace[k] *= exp_decay_stdp_post;
      }
    }
    for(size_t neuron_idx=0; neuron_idx < N; ++neuron_idx) {
      g_e[neuron_idx] *= exp_tau_e;
      g_i[neuron_idx] *= exp_tau_i;

      // adaptive threshold 
      //      V_threshold[neuron_idx] += dt_over_tau_th * (V_threshold_initial - V_threshold[neuron_idx]);

      if(neuron_idx < NE) {
	input_ext = input_ext_e + I_of_t[t_i];
	// std::cout << I_of_t[t_i] << "\n";
	// fake THETA-like  ff input at 4Hz
	// * (1.0 + exp(50.0 * cos(2 * 3.142 * 4e-3 * cur_t - 2.0 * neuron_idx * 3.142 / (double)NE) - 500.0 ));
      }
      else {
	input_ext = input_ext_i + I_of_t[t_i];
      }


      Vm_new = (1.0 - dt_over_tau) * Vm[neuron_idx] + dt * (input_ext
      							    + g_e[neuron_idx]
      							    + g_i[neuron_idx])
      	+ dt_tau_times_rest;


      /* Vm_new = (1.0 - dt_over_tau) * Vm[neuron_idx] + dt * (input_ext */
      /* 							    + g_e[neuron_idx] */
      /* 							    + g_i[neuron_idx]) */
      /* 	+ dt_tau_times_rest; */
      
      
      

      Vm[neuron_idx] = Vm_new;

      if(STP_ON) {
	if(neuron_idx < NE) {
	  stp_u[neuron_idx] *= exp_tau_stp_f;
	  stp_x_old[neuron_idx] = stp_x[neuron_idx];
	  stp_x[neuron_idx] += dt_over_stp_tau_d * (1 - stp_x[neuron_idx]);
	}
      }
    }
    /* - - - */
    cur_t = dt * t_i;
    // n_spikes =
    detect_spikes(cur_t);

    /* - - - */
    // if (STDP_ON) {
    //   for(size_t lll = 0; lll < n_stdp_connections; ++lll) {
    // 	fprintf(fpw, "%f ", stdp_weights[lll]);
    //   }
    //   fprintf(fpw, "\n");
    // }

    //    std::cout << stdp_weights[0] << "\n";

    //    fprintf(fpvm, "%f %f %f %f\n", cur_t, Vm[0], Vm[1], g_e[1]);



	// if (STDP_ON) {
	//   for(size_t lll = 0; lll < n_connections; ++lll) {
	//     //	    if(IS_STDP_CON[lll]) {
	//       fprintf(fpw, "%f ", stdp_weights[lll]);
	//       //	    }
	//   }
	//   fprintf(fpw, "\n");
	// }


    /* - - - - - */
    if((t_i >= discard_time && t_i % (unsigned int)((unsigned int)n_steps / 100) == 0) || t_i == n_steps-1) {
      get_pop_rates(time_interval);
      ProgressBar((float)t_i / (float)n_steps, pop_rate_e, pop_rate_i);
      if(t_i >= n_discard_steps) {
	fprintf(fp_rates_e, "%f %f\n", cur_t - discard_time, pop_rate_e);
	fprintf(fp_rates_i, "%f\n", pop_rate_i);


	if (STDP_ON) {
	  for(size_t lll = 0; lll < n_connections; ++lll) {
	    if(IS_STDP_CON[lll]) {
	      fprintf(fpw, "%f ", stdp_weights[lll]);
	    }
	  }
	  fprintf(fpw, "\n");
	}


	//   fprintf(fpw, "\n");
	  // fprintf(fpvt, "%f\n", stdp_pre_trace[0]);
	  // fprintf(fpge, "%f\n", stdp_post_trace[0]);
	// }
      }
    }
  }

  /* - - - */
  fflush(fpw);
  fclose(fpw);
  fflush(fpvm);
  fclose(fpvm);
  fflush(fpvt);
  fclose(fpvt);
  fflush(fpge);
  fclose(fpge);
  fflush(fp_rates_e);
  fclose(fp_rates_e);
  fflush(fp_rates_i);
  fclose(fp_rates_i);  
}


void integrate_brunel() {
  std::cout << "INTEGRATING BRUNEL"  << "\n";
  n_steps = (size_t) (t_stop / dt); // t_stop is set in read_params func to tstop + discard_time
  size_t n_discard_steps = (size_t) discard_time / dt;
  get_ff_input("./data/ff_input.txt");
  double Vm_new, dt_over_tau, exp_dt_ovet_tau, exp_tau_e, exp_tau_i, exp_tau_thresh, dt_over_tau_th;
  // stp
  double exp_tau_stp_d, exp_tau_stp_f;
  exp_tau_stp_f = exp(-dt / stp_tau_f);
  exp_tau_stp_d = exp(-dt / stp_tau_d);  
  dt_over_tau = dt / tau_membrane;
  double dt_tau_times_rest = dt_over_tau * V_rest;
  dt_over_tau_th = dt / tau_threshold;
  exp_dt_ovet_tau = exp(-dt_over_tau);
  exp_tau_e = exp(-dt / tau_e);
  exp_tau_i = exp(-dt / tau_i);
  exp_tau_thresh = exp(-dt / tau_threshold);
  double dt_over_stp_tau_d = dt / stp_tau_d;

  // double R = 1.0;
  
  /* // STDP */
  /* double exp_decay_stdp_pre, exp_decay_stdp_post; */
  /* exp_decay_stdp_pre = exp(-dt / stdp_tau_pre); */
  /* exp_decay_stdp_post = exp(-dt / stdp_tau_post); */
  std::cout << "n steps = " << n_steps  << "\n";  
  // store results in text files
  FILE *fpw = fopen("./data/all_weights.txt", "w");  
  FILE *fpvm = fopen("./data/weights.txt", "w");
  FILE *fpvt = fopen("./data/pre_trace.txt", "w");
  FILE *fpge = fopen("./data/post_trace.txt", "w");
  FILE *fp_rates_i = fopen("./data/pop_rates_i.txt", "w");
  FILE *fp_rates_e = fopen("./data/pop_rates_e.txt", "w");  
  // double cur_t;
  double time_interval = 1e-3 * dt * n_steps / 100;  // in seconds
  std::cout << "tim win = " << time_interval << "s" << "\n";
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - */
  float input_ext;
  float input_ext_e, input_ext_i;
  gsl_rng * r = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(r, 1234);

  
  input_ext_e = K * Je0 * v_ext;
  input_ext_i = K * Ji0 * v_ext;

  double ext_noise_std_e = 0.1 * sqrt(K * v_ext * tau_membrane) / sqrt(dt); 
  double ext_noise_std_i = 0.10 * sqrt(0.25 * K * v_ext * tau_membrane) / sqrt(dt);   
  /* if (scaling_type) { */
  /*   input_ext_e = Je0 * v_ext;  */
  /*   input_ext_i = Ji0 * v_ext; */
  /*   } */
  /* else { */
  /*   input_ext_e = Je0 * v_ext * sqrt_K;  */
  /*   input_ext_i = Ji0 * v_ext * sqrt_K;  */
  /* } */
  std::cout << "I_ext = " << input_ext_e << "\n";
  double v_thresh_brunel = V_threshold_initial / (Je0 * K * tau_membrane);
  std::cout <<  "v_ext / v_thresh = " << v_ext / v_thresh_brunel << "\n";
  // std::cout << "Je0 = " << Je0 << "\n";    
  // std::cout << "sqrt_K = " << sqrt_K << "\n";
  std::cout << "v_ext = " << v_ext << "\n";

  std::cout << "v_ext = " << v_ext << "\n";
  
  for(size_t t_i = 0; t_i < n_steps; ++t_i) {
    /* if(STDP_ON) { */
    /*   for(size_t k = 0; k < n_connections; ++k) { // n_stdp_cons */
    /* 	stdp_pre_trace[k] *= exp_decay_stdp_pre; */
    /* 	stdp_post_trace[k] *= exp_decay_stdp_post; */
    /*   } */
    /* } */
    for(size_t neuron_idx=0; neuron_idx < N; ++neuron_idx) {
      g_e[neuron_idx] *= exp_tau_e;
      g_i[neuron_idx] *= exp_tau_i;

      // adaptive threshold 
      V_threshold[neuron_idx] += dt_over_tau_th * (V_threshold_initial - V_threshold[neuron_idx]);

      if(neuron_idx < NE) {
	input_ext = input_ext_e + I_of_t[t_i]+ ext_noise_std_e * gsl_ran_gaussian(r, 1.0) ;
	// std::cout << I_of_t[t_i] << "\n";
	// fake THETA-like  ff input at 4Hz
	// * (1.0 + exp(50.0 * cos(2 * 3.142 * 4e-3 * cur_t - 2.0 * neuron_idx * 3.142 / (double)NE) - 500.0 ));
      }
      else {
	input_ext = input_ext_i + I_of_t[t_i]+ ext_noise_std_i * gsl_ran_gaussian(r, 1.0) ;
      }
      /* Vm_new = (1.0 - dt_over_tau) * Vm[neuron_idx] + dt * (input_ext */
      /* 							    + g_e[neuron_idx] */
      /* 							    + g_i[neuron_idx]) */
      /* 	+ dt_tau_times_rest; */

      Vm_new = (1 - dt_over_tau) * Vm[neuron_idx] + (input_ext + g_e[neuron_idx] + g_i[neuron_idx]) * dt_over_tau;

      Vm[neuron_idx] = Vm_new;

      // Vm[neuron_idx] *= exp_tau_e;
      // Vm[neuron_idx] += (input_ext + g_e[neuron_idx] + g_i[neuron_idx]) / tau_e;
      

      /* if(STP_ON) { */
      /* 	if(neuron_idx < NE) { */
      /* 	  stp_u[neuron_idx] *= exp_tau_stp_f; */
      /* 	  stp_x_old[neuron_idx] = stp_x[neuron_idx]; */
      /* 	  stp_x[neuron_idx] += dt_over_stp_tau_d * (1 - stp_x[neuron_idx]); */
      /* 	} */
      /* } */
    }
    /* - - - */
    cur_t = dt * t_i;
    // n_spikes =
    detect_spikes_brunel(cur_t);

    shift_matrix(syn_delay_buffer, N, n_delay_bins);    
    
    /* - - - */
    // if (STDP_ON) {
    //   for(size_t lll = 0; lll < n_stdp_connections; ++lll) {
    // 	fprintf(fpw, "%f ", stdp_weights[lll]);
    //   }
    //   fprintf(fpw, "\n");
    // }
    //    std::cout << stdp_weights[0] << "\n";
    fprintf(fpvm, "%f %f %f %f\n", cur_t, Vm[0], V_threshold[0], g_e[1]);
	// if (STDP_ON) {
	//   for(size_t lll = 0; lll < n_connections; ++lll) {
	//     //	    if(IS_STDP_CON[lll]) {
	//       fprintf(fpw, "%f ", stdp_weights[lll]);
	//       //	    }
	//   }
	//   fprintf(fpw, "\n");
	// }
    /* - - - - - */
    if((t_i >= discard_time && t_i % (unsigned int)((unsigned int)n_steps / 100) == 0) || t_i == n_steps-1) {
      get_pop_rates(time_interval);
      ProgressBar((float)t_i / (float)n_steps, pop_rate_e, pop_rate_i);
      if(t_i >= n_discard_steps) {
	fprintf(fp_rates_e, "%f %f\n", cur_t - discard_time, pop_rate_e);
	fprintf(fp_rates_i, "%f\n", pop_rate_i);
	/* if (STDP_ON) { */
	/*   for(size_t lll = 0; lll < n_connections; ++lll) { */
	/*     if(IS_STDP_CON[lll]) { */
	/*       fprintf(fpw, "%f ", stdp_weights[lll]); */
	/*     } */
	/*   } */
	/*   fprintf(fpw, "\n"); */
	/* } */
	//   fprintf(fpw, "\n");
	  // fprintf(fpvt, "%f\n", stdp_pre_trace[0]);
	  // fprintf(fpge, "%f\n", stdp_post_trace[0]);
	// }
      }
    }
  }
  /* - - - */
  fflush(fpw);
  fclose(fpw);
  fflush(fpvm);
  fclose(fpvm);
  fflush(fpvt);
  fclose(fpvt);
  fflush(fpge);
  fclose(fpge);
  fflush(fp_rates_e);
  fclose(fp_rates_e);
  fflush(fp_rates_i);
  fclose(fp_rates_i);

  gsl_rng_free(r);
}

#endif
