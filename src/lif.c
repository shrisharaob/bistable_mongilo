#include "utils.h"



int main(void) {
  read_params();

  init_state_vectors();

   gen_conmat();
  /* recip_ei(); */

  if(test_stdp) {
    integrate_test();
  }
  else {
    if(scaling_type == 2){
      integrate_brunel();
    }
    else {
      integrate();
    }
  }

  delete_state_vectors();
  return 0;
}
