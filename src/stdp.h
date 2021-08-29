#include "globals.h"

inline double heavside(double x) {
  if (x < 0) {
      return 0.0;
    }
  else {
    return 1.0;
  }
}

inline double stdp_a_plus(double w) {
  //  return stdp_lr_pre * heavside(stdp_max_weight - w);
  return stdp_lr_pre * (stdp_max_weight - w);
}

inline double stdp_a_minus(double w) {
  //  return -1.0 * stdp_lr_post * w; // use this for asymmetric kernel i.e positive stdp_lr_post
  return -1.0 * stdp_lr_post * (stdp_max_weight - w);
}

