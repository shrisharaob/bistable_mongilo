LIF network with STP and STDP
-------

## Setup

**Download**

`git clone git@gitlab.com:rbm-hippo/lif_stp.git <path_to_dir>`

**Requirements**

`libconfig`: http://macappstore.org/libconfig/

python: see requirements.txt


**Build**

`cd src`

`CC="g++ -lconfig++ -lgsl -stdlib=libc++" python setup.py build --force`

Tested on Mac OS 10.14.6

**Configuration**

Edit the file `config.toml` and set the path to `rbm_analysis` folder

`rbm_analysis` can be downloaded as:

`git@gitlab.com:rbm-hippo/rbm_analysis.git <path_to_dir>`

## Usage

Example usage in experiments.ipynb


Default simulation parameters are in the file src/params_bkp.cfg

First import module **simulate**

To change parameters pass them as keyword argument pairs:

The following will run the simulation with 100 excitatory neurons

`import simulate`

`simulate.runsim(NE=100)`

### TOFIX: 

- set parameter types, currently floating point parameters such as time constants must be passed with a decimal point
- move test code to a different folder


## Model

The model consists $`N_E`$ excitatory and $`N_I`$ inhibitory LIF neurons.


Neuron $`i`$ of population $`A`$ i.e. $`(i, A)`$ receives a connection from neuron $`(j, B)`$ with a probability given by:

$`P(C_{AB}^{ij} = 1) = \frac{K}{N_B}`$

The membrane voltage evolves as follows, 

$`\tau_{m} \, \frac{dV_{A}^{i}}{dt} = - (V - V_{rest}) + I_{rec}^i + I_{FF}`$


Reccurent input $`I_{rec}^i`$ is depends on all presynatic spikes,

$`I_{rec}^i = \sum_{B = (E, I)} J_{AB} \sum_j \, C_{AB}^{ij} \sum_k \delta(t - t^j_k)`$



The feedforward input depends on the variable `scaling_type (st)`, 

```math
I_{FF} = 
\left\{ \begin{array}{l}
     \sqrt{K} J_{E0}\, v_{ext} \;\; \text{if st=0} \\
     J_{E0}\, v_{ext} \;\; \text{if st=1} 	 
 \end{array} \right.
 ```

scaling_type = 0, by default. 

### Scaling

```math
J_{AB} = 
\left\{ \begin{array}{l}
     \frac{\tilde{J}^{STP}_{AB}}{\sqrt{K}} + \frac{J^{STDP}}{K} \;\; \text{if st=0} \\
     \frac{\tilde{J}^{STP}_{AB}}{K} + \frac{J^{STDP}}{K} \;\; \text{if st=1} \\	 
 \end{array} \right.
 ```


### Plasticity

$`J_{AB} = J_{AB}^{STP} + J_{AB}^{STDP}`$


Pairwise STDP: http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Basic_STDP_Model

STP: http://www.scholarpedia.org/article/Short-term_synaptic_plasticity#Phenomenological_model


Plasticity can be turned ON (1) and OFF (0) by setting the parameters `STDP_ON` and `STP_ON`, by default they are set to 1.


### Parameters




