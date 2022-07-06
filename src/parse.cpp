#include <argp.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <ios>
#include <ios>
#include <fstream>

#include "parse.hpp"
#include "SPBasis.hpp"
#include "PairingSPBasis.hpp"
#include "ElectronGasSPBasis.hpp"
#include "InfMatterSPBasis.hpp"

const char *argp_program_version =
  "ccd 1.0";
const char *argp_program_bug_address =
  "<lietz@nscl.msu.edu>";

/* Program documentation. */
static char doc[] =
  "CCD -- does some nucleus stuff i dont know man";

/* A description of the arguments we accept. */
static char args_doc[] = "";

#define pairing_key 0x100
#define infMatter_key 0x101
#define electronGas_key 0x102
#define qDots_key 0x103
#define config_key 'c'
#define density_key 'd'
#define g_key 'g'
#define nParticleShells_key 'p'
#define nParticles_key 'n'
#define nPairStates_key 'l'
#define r_s_key 'r'
#define nShells_key 's'
#define tolerance_key 't'
#define tzMax_key 'z'
#define xi_key 'x'
#define saveMemory_key 'm'
#define time_key 'T'
#define timeOverwrite_key 'W'
#define model_key 'M'
#define verbose_key 'v'

struct arguments{
  int *basis;
  char *configFile;
  double *density;
  double *g;
  int *nParticleShells;
  int *nParticles;
  int *nPairStates;
  double *r_s;
  int *nShells;
  double *tolerance;
  int *tzMax;
  double *xi;
  int *saveMemory;
  char **timeFile;
  char *timeMode;
  char **modelFile;
  int *verbose;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state){
  struct arguments *arguments = (struct arguments*)state->input;
  switch(key){
    case pairing_key:
    case infMatter_key:
    case electronGas_key:
    case qDots_key:
      *(arguments->basis) = key - 0x100;
      break;
    case config_key:
      arguments->configFile = arg;
      break;
    case density_key:
      *(arguments->density) = atof(arg);
      break;
    case g_key:
      *(arguments->g) = atof(arg);
      break;
    case nParticleShells_key:
      *(arguments->nParticleShells) = atoi(arg);
      break;
    case nParticles_key:
      *(arguments->nParticles) = atoi(arg);
      break;
    case nPairStates_key:
      *(arguments->nPairStates) = atoi(arg);
      break;
    case r_s_key:
      *(arguments->r_s) = atof(arg);
      break;
    case nShells_key:
      *(arguments->nShells) = atoi(arg);
      break;
    case tolerance_key:
      *(arguments->tolerance) = atof(arg);
      break;
    case tzMax_key:
      *(arguments->tzMax) = atoi(arg);
      break;
    case xi_key:
      *(arguments->xi) = atof(arg);
      break;
    case saveMemory_key:
      *(arguments->saveMemory) = atoi(arg);
      break;
    case time_key:
      *(arguments->timeFile) = arg;
      break;
    case timeOverwrite_key:
      *(arguments->timeMode) = 'w';
      break;
    case model_key:
      *(arguments->modelFile) = arg;
      break;
    case verbose_key:
      *(arguments->verbose) = atoi(arg);
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp_option options[] = {
  {"config", config_key, "FILE", 0, "use FILE to get default configuration"},
  {"saveMemory", saveMemory_key, "INT", 0, "memory saving mode (0 for off, 1 for on)"},
  {"time", time_key, "FILE", 0, "record timing info for load balancer in FILE"},
  {"timeOverwrite", timeOverwrite_key, 0, 0, "overwrite existing timings"},
  {"model", model_key, "FILE", 0, "use model info in FILE for load balancer"},
  {"verbose", verbose_key, "INT", 0, "verbosity level"},
  {0, 0, 0, 0, "Basis Choices:"},
  {"pairing", pairing_key, 0, 0, "pairing basis"},
  {"infMatter", infMatter_key, 0, 0, "infinite matter basis"},
  {"qDots", qDots_key, 0, 0, "quantum dots basis"},
  {"electronGas", electronGas_key, 0, 0, "electron gas basis"},
  {0, 0, 0, 0, "Basis-Specific options:"},
  {"density", density_key, "FLOAT", 0, "particle density in fm^-3"},
  {"g", g_key, "FLOAT", 0, "pairing interaction strength parameter"},
  {"nParticleShells", nParticleShells_key, "INT", 0, "how many shells of particles"},
  {"nParticles", nParticles_key, "INT", 0, "Number of particles"},
  {"nPairStates", nPairStates_key, "INT", 0, "Number of degenerate levels"},
  {"r_s", r_s_key, "FLOAT", 0, "Wigner-Seitz Radius"},
  {"nShells", nShells_key, "INT", 0, "how many non-empty energy shells (Fermi spheres)"},
  {"tolerance", tolerance_key, "FLOAT", 0, "tolerance of CCD energy for iterative solver"},
  {"tzMax", tzMax_key, "INT", 0, "number of species, 1 for neutrons, 2 for protons and neutrons"},
  {"xi", xi_key, "FLOAT", 0, "single particle level energy"},
  {0}
};

char* parse(int argc,
            char **argv,
            SPBasis **basis,
            double *tolerance,
            int *saveMemory,
            char **timeFile,
            char *timeMode,
            char **modelFile,
            int *verbose,
						int rank){
  char *parseMem = NULL;

  // default parameters. Can be overridden
  double xi = 1.0;
  double g = 0.5;
  double density = 0.08;
  double r_s = 0.5;
  int tzMax = 1;
  int nShells = 4;
  int nParticleShells = 2;
  int nParticles = 4;
  int nPairStates = 4;
  int basisNum = 1;

  *timeMode = 'a';

  struct arguments arguments;

  arguments.configFile = 0;

  arguments.basis = &basisNum;
  arguments.density = &density;
  arguments.g = &g;
  arguments.nParticleShells = &nParticleShells;
  arguments.nParticles = &nParticles;
  arguments.nPairStates = &nPairStates;
  arguments.r_s = &r_s;
  arguments.nShells = &nShells;
  arguments.tzMax = &tzMax;
  arguments.xi = &xi;

  arguments.tolerance = tolerance;
  arguments.timeFile = timeFile;
  arguments.timeMode = timeMode;
  arguments.modelFile = modelFile;
  arguments.verbose = verbose;
  arguments.saveMemory = saveMemory;

  static struct argp argp = {options, parse_opt, args_doc, doc};

  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  if(arguments.configFile){
    std::ifstream file(arguments.configFile);
    file.seekg(0, std::ios::end);
    int config_len = file.tellg();
    file.seekg(0, std::ios::beg);
    char *config_argvv = new char[config_len];
    char *config_argv[config_len + 1];
    file.read(config_argvv, config_len);
    config_argvv[config_len] = 0;
    file.close();
    int config_argc = 1;
    char *scan = config_argvv;
    while (config_argv[config_argc] = strtok(scan, " \t\r\n")){
      config_argc++;
      scan = NULL;
    }
    config_argv[0] = argv[0];
    argp_parse(&argp, config_argc, config_argv, 0, 0, &arguments);
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    parseMem = config_argvv;
  }

  switch(basisNum){
    case 0:
      *basis = new PairingSPBasis (0,xi,g,nPairStates,nParticles);
      break;
    case 1:
      *basis = new InfMatterSPBasis (1,density,tzMax,nShells,nParticleShells);
      break;
    case 2:
      *basis = new ElectronGasSPBasis (2,r_s,1,nShells,nParticleShells);
      break;
    default:
      printf("not ready\n");
      exit(1);
  }

  // Print out input conditions
	if( rank == 0 ){
	  if(*verbose >= 1){
	    switch(basisNum){
	      case 0:
	        printf("Basis 0: Pairing\n");
	        printf("xi = %f, g = %f, nPairStates = %d, nParticles = %d, nSpstates = %zu\n",xi,g,nPairStates,nParticles,(*basis)->nSpstates);
	        break;
	      case 1:
	        printf("Basis 1: infMatter\n");
	        printf("density = %f, tzMax = %d, nShells = %d, nParticleShells = %d, nSpStates = %zu, nParticles = %zu\n",
	               density,tzMax,nShells,nParticleShells,(*basis)->nSpstates,(*basis)->nParticles);
	        break;
	      case 2:
	        printf("Basis 2: electronGas\n");
	        printf("r_s = %f, nShells = %d, nParticleShells = %d, nSpStates = %zu, nParticles = %zu\n",
	               r_s,nShells,nParticleShells,(*basis)->nSpstates,(*basis)->nParticles);
	        break;
	    }
	  }
	}
  return parseMem;
}

void parseFree(char *parseMem){
  if(parseMem != NULL){
    delete[] parseMem;
  }
}
