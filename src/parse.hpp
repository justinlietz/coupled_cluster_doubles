#ifndef parse_hpp_
#define parse_hpp_
#include "SPBasis.hpp"
char* parse(int argc,
            char **argv,
            SPBasis **basis,
            double *tolerance,
            int *saveMemory,
            char **timeFile,
            char *timeMode,
            char **modelFile,
            int *verbose,
						int rank);
void parseFree(char *mem);
#endif
