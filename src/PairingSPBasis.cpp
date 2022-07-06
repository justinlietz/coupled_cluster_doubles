// Copyright (c) 2015-2016, Justin Gage Lietz
// All rights reserved.

// Need to write a PDF about this biz.

#include "SPBasis.hpp"
#include "PairingSPBasis.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

PairingSPBasis::PairingSPBasis(std::size_t basisIndicatorIn, double xiIn, double gIn,
		std::size_t nPairStatesIn, std::size_t nParticlesIn){
	this->basisIndicator = basisIndicatorIn;
	this->xi = xiIn;
	this->g = gIn;
	this->nSpstates = 2*nPairStatesIn;
	this->nParticles = nParticlesIn;
	this->nChannels = 3;
	this->generateIndexMap();
	this->generateBasis();
}


void PairingSPBasis::generateIndexMap(){

	this->indexMap = new int* [nSpstates];
	for(std::size_t i = 0; i < this->nSpstates; i++){
		this->indexMap[i] = new int[2];
	}

	std::size_t count = 0;
	for(std::size_t i=0; i<nSpstates/2; i++){
		this->indexMap[count][0] = i+1;
		this->indexMap[count][1] = +1;
		count++;
		this->indexMap[count][0] = i+1;
		this->indexMap[count][1] = -1;
		count++;
	}

} // end generateIndexMap


// requires generate index map first
void PairingSPBasis::generateBasis(){

	this->spEnergy = new double[this->nSpstates];

	double E;
	// vector<state> psi; // vector of sp states
	for(std::size_t i = 0; i < this->nSpstates; i++){
		E = (indexMap[i][0] - 1.0)*xi;
		this->spEnergy[i] = E;
	}
} // end generateBasis


int PairingSPBasis::checkSympqrs(std::size_t p, std::size_t q, std::size_t r, std::size_t s){
	return this->indexMap[p][1] + this->indexMap[q][1] == this->indexMap[r][1] + this->indexMap[s][1];
} // end checkSympqrs

int PairingSPBasis::checkModSympqrs(std::size_t p, std::size_t q, std::size_t r, std::size_t s){
	return this->indexMap[p][1] - this->indexMap[q][1] == this->indexMap[r][1] - this->indexMap[s][1];
} // end checkSympqrs

int PairingSPBasis::checkChanSym(std::size_t p, std::size_t q, std::size_t ichan){
	return indexMap[p][1] + indexMap[q][1] == this->chanValue[ichan].chanSz;
} // end checkChanSym

int PairingSPBasis::checkChanModSym(std::size_t p, std::size_t q, std::size_t ichan){
	return indexMap[p][1] - indexMap[q][1] == this->chanValue[ichan].chanSz;
} // end checkChanModSym


void PairingSPBasis::setUpTwoStateChannels(){
	this->setUpChannelValues();
	this->setUpChannelDims();
	this->setUpChannelMaps();
}

void PairingSPBasis::setUpChannelValues(){
	this->chanValue = new PairingBundle[this->nChannels];
	this->chanModValue = new PairingBundle[this->nChannels];


	int channelSzMax = 2;
	std::size_t channelCount = 0;
	for(int ichanSz = -channelSzMax; ichanSz <= channelSzMax; ichanSz = ichanSz +2){
		this->chanValue[channelCount].chanSz = ichanSz;
		this->chanModValue[channelCount].chanSz = ichanSz;
		channelCount++;
	}
} // end setUpChannelValues


void PairingSPBasis::printBasis(){
	std::cout << "# p sz E" << std::endl;
	for(std::size_t i = 0; i < this->nSpstates; i++){
		std::cout << i << " " << this->indexMap[i][0] << " " << this->indexMap[i][1]
			<< " " << this->spEnergy[i] << std::endl;
	}
}

void PairingSPBasis::deallocate(){

	for(std::size_t i = 0; i < this->nSpstates; i++){
		delete [] this->indexMap[i];
	}
	delete [] this->indexMap;
	delete [] this->spEnergy;
	delete [] this->chanValue;
	/* delete [] this->chanDims; */
	/* delete [] this->chanMaps; */
	delete [] this->chanModValue;
	/* delete [] this->chanModDims; */
	/* delete [] this->chanModMaps; */
  // probably won't need these
	/* delete[] this->chan3BValue; */
	/* delete[] this->threeByThreeChanDims; */

} // end deallocate

// for antisymmetrized, can use -0.5*g
double PairingSPBasis::calcTBME(std::size_t p, std::size_t q, std::size_t r, std::size_t s){
	double vout = 0.;

	if(kronecker_del(this->indexMap[p][0],this->indexMap[q][0]) *
			kronecker_del(this->indexMap[r][0],this->indexMap[s][0]) *
			kronecker_del(this->indexMap[p][1], -this->indexMap[q][1]) *
			kronecker_del(-this->indexMap[r][1], this->indexMap[s][1]) == 1){
		vout = -0.5*this->g;

		if( this->indexMap[p][1] == -this->indexMap[r][1] ){
			vout = -vout;
		}
	}
	return vout;
}


double PairingSPBasis::calc_TBME_not_antisym(std::size_t p, std::size_t q, std::size_t r, std::size_t s){
	double vout = 0.;

	if(kronecker_del(this->indexMap[p][0],this->indexMap[q][0]) *
			kronecker_del(this->indexMap[r][0],this->indexMap[s][0]) *
			kronecker_del(this->indexMap[p][1], -this->indexMap[q][1]) *
			kronecker_del(-this->indexMap[r][1], this->indexMap[s][1]) == 1){
		vout = -0.25*this->g;

		if( this->indexMap[p][1] == -this->indexMap[r][1] ){
			vout = -vout;
		}
	}
	return vout;
}

int PairingSPBasis::kronecker_del(int i, int j){
	if(i != j){
		return 0;
	}
	return 1;
} // end kronecker_del


std::size_t PairingSPBasis::TBchanIndexFunction(std::size_t p, std::size_t q){
	// chanSz is {-2,0,2}
	// want {0,1,2}
	int chanSz = this->indexMap[p][1] + this->indexMap[q][1];
	size_t index = (chanSz + 2)/2;
  return this->inverseChannelIndices[index];
}

std::size_t PairingSPBasis::TBmodChanIndexFunction(std::size_t p, std::size_t q){
	// modChanSz is {-2,0,2}
	int modChanSz = this->indexMap[p][1] - this->indexMap[q][1];
	size_t index = (modChanSz + 2)/2;
  return this->inverseModChannelIndices[index];
}

std::size_t PairingSPBasis::spIndex_from3Body(std::size_t q_inv, std::size_t r, std::size_t s){
	printf("PairingSPBasis::spIndex_from3Body called\nthis is not defined\n");
	return -1;
}
int PairingSPBasis::spIndexExists_from3Body(std::size_t q_inv, std::size_t r, std::size_t s){
	printf("PairingSPBasis::spIndexExists_from3Body called\nthis is not defined\n");
	return -1;
}



