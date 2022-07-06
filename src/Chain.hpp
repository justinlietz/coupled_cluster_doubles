#ifndef Chain_hpp_
#define Chain_hpp_
#include <cstddef>
#include <iostream>
#include <mpi.h>

struct ChainIndex {
  std::size_t channel;
  std::size_t row;
  std::size_t column;
};

MPI_Datatype ChainIndex_MPI();

struct MapDistributedData;

std::size_t MDD_Memory(MapDistributedData *data);

template <typename T>
class Chain {
  private:
  std::size_t *offsets;
  T*** elements;
  T** blocks;
  T* data;
  std::size_t myNChannels;
  std::size_t *nRowss;
  std::size_t *nColumnss;

  public:
  std::size_t startChannel;
  std::size_t endChannel;
  std::size_t nChannels;
  std::size_t maxBufferSize;

  /**
   * The Chain class represents a chain of symmetry blocks.
   * @param startChannel the starting channel of this piece of the chain (inclusive)
   * @param endChannel the terminating channel of this piece of the chain (exclusive)
   * @param nChannels the total number of channels in the chain
   * @param nRowss a list of the number of rows in each block (will be copied)
   * @param nColumnss a list of the number of columns in each block (will be copied)
   */
  Chain(std::size_t startChannel, std::size_t endChannel, std::size_t nChannels, std::size_t *nRowss, std::size_t *nColumnss);

  /**
   * Returns the amount of memory used by a chain class.
   * @param startChannel the starting channel of this piece of the chain (inclusive)
   * @param endChannel the terminating channel of this piece of the chain (exclusive)
   * @param nChannels the total number of channels in the chain
   * @param nRowss a list of the number of rows in each block (will be copied)
   * @param nColumnss a list of the number of columns in each block (will be copied)
   */
  static std::size_t ChainMemory(std::size_t startChannel, std::size_t endChannel, std::size_t nChannels, std::size_t *nRowss, std::size_t *nColumnss);


  ~Chain();

  /**
   * set a block to zero
   * @param channel which channel to set to zero
   */
  void zero(std::size_t channel);

  /**
   * set a chain to zero
   */
  void zero(void);

  /**
   * set a block equal to a block in another chain
   * @param channel which channel to set
   * @param chain which chain
   */
  void set(std::size_t channel, Chain *other);

  /**
   * set a chain equal to another chain
   * @param channel which chain
   */
  void set(Chain *other);

  /**
   * copy the contents of a block into memory
   * @param channel which channel to set
   * @param buffer a buffer of size large enough to hold the block.
   */
  void copy(std::size_t channel, T *buffer);

  /**
   * print out a channel
   * @param channel which channel to print
   */
  void print(std::size_t channel);

  /**
   * permute the blocks of a chain
   * @param channels a list of how channels from this chain should be ordered in the new one.
   * @return the permuted chain
   */
  void permute(std::size_t *channels);

  /**
   * get a pointer to a particular block
   * @param channel which channel u want?
   * @return a pointer to the corresponding block
   */
  inline T* get(std::size_t channel){
    #ifdef DEBUG
    if (channel < this->startChannel){
      std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
      exit(1);
    }
    if (channel >= this->endChannel){
      std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
      exit(1);
    }
    #endif
    return this->blocks[channel - this->startChannel];
  };

  /**
   * get an element of the chain
   * @param channel which channel?
   * @param row which row?
   * @param column which column?
   * @return the specified element
   */
  inline T get(std::size_t channel, std::size_t row, std::size_t column){
    #ifdef DEBUG
    if (channel < this->startChannel){
      std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
      exit(1);
    }
    if (channel >= this->endChannel){
      std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
      exit(1);
    }
    if (row >= this->nRowss[channel - this->startChannel]){
      std::cerr << "error: row out of bounds" << std::endl;
      exit(1);
    }
    if (column >= this->nColumnss[channel - this->startChannel]){
      std::cerr << "error: column out of bounds" << std::endl;
      exit(1);
    }
    #endif
    return this->elements[channel - this->startChannel][row][column];
  };

  /**
   * get an element of the chain
   * @param index a ChainIndex struct specifying the element to get
   * @return the specified element
   */
  inline T get(struct ChainIndex index){
    return this->get(index.channel, index.row, index.column);
  };

  /**
   * set an element of the chain
   * @param channel which channel?
   * @param row which row?
   * @param column which column?
   * @param value the value to set
   */
  inline void set(std::size_t channel, std::size_t row, std::size_t column, T value){
    #ifdef DEBUG
    if (channel < this->startChannel){
      std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
      exit(1);
    }
    if (channel >= this->endChannel){
      std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
      exit(1);
    }
    // What has Peter got in his hot pocketses?! Rowss obvi
    if (row >= this->nRowss[channel - this->startChannel]){
      std::cerr << "error: row out of bounds" << std::endl;
      exit(1);
    }
    if (column >= this->nColumnss[channel - this->startChannel]){
      std::cerr << "error: column out of bounds" << std::endl;
      exit(1);
    }
    #endif
    this->elements[channel - this->startChannel][row][column] = value;
  };

  /**
   * set an element of the chain
   * @param index a ChainIndex struct specifying the element to set
   * @param value the value to set
   */
  inline void set(struct ChainIndex index, T value){
    this->set(index.channel, index.row, index.column, value);
  };

  /**
   * get the number of rows in a block
   * @param channel the block of interest
   * @return the number of rows in that block
   */
  inline std::size_t nRows(std::size_t channel){
    #ifdef DEBUG
    if (channel < this->startChannel){
      std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
      exit(1);
    }
    if (channel >= this->endChannel){
      std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
      exit(1);
    }
    #endif
    return this->nRowss[channel - this->startChannel];
  };

  /**
   * get the number of columns in a block
   * @param channel the block of interest
   * @return the number of columns in that block
   */
  inline std::size_t nColumns(std::size_t channel){
    #ifdef DEBUG
    if (channel < this->startChannel){
      std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
      exit(1);
    }
    if (channel >= this->endChannel){
      std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
      exit(1);
    }
    #endif
    return this->nColumnss[channel - this->startChannel];
  };

  /**
   * Map the calling chain into the "to" chain using indicies in the map. The
   * calling chain element in a particular channel, row, and column is mapped
   * to the index specified by the element of the map in that same channel, row
   * and column. The dimensions of the map are assumed to align with the
   * dimensions of the calling chain.
   * @param to the chain which elements will be mapped into
   * @param map the chain which specifies where elements will go
   */
  void mapTo(Chain<T> *to, Chain<struct ChainIndex> *map);

  /**
   * Map the "from" chain into the calling chain using indicies in the map. The
   * calling chain element in a particular channel, row, and column is assigned
   * the element of the from chain whose index is specified by the element of
   * the map in that same channel, row and column. The dimensions of the map
   * are assumed to align with the dimensions of the calling chain.
   * @param from the chain which elements will be mapped from
   * @param map the chain which specifies where elements will come from
   */
  void mapFrom(Chain<T> *from, Chain<struct ChainIndex> *map);

  /**
   * Creates a blueprint for a distributed mapTo operation. Can only be called
   * on a map class.
   * @param startChannel the ending channel for the elements of the map that
   *          this rank is responsible for.
   * @param endChannel the ending channel for the elements of the map that this
   *          rank is responsible for.
   * @param toStartChannels An array of starting channels of the destination
   *          Chain that each rank is responsible for.
   * @param toEndChannels An array of starting channels of the destination
   *          Chain that each rank is responsible for.
   * @param comm the communicator for the mapTo operation.
   * @return a pointer to a struct containing the necessary information for a
   *           distributed mapTo operation
   */
  MapDistributedData* mapToDistributedCreate(std::size_t startChannel, std::size_t endChannel, std::size_t *toStartChannels, std::size_t *toEndChannels, MPI_Comm comm);

  // Destroy to match create
  void mapToDistributedDestroy(MapDistributedData* data);

  /**
   * Map the calling chain into the "to" chain using indicies in the map. The
   * calling chain element in a particular channel, row, and column is mapped
   * to the index and processor specified by "data". The dimensions of the map
   * used to create "data" are assumed to align with the dimensions of the
   * calling chain.
   * @param to the chain which elements will be mapped into
   * @param data a struct specifying where the elements will go
   */
  void mapToDistributed(Chain<T> *to, MapDistributedData *data);

  /**
   * Creates a blueprint for a distributed mapFrom operation. Can only be called
   * on a map class.
   * @param startChannel the ending channel for the elements of the map that
   *          this rank is responsible for.
   * @param endChannel the ending channel for the elements of the map that this
   *          rank is responsible for.
   * @param toStartChannels An array of starting channels of the source Chain
   *          that each rank is responsible for.
   * @param toEndChannels An array of ending channels of the source Chain
   *          that each rank is responsible for.
   * @param comm the communicator for the mapFrom operation.
   * @return a pointer to a struct containing the necessary information for a
   *           distributed mapFrom operation
   */
  MapDistributedData* mapFromDistributedCreate(std::size_t startChannel, std::size_t endChannel, std::size_t *fromStartChannels, std::size_t *fromEndChannels, MPI_Comm comm);

  // Destroy to match create
  void mapFromDistributedDestroy(MapDistributedData* data);

  /**
   * Map the "from" chain into the calling chain using indicies in the map. The
   * calling chain element in a particular channel, row, and column is assigned
   * the element of the from chain whose index  and processor are specified by
   * "data". The dimensions of the map used to create "data" are assumed to
   * align with the dimensions of the calling chain.
   * @param from the chain which elements will be mapped from
   * @param data a struct specifying where the elements come from
   */
  void mapFromDistributed(Chain<T> *from, MapDistributedData *data);

  /**
   * Update copies of a chain on all ranks with regions from each rank.
   * @param startChannels The starting channels of valid regions on each rank.
   * @param endChannels The ending channels of valid regions on each rank.
   * @param comm the communicator for this operation.
   */
  void updateDistributed(std::size_t *startChannels, std::size_t *endChannels, MPI_Comm comm);
};

#include <cstddef>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <type_traits>

template <typename T>
std::size_t Chain<T>::ChainMemory(std::size_t startChannel, std::size_t endChannel, std::size_t nChannels, std::size_t *nRowss, std::size_t *nColumnss){
  #ifdef DEBUG
  if (startChannel > nChannels){
    std::cerr << "error: startChannel not part of chain (startChannel > nChannels)" << std::endl;
    exit(1);
  }
  if (endChannel > nChannels){
    std::cerr << "error: endChannel not part of chain (endChannel > nChannels)" << std::endl;
    exit(1);
  }
  if (startChannel > endChannel){
    std::cerr << "error: endChannel not part of chain (startChannel > endChannel)" << std::endl;
    exit(1);
  }
  #endif
  std::size_t myNChannels = endChannel - startChannel;
  if(!myNChannels){
    return 0;
  }
  std::size_t size = sizeof(Chain<T>);
  size += sizeof(std::size_t) * (myNChannels + 1);
  size += sizeof(std::size_t) * myNChannels;
  size += sizeof(std::size_t) * myNChannels;
  size += sizeof(T**) * myNChannels;
  size += sizeof(T*) * myNChannels;

  for(std::size_t i = 0; i < myNChannels; i++){
    if(nRowss[i] && nColumnss[i]){
      size += sizeof(T*) * nRowss[i];
      size += sizeof(T) * nRowss[i] * nColumnss[i];
    }
  }

  return size;
}

template <typename T>
Chain<T>::Chain(std::size_t startChannel, std::size_t endChannel, std::size_t nChannels, std::size_t *nRowss, std::size_t *nColumnss){
  #ifdef DEBUG
  if (startChannel > nChannels){
    std::cerr << "error: startChannel not part of chain (startChannel > nChannels)" << std::endl;
    exit(1);
  }
  if (endChannel > nChannels){
    std::cerr << "error: endChannel not part of chain (endChannel > nChannels)" << std::endl;
    exit(1);
  }
  if (startChannel > endChannel){
    std::cerr << "error: endChannel not part of chain (startChannel > endChannel)" << std::endl;
    exit(1);
  }
  #endif
  this->startChannel = startChannel;
  this->endChannel = endChannel;
  this->nChannels = nChannels;
  this->myNChannels = endChannel - startChannel;
  if(!this->myNChannels){
    return;
  }
  this->offsets = new std::size_t[myNChannels + 1];
  this->offsets[0] = 0;
  this->nRowss = new std::size_t[myNChannels];
  this->nColumnss = new std::size_t[myNChannels];
  this->elements = new T**[myNChannels];
  this->blocks = new T*[myNChannels];

  this->maxBufferSize = 0;
  std::memcpy(this->nRowss, nRowss, sizeof(std::size_t) * this->myNChannels);
  std::memcpy(this->nColumnss, nColumnss, sizeof(std::size_t) * this->myNChannels);
  for(std::size_t i = 0; i < this->myNChannels; i++){
    this->offsets[i + 1] = this->offsets[i] + this->nRowss[i] * this->nColumnss[i];
    if(this->nRowss[i] * this->nColumnss[i] > this->maxBufferSize){
      this->maxBufferSize = this->nRowss[i] * this->nColumnss[i];
    }
  }

  this->data = new T[this->offsets[this->myNChannels]];
  for(std::size_t i = 0; i < this->myNChannels; i++){
    if(this->nRowss[i] && this->nColumnss[i]){
      this->elements[i] = new T*[this->nRowss[i]];
      this->blocks[i] = this->data + this->offsets[i];
      for(std::size_t j = 0; j < this->nRowss[i]; j++){
        this->elements[i][j] = this->data + this->offsets[i] + j * this->nColumnss[i];
      }
    }else{
      this->elements[i] = NULL;
      this->blocks[i] = NULL;
    }
  }
}

template <typename T>
Chain<T>::~Chain(){
  if(this->myNChannels){
    delete[] this->offsets;
    delete[] this->nRowss;
    delete[] this->nColumnss;
    for(std::size_t i = 0; i < this->myNChannels; i++){
      if(this->elements[i]){
        delete[] this->elements[i];
      }
    }
    delete[] this->elements;
    delete[] this->blocks;
    delete[] this->data;
  }
}

template <typename T>
void Chain<T>::zero(std::size_t channel){
  #ifdef DEBUG
  if (channel < this->startChannel){
    std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
    exit(1);
  }
  if (channel >= this->endChannel){
    std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
    exit(1);
  }
  #endif
  std::memset(this->data + this->offsets[channel - this->startChannel],
              0,
              sizeof(T) * this->nRowss[channel - this->startChannel]
                        * this->nColumnss[channel - this->startChannel]);
}

template <typename T>
void Chain<T>::zero(void){
  if(myNChannels){
    std::memset(this->data,
                0,
                sizeof(T) * this->offsets[this->endChannel - this->startChannel]);
  }
}

template <typename T>
void Chain<T>::set(std::size_t channel, Chain *other){
  #ifdef DEBUG
  if (channel < this->startChannel){
    std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
    exit(1);
  }
  if (channel >= this->endChannel){
    std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
    exit(1);
  }
  if (channel < other->startChannel){
    std::cerr << "error: channel not part of other chain (channel < startChannel)" << std::endl;
    exit(1);
  }
  if (channel >= other->endChannel){
    std::cerr << "error: channel not part of other chain (channel >= endChannel)" << std::endl;
    exit(1);
  }
  if (this->nRowss[channel - this->startChannel] != other->nRowss[channel - other->startChannel]
      || this->nRowss[channel - this->startChannel] != other->nRowss[channel - other->startChannel]){
    std::cerr << "error: channel size does not match other channel" << std::endl;
    exit(1);
  }
  #endif
  std::memcpy(this->data + this->offsets[channel - this->startChannel],
              other->data + other->offsets[channel - other->startChannel],
              sizeof(T) * this->nRowss[channel - this->startChannel]
                        * this->nColumnss[channel - this->startChannel]);
}

template <typename T>
void Chain<T>::set(Chain *other){
  #ifdef DEBUG
  if (this->startChannel < other->startChannel){
    std::cerr << "error: startChannel not part of other chain" << std::endl;
    exit(1);
  }
  if (this->endChannel > other->endChannel){
    std::cerr << "error: endChannel not part of other chain" << std::endl;
    exit(1);
  }
  for(std::size_t channel = this->startChannel; channel < this->endChannel; channel++){
    if (this->nRowss[channel - this->startChannel] != other->nRowss[channel - other->startChannel]
        || this->nRowss[channel - this->startChannel] != other->nRowss[channel - other->startChannel]){
      std::cerr << "error: channel size does not match other channel" << std::endl;
      exit(1);
    }
  }
  #endif
  if(myNChannels){
    std::memcpy(this->data,
                other->data + other->offsets[this->startChannel - other->startChannel],
                sizeof(T) * this->offsets[this->endChannel - this->startChannel]);
  }
}

template <typename T>
void Chain<T>::copy(std::size_t channel, T *buffer){
  #ifdef DEBUG
  if (channel < this->startChannel){
    std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
    exit(1);
  }
  if (channel >= this->endChannel){
    std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
    exit(1);
  }
  #endif
  std::memcpy(buffer,
              this->data + this->offsets[channel - this->startChannel],
              sizeof(std::size_t) * this->nRowss[channel - this->startChannel]
                                  * this->nColumnss[channel - this->startChannel]);
}

template <typename T>
void Chain<T>::print(std::size_t channel){
  #ifdef DEBUG
  if (channel < this->startChannel){
    std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
    exit(1);
  }
  if (channel >= this->endChannel){
    std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
    exit(1);
  }
  #endif
  std::cout << std::setprecision(4);
  for(std::size_t j = 0; j < this->nRowss[channel - this->myNChannels]; j++){
    for(std::size_t k = 0; k < this->nColumnss[channel - this->myNChannels]; k++){
      std::cout << std::setw(8) << this->get(channel, j, k) << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
void Chain<T>::permute(std::size_t *channels){
  #ifdef DEBUG
  int permutedCheck[this->myNChannels];
  std::memset(permutedCheck, 0, sizeof(int) * this->myNChannels);
  for(std::size_t i = 0; i < this->myNChannels; i++){
    if (channels[i] < this->startChannel){
      std::cerr << "error: channel not part of chain (channel < startChannel)" << std::endl;
      exit(1);
    }
    if (channels[i] >= this->endChannel){
      std::cerr << "error: channel not part of chain (channel >= endChannel)" << std::endl;
      exit(1);
    }
    if(permutedCheck[channels[i] - this->startChannel]){
      std::cerr << "error: permutation not bijective" << std::endl;
      exit(1);
    }
    permutedCheck[channels[i] - this->startChannel] = 1;
  }
  #endif
  if(myNChannels){
    std::size_t *permutedNRowss = new std::size_t[this->myNChannels];
    std::size_t *permutedNColumnss = new std::size_t[this->myNChannels];
    std::size_t *permutedOffsets = new std::size_t[this->myNChannels + 1];
    T *permutedData = new T[this->offsets[this->myNChannels]];

    permutedOffsets[0] = 0;
    for(std::size_t i = 0; i < this->myNChannels; i++){
      permutedNRowss[i] = this->nRowss[channels[i] - this->startChannel];
      permutedNColumnss[i] = this->nColumnss[channels[i] - this->startChannel];
      permutedOffsets[i + 1] = permutedOffsets[i] + permutedNRowss[i] * permutedNColumnss[i];
      std::memcpy(permutedData + permutedOffsets[i],
                  this->data + this->offsets[channels[i] - this->startChannel],
                  sizeof(std::size_t) * this->nRowss[channels[i] - this->startChannel]
                                      * this->nColumnss[channels[i] - this->startChannel]);
    }

    delete[] this->data;
    this->data = permutedData;
    delete[] this->nRowss;
    this->nRowss = permutedNRowss;
    delete[] this->nColumnss;
    this->nColumnss = permutedNColumnss;
    delete[] this->offsets;
    this->offsets = permutedOffsets;

    for(std::size_t i = 0; i < this->myNChannels; i++){
      if(this->elements[i]){
        delete[] this->elements[i];
      }
    }
    for(std::size_t i = 0; i < this->myNChannels; i++){
      if(this->nRowss[i] && this->nColumnss[i]){
        this->elements[i] = new T*[this->nRowss[i]];
        this->blocks[i] = this->data + this->offsets[i];
        for(std::size_t j = 0; j < this->nRowss[i]; j++){
          this->elements[i][j] = this->data + this->offsets[i] + j * this->nColumnss[i];
        }
      }else{
        this->elements[i] = NULL;
        this->blocks[i] = NULL;
      }
    }
  }
}

template <typename T>
void Chain<T>::mapTo(Chain<T> *to, Chain<struct ChainIndex> *map){
  #ifdef DEBUG
  if(this->startChannel < map->startChannel || this->endChannel > map->endChannel){
    std::cerr << "error: map does not have necessary channels" << std::endl;
    exit(1);
  }
  for(std::size_t i = this->startChannel; i < this->endChannel; i++){
    if(this->nColumns(i) != map->nColumns(i) || this->nColumns(i) != map->nColumns(i)){
      std::cerr << "error: dimensions of from block do not match map block" << std::endl;
      exit(1);
    }
  }
  #endif
  for(std::size_t i = this->startChannel; i < this->endChannel; i++){
    for(std::size_t j = 0; j < this->nRows(i); j++){
      for(std::size_t k = 0; k < this->nColumns(i); k++){
        to->set(map->get(i, j, k), this->get(i, j, k));
      }
    }
  }
}

template <typename T>
void Chain<T>::mapFrom(Chain<T> *from, Chain<struct ChainIndex> *map){
  #ifdef DEBUG
  if(this->startChannel < map->startChannel || this->endChannel > map->endChannel){
    std::cerr << "error: map does not have necessary channels" << std::endl;
    exit(1);
  }
  for(std::size_t i = this->startChannel; i < this->endChannel; i++){
    if(this->nColumns(i) != map->nColumns(i) || this->nColumns(i) != map->nColumns(i)){
      std::cerr << "error: dimensions of to block do not match map block" << std::endl;
      exit(1);
    }
  }
  #endif
  for(std::size_t i = this->startChannel; i < this->endChannel; i++){
    for(std::size_t j = 0; j < this->nRows(i); j++){
      for(std::size_t k = 0; k < this->nColumns(i); k++){
        this->set(i, j, k, from->get(map->get(i, j, k)));
      }
    }
  }
}

struct MapDistributedData {
  MPI_Comm comm;
  int rank;
  int size;
  int *ranks;
  int *sendCounts;
  int *recvCounts;
  std::size_t maxChannel;
  std::size_t startChannel;
  std::size_t endChannel;
  std::size_t sendCount;
  std::size_t recvCount;
  int *sendOffsets;
  int *recvOffsets;
  ChainIndex *sendIndices;
  ChainIndex *recvIndices;
  Chain<ChainIndex> *map;

  ~MapDistributedData(){
    delete[] this->ranks;
    delete[] this->sendCounts;
    delete[] this->recvCounts;
    delete[] this->sendOffsets;
    delete[] this->recvOffsets;
    delete[] this->sendIndices;
    delete[] this->recvIndices;
  }
};

template<typename T>
void Chain<T>::mapFromDistributed(Chain<T> *from, MapDistributedData *data){
  T *sendElements = new T[data->sendCount];
  T *recvElements = new T[data->recvCount];
  int recvCounters[data->size];//should be size_t

  for(std::size_t i = 0; i < data->sendCount; i++){
    sendElements[i] = from->get(data->sendIndices[i]);
  }

  MPI_Datatype datatype;
  if(std::is_same<T, double>::value){
    datatype = MPI_DOUBLE;
  }
  if(std::is_same<T, unsigned short>::value){
    datatype = MPI_UNSIGNED_SHORT;
  }
  if(std::is_same<T, unsigned>::value){
    datatype = MPI_UNSIGNED;
  }
  if(std::is_same<T, unsigned long>::value){
    datatype = MPI_UNSIGNED_LONG;
  }
  if(std::is_same<T, unsigned long long>::value){
    datatype = MPI_LONG_LONG;
  }
  if(std::is_same<T, ChainIndex>::value){
    datatype = ChainIndex_MPI();
  }

  MPI_Alltoallv(sendElements, data->sendCounts, data->sendOffsets, datatype, recvElements, data->recvCounts, data->recvOffsets, datatype, data->comm);

  std::memcpy(recvCounters, data->recvOffsets, sizeof(int) * data->size); //should be size_t
  for(std::size_t channel = data->startChannel; channel < data->endChannel; channel++){
    for(std::size_t row = 0; row < this->nRows(channel); row++){
      for(std::size_t col = 0; col < this->nColumns(channel); col++){
        ChainIndex index = data->map->get(channel, row, col);
        this->set(channel, row, col, recvElements[recvCounters[data->ranks[index.channel]]]);
        recvCounters[data->ranks[index.channel]]++;
      }
    }
  }

  delete[] recvElements;
  delete[] sendElements;
}

template<typename T>
void Chain<T>::mapToDistributed(Chain<T> *to, MapDistributedData *data){
  T *sendElements = new T[data->sendCount];
  T *recvElements = new T[data->recvCount];
  int sendCounters[data->size];//should be size_t

  std::memcpy(sendCounters, data->sendOffsets, sizeof(int) * data->size); //should be size_t
  for(std::size_t channel = data->startChannel; channel < data->endChannel; channel++){
    for(std::size_t row = 0; row < this->nRows(channel); row++){
      for(std::size_t col = 0; col < this->nColumns(channel); col++){
        ChainIndex index = data->map->get(channel, row, col);
        sendElements[sendCounters[data->ranks[index.channel]]] = this->get(channel, row, col);
        sendCounters[data->ranks[index.channel]]++;
      }
    }
  }

  MPI_Datatype datatype;
  if(std::is_same<T, double>::value){
    datatype = MPI_DOUBLE;
  }
  if(std::is_same<T, unsigned short>::value){
    datatype = MPI_UNSIGNED_SHORT;
  }
  if(std::is_same<T, unsigned>::value){
    datatype = MPI_UNSIGNED;
  }
  if(std::is_same<T, unsigned long>::value){
    datatype = MPI_UNSIGNED_LONG;
  }
  if(std::is_same<T, unsigned long long>::value){
    datatype = MPI_LONG_LONG;
  }
  if(std::is_same<T, ChainIndex>::value){
    datatype = ChainIndex_MPI();
  }

  MPI_Alltoallv(sendElements, data->sendCounts, data->sendOffsets, datatype, recvElements, data->recvCounts, data->recvOffsets, datatype, data->comm);

  for(std::size_t i = 0; i < data->recvCount; i++){
    to->set(data->recvIndices[i], recvElements[i]);
  }

  delete[] recvElements;
  delete[] sendElements;
}

template <typename T>
void Chain<T>::updateDistributed(std::size_t *startChannels, std::size_t *endChannels, MPI_Comm comm){
  int rank;
  MPI_Comm_rank(comm, &rank);

  int size;
  MPI_Comm_size(comm, &size);

  MPI_Datatype datatype;
  if(std::is_same<T, double>::value){
    datatype = MPI_DOUBLE;
  }
  if(std::is_same<T, unsigned short>::value){
    datatype = MPI_UNSIGNED_SHORT;
  }
  if(std::is_same<T, unsigned>::value){
    datatype = MPI_UNSIGNED;
  }
  if(std::is_same<T, unsigned long>::value){
    datatype = MPI_UNSIGNED_LONG;
  }
  if(std::is_same<T, unsigned long long>::value){
    datatype = MPI_LONG_LONG;
  }
  if(std::is_same<T, ChainIndex>::value){
    datatype = ChainIndex_MPI();
  }

  int counts[this->nChannels];
  int displs[this->nChannels];
  for(int i = 0; i < size; i++){
    displs[i] = this->offsets[startChannels[i]];
    counts[i] = this->offsets[endChannels[i]] - displs[i];
  }

  T sendbuf[counts[rank]];
  for(std::size_t i = startChannels[rank]; i < endChannels[rank]; i++){
    this->copy(i, sendbuf + this->offsets[i] - this->offsets[startChannels[rank]]);
  }

  MPI_Allgatherv(sendbuf, counts[rank], datatype, this->data, counts, displs, datatype, comm);
}

#endif
