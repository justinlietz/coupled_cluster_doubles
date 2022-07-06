#include "Chain.hpp"

MPI_Datatype ChainIndex_MPI(){
  static int ChainIndexCreated = 0;
  static MPI_Datatype ChainIndexType = 0;
  if(!ChainIndexCreated){
    MPI_Type_contiguous(3, MPI_LONG_LONG, &ChainIndexType);
    MPI_Type_commit(&ChainIndexType);
    ChainIndexCreated = 1;
  }
  return ChainIndexType;
}

template<>
void Chain<ChainIndex>::mapFromDistributedDestroy(MapDistributedData* data){
  delete data;
}

template<>
void Chain<ChainIndex>::mapToDistributedDestroy(MapDistributedData* data){
  delete data;
}

std::size_t MDD_Memory(MapDistributedData *data){
  std::size_t nBytes = 0;

  std::size_t myNChannels = data->endChannel - data->startChannel;
  if(!myNChannels){
    // Potentially not a bug, but send to cerr for now.
    std::cerr << "error: MapDistributedData has myNChannels == 0)" << std::endl;
    return 0;
  }
   //1 comm
   nBytes += sizeof(data->comm);
   //2 int
   nBytes += 2*sizeof(data->rank);
   //5 int*
   nBytes += 5*sizeof(data->sendCounts);
   //5 size_t
   nBytes += 5*sizeof(data->startChannel);
   //2 ChainIndex*
   nBytes += 2*sizeof(data->sendIndices);
   //1 Chain<ChainIndex>
   nBytes += sizeof(data->map);

   // maybe make all of these size_t's
   // this->ranks;
   nBytes += data->maxChannel*sizeof(int);
   // this->sendCounts;
   nBytes += data->size*sizeof(int);
   // this->recvCounts;
   nBytes += data->size*sizeof(int);
   // this->sendOffsets;
   nBytes += data->size*sizeof(int);
   // this->recvOffsets;
   nBytes += data->size*sizeof(int);
   // this->sendIndices;
   nBytes += data->sendCount*sizeof(ChainIndex);
   // this->recvIndices;
   nBytes += data->recvCount*sizeof(ChainIndex);
   return nBytes;
}


template<>
MapDistributedData* Chain<ChainIndex>::mapFromDistributedCreate(std::size_t startChannel, std::size_t endChannel, std::size_t *fromStartChannels, std::size_t *fromEndChannels, MPI_Comm comm){
  MapDistributedData *data = new MapDistributedData;
  data->map = this;
  data->startChannel = startChannel;
  data->endChannel = endChannel;
  data->comm = comm;
  MPI_Comm_rank(comm, &data->rank);
  MPI_Comm_size(comm, &data->size);

  std::size_t fromMaxChannel = 0;
  for(int i = 0; i < data->size; i++){
    fromMaxChannel = std::max(fromEndChannels[i], fromMaxChannel);
  }
  data->maxChannel = fromMaxChannel;

  data->ranks = new int[fromMaxChannel];
  for(int i = 0; i < data->size; i++){
    for(std::size_t j = fromStartChannels[i]; j < fromEndChannels[i]; j++){
      data->ranks[j] = i;
    }
  }

  data->sendCounts = new int[data->size];//should be size_t
  data->recvCounts = new int[data->size];//should be size_t
  std::memset(data->recvCounts, 0, sizeof(int) * data->size);//should be size_t

  for(std::size_t channel = data->startChannel; channel < data->endChannel; channel++){
    for(std::size_t row = 0; row < this->nRows(channel); row++){
      for(std::size_t col = 0; col < this->nColumns(channel); col++){
        data->recvCounts[data->ranks[this->get(channel, row, col).channel]]++;
      }
    }
  }

  MPI_Alltoall(data->recvCounts, 1, MPI_INT, data->sendCounts, 1, MPI_INT, comm);//should be MPI_LONG_LONG

  data->sendCount = 0;
  data->recvCount = 0;

  for(int i = 0; i < data->size; i++){
    data->sendCount += data->sendCounts[i];
    data->recvCount += data->recvCounts[i];
  }

  data->sendOffsets = new int[data->size];//should be size_t
  data->recvOffsets = new int[data->size];//should be size_t
  data->sendOffsets[0] = 0;
  data->recvOffsets[0] = 0;
  for(int i = 1; i < data->size; i++){
    data->sendOffsets[i] = data->sendOffsets[i - 1] + data->sendCounts[i - 1];
    data->recvOffsets[i] = data->recvOffsets[i - 1] + data->recvCounts[i - 1];
  }

  data->sendIndices = new ChainIndex[data->sendCount];
  data->recvIndices = new ChainIndex[data->recvCount];
  int recvCounters[data->size];//should be size_t

  std::memcpy(recvCounters, data->recvOffsets, sizeof(int) * data->size); //should be size_t
  for(std::size_t channel = data->startChannel; channel < data->endChannel; channel++){
    for(std::size_t row = 0; row < this->nRows(channel); row++){
      for(std::size_t col = 0; col < this->nColumns(channel); col++){
        ChainIndex index = this->get(channel, row, col);
        data->recvIndices[recvCounters[data->ranks[index.channel]]] = index;
        recvCounters[data->ranks[index.channel]]++;
      }
    }
  }

  MPI_Alltoallv(data->recvIndices, data->recvCounts, data->recvOffsets, ChainIndex_MPI(), data->sendIndices, data->sendCounts, data->sendOffsets, ChainIndex_MPI(), data->comm);

  return data;
}

template<>
MapDistributedData* Chain<ChainIndex>::mapToDistributedCreate(std::size_t startChannel, std::size_t endChannel, std::size_t *toStartChannels, std::size_t *toEndChannels, MPI_Comm comm){
  MapDistributedData *data = new MapDistributedData;
  data->map = this;
  data->startChannel = startChannel;
  data->endChannel = endChannel;
  data->comm = comm;
  MPI_Comm_rank(comm, &data->rank);
  MPI_Comm_size(comm, &data->size);

  std::size_t toMaxChannel = 0;
  for(int i = 0; i < data->size; i++){
    toMaxChannel = std::max(toEndChannels[i], toMaxChannel);
  }
  data->maxChannel = toMaxChannel;

  data->ranks = new int[toMaxChannel];
  for(int i = 0; i < data->size; i++){
    for(std::size_t j = toStartChannels[i]; j < toEndChannels[i]; j++){
      data->ranks[j] = i;
    }
  }

  data->sendCounts = new int[data->size];//should be size_t
  data->recvCounts = new int[data->size];//should be size_t
  std::memset(data->sendCounts, 0, sizeof(int) * data->size);//should be size_t

  for(std::size_t channel = data->startChannel; channel < data->endChannel; channel++){
    for(std::size_t row = 0; row < this->nRows(channel); row++){
      for(std::size_t col = 0; col < this->nColumns(channel); col++){
        data->sendCounts[data->ranks[this->get(channel, row, col).channel]]++;
      }
    }
  }

  MPI_Alltoall(data->sendCounts, 1, MPI_INT, data->recvCounts, 1, MPI_INT, comm);//should be MPI_LONG_LONG

  data->sendCount = 0;
  data->recvCount = 0;

  for(int i = 0; i < data->size; i++){
    data->sendCount += data->sendCounts[i];
    data->recvCount += data->recvCounts[i];
  }

  data->sendOffsets = new int[data->size];//should be size_t
  data->recvOffsets = new int[data->size];//should be size_t
  data->sendOffsets[0] = 0;
  data->recvOffsets[0] = 0;
  for(int i = 1; i < data->size; i++){
    data->sendOffsets[i] = data->sendOffsets[i - 1] + data->sendCounts[i - 1];
    data->recvOffsets[i] = data->recvOffsets[i - 1] + data->recvCounts[i - 1];
  }

  data->sendIndices = new ChainIndex[data->sendCount];
  data->recvIndices = new ChainIndex[data->recvCount];
  int sendCounters[data->size];//should be size_t

  std::memcpy(sendCounters, data->sendOffsets, sizeof(int) * data->size); //should be size_t
  for(std::size_t channel = data->startChannel; channel < data->endChannel; channel++){
    for(std::size_t row = 0; row < this->nRows(channel); row++){
      for(std::size_t col = 0; col < this->nColumns(channel); col++){
        ChainIndex index = this->get(channel, row, col);
        data->sendIndices[sendCounters[data->ranks[index.channel]]] = index;
        sendCounters[data->ranks[index.channel]]++;
      }
    }
  }

  MPI_Alltoallv(data->sendIndices, data->sendCounts, data->sendOffsets, ChainIndex_MPI(), data->recvIndices, data->recvCounts, data->recvOffsets, ChainIndex_MPI(), data->comm);

  return data;
}

