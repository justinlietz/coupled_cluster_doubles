add_library (src SPBasis.cpp MBPT2Corr.cpp balance.cpp Chain.cpp ElectronGasSPBasis.cpp InfMatterSPBasis.cpp PairingSPBasis.cpp load.cpp calc.cpp util.cpp)

add_executable(ccd.exe parse.cpp main.cpp)

target_link_libraries(ccd.exe src)
